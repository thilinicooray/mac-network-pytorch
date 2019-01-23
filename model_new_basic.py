import torch
import torch.nn as nn
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet
import torch.nn.functional as F
import torchvision as tv
import utils
import numpy as np

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1).sqrt()
    #X = torch.div(X, norm.expand_as(X))
    X = torch.div(X, norm.unsqueeze(1).expand_as(X))
    return X

class vgg16_modified(nn.Module):
    def __init__(self):
        super(vgg16_modified, self).__init__()
        vgg = tv.models.vgg16(pretrained=True)
        self.vgg_features = vgg.features

    def rep_size(self):
        return 1024

    def base_size(self):
        return 512

    def forward(self,x):
        #return self.dropout2(self.relu2(self.lin2(self.dropout1(self.relu1(self.lin1(self.vgg_features(x).view(-1, 512*7*7)))))))
        features = self.vgg_features(x)

        return features

class TopDown(nn.Module):
    def __init__(self,
                 embed_hidden=300,
                 mlp_hidden=512):
        super(TopDown, self).__init__()

        self.q_emb = nn.LSTM(embed_hidden, mlp_hidden,
                             batch_first=True, bidirectional=True)
        self.q_prep = FCNet([mlp_hidden, mlp_hidden])
        self.lstm_proj = nn.Linear(mlp_hidden * 2, mlp_hidden)
        self.verb_transform = nn.Linear(embed_hidden, mlp_hidden)
        self.v_att = Attention(mlp_hidden, mlp_hidden, mlp_hidden)


    def forward(self, img, q):
        batch_size = q.size(0)
        w_emb = q
        lstm_out, (h, _) = self.q_emb(w_emb)
        q_emb = h.permute(1, 0, 2).contiguous().view(batch_size, -1)
        q_emb = self.lstm_proj(q_emb)

        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1) # [batch, v_dim]

        return q_emb, v_emb

class RoleQHandler(nn.Module):
    def __init__(self):
        super(RoleQHandler, self).__init__()

        self.vqa_model = TopDown()



    def forward(self, img, q):
        q_emb , v_emb = self.vqa_model(img, q)
        return q_emb , v_emb


class ImSituationHandler(nn.Module):
    def __init__(self,
                 encoder,
                 qword_embeddings,
                 vocab_size,
                 gpu_mode,
                 mlp_hidden=512):
        super(ImSituationHandler, self).__init__()


        self.encoder = encoder
        self.qword_embeddings = qword_embeddings
        self.vocab_size = vocab_size
        self.gpu_mode = gpu_mode
        self.role_handler = RoleQHandler()
        self.q_net = FCNet([mlp_hidden, mlp_hidden])
        self.v_net = FCNet([mlp_hidden, mlp_hidden])
        self.classifier = SimpleClassifier(
            mlp_hidden, 2 * mlp_hidden, self.vocab_size, 0.5)
        self.mlp_hidden = mlp_hidden

        self.g = nn.Sequential(
            nn.Linear(mlp_hidden*2, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
        )

        self.f = nn.Sequential(
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden)
        )



    def forward(self, img, verb):
        batch_size = verb.size(0)

        role_qs, _ = self.encoder.get_role_questions_batch(verb)
        max_role_count = role_qs.size(1)
        #roles = self.encoder.get_role_ids_batch(verb)

        if self.gpu_mode >= 0:
            role_qs = role_qs.to(torch.device('cuda'))
            #roles = roles.to(torch.device('cuda'))
        role_qs = role_qs.view(-1, role_qs.size(-1))

        q_emb, v_emb = self.role_handler(img, self.qword_embeddings(role_qs))

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)

        #get a relational module to get relationship among 6 roles and get the final rep, assumkng it should equal back to the image
        reshaped_joint_rep = joint_repr.contiguous().view(batch_size, -1, self.mlp_hidden)
        rolerep1 = reshaped_joint_rep.unsqueeze(1).expand(batch_size, max_role_count, max_role_count, self.mlp_hidden)
        rolerep2 = reshaped_joint_rep.unsqueeze(2).expand(batch_size, max_role_count, max_role_count, self.mlp_hidden)
        rolerep1 = rolerep1.contiguous().view(-1, max_role_count * max_role_count, self.mlp_hidden)
        rolerep2 = rolerep2.contiguous().view(-1, max_role_count * max_role_count, self.mlp_hidden)

        concat_vec = torch.cat([rolerep1, rolerep2], 2).view(-1, self.mlp_hidden*2)
        g = self.g(concat_vec)
        g = g.view(-1, max_role_count * max_role_count, self.mlp_hidden).sum(1).squeeze()

        recreated_img = l2norm(self.f(g))

        return logits, recreated_img


class BaseModel(nn.Module):
    def __init__(self, encoder,
                 gpu_mode,
                 embed_hidden=300,
                 mlp_hidden=512):
        super(BaseModel, self).__init__()

        self.normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.train_transform = tv.transforms.Compose([
            tv.transforms.RandomRotation(10),
            tv.transforms.RandomResizedCrop(224),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            self.normalize,
        ])

        self.dev_transform = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            self.normalize,
        ])

        self.encoder = encoder
        self.gpu_mode = gpu_mode
        self.n_roles = self.encoder.get_num_roles()
        self.n_verbs = self.encoder.get_num_verbs()
        self.vocab_size = self.encoder.get_num_labels()
        self.max_role_count = self.encoder.get_max_role_count()
        self.n_role_q_vocab = len(self.encoder.question_words)

        self.conv = vgg16_modified()
        self.img_serialize = nn.Sequential(
            nn.Linear(mlp_hidden*7*7, mlp_hidden*2),
            nn.BatchNorm1d(mlp_hidden*2),
            nn.ReLU(),
            nn.Linear(mlp_hidden*2, mlp_hidden)
        )
        self.role_lookup = nn.Embedding(self.n_roles +1, embed_hidden, padding_idx=self.n_roles)
        self.ans_lookup = nn.Embedding(self.vocab_size + 1, embed_hidden, padding_idx=self.vocab_size)
        self.w_emb = nn.Embedding(self.n_role_q_vocab + 1, embed_hidden, padding_idx=self.n_role_q_vocab)

        self.vsrl_model = ImSituationHandler(self.encoder, self.w_emb, self.vocab_size,
                                             self.gpu_mode)

        self.conv_hidden = self.conv.base_size()
        self.mlp_hidden = mlp_hidden
        self.embed_hidden = embed_hidden

    def train_preprocess(self):
        return self.train_transform

    def dev_preprocess(self):
        return self.dev_transform

    def forward(self, img, verb):

        img_features = self.conv(img)
        ser_img = l2norm(self.img_serialize(img_features.view(-1, 512*7*7)))
        batch_size, n_channel, conv_h, conv_w = img_features.size()
        img = img_features.view(batch_size, n_channel, -1)
        img = img.permute(0, 2, 1)

        img = img.expand(self.max_role_count,img.size(0), img.size(1), img.size(2))
        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size* self.max_role_count, -1, self.mlp_hidden)

        role_pred, recreated_img = self.vsrl_model(img, verb)
        role_pred = role_pred.contiguous().view(batch_size, -1, self.vocab_size)

        #print('ans sizes :', verb_pred.size(), role_pred.size())

        return role_pred, ser_img, recreated_img

    def calculate_loss(self, gt_verbs, role_label_pred, gt_labels,args):

        batch_size = role_label_pred.size()[0]
        if args.train_all:
            loss = 0
            for i in range(batch_size):
                for index in range(gt_labels.size()[1]):
                    frame_loss = 0
                    #verb_loss = utils.cross_entropy_loss(verb_pred[i], gt_verbs[i])
                    #frame_loss = criterion(role_label_pred[i], gt_labels[i,index])
                    for j in range(0, self.max_role_count):
                        frame_loss += utils.cross_entropy_loss(role_label_pred[i][j], gt_labels[i,index,j] ,self.vocab_size)
                    frame_loss = frame_loss/len(self.encoder.verb2_role_dict[self.encoder.verb_list[gt_verbs[i]]])
                    #print('frame loss', frame_loss, 'verb loss', verb_loss)
                    loss += frame_loss
        else:
            #verb from pre-trained
            loss = 0
            for i in range(batch_size):
                for index in range(gt_labels.size()[1]):
                    frame_loss = 0
                    #verb_loss = utils.cross_entropy_loss(verb_pred[i], gt_verbs[i])
                    #frame_loss = criterion(role_label_pred[i], gt_labels[i,index])
                    for j in range(0, self.max_role_count):
                        frame_loss += utils.cross_entropy_loss(role_label_pred[i][j], gt_labels[i,index,j] ,self.vocab_size)
                    frame_loss = frame_loss/len(self.encoder.verb2_role_dict[self.encoder.verb_list[gt_verbs[i]]])
                    #print('frame loss', frame_loss, 'verb loss', verb_loss)
                    loss += frame_loss


        final_loss = loss/batch_size
        #print('loss :', final_loss)
        return final_loss

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=True):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = self.order_sim
        else:
            self.sim = self.cosine_sim

        self.max_violation = max_violation

    def cosine_sim(self, im, s):
        """Cosine similarity between all the image and sentence pairs
        """
        return im.mm(s.t())


    def order_sim(self, im, s):
        """Order embeddings similarity measure $max(0, s-im)$
        """
        YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
               - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
        # score = -YmX.clamp(min=0).pow(2).sum(2).squeeze(2).sqrt().t()
        score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
        return score

    '''
    For single-modal retrieval, emb1=query, emb2=data
    For cross-modal retrieval, emb1=query in source domain, emb2=data in target domain
    '''
    def forward(self, emb1, emb2):
        # compute image-sentence score matrix
        scores = self.sim(emb1, emb2)
        diagonal = scores.diag().view(emb1.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = mask
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


