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
        self.lstm_proj = nn.Linear(mlp_hidden * 2, mlp_hidden)
        self.v_att = Attention(mlp_hidden, mlp_hidden, mlp_hidden)


    def forward(self, img, q):
        batch_size = q.size(0)
        w_emb = q
        lstm_out, (h, _) = self.q_emb(w_emb)
        q_emb = h.permute(1, 0, 2).contiguous().view(batch_size, -1)
        q_emb = self.lstm_proj(q_emb)

        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1) # [batch, v_dim]

        return lstm_out, q_emb, v_emb

class RoleQHandler(nn.Module):
    def __init__(self):
        super(RoleQHandler, self).__init__()

        self.vqa_model = TopDown()



    def forward(self, img, q):
        lstm_out, q_emb , v_emb = self.vqa_model(img, q)
        return lstm_out, q_emb , v_emb

class LabelHandler(nn.Module):
    def __init__(self,
                 label_embedding,
                 embed_hidden=300,
                 mlp_hidden=512):
        super(LabelHandler, self).__init__()

        self.label_embedding = label_embedding
        self.v_att = Attention(mlp_hidden, embed_hidden, mlp_hidden)
        self.vocab_size = self.label_embedding.weight.size(0)
        self.mlp_hidden =mlp_hidden
        self.embed_hidden = embed_hidden


    def forward(self, img):
        batch_size = img.size(0)
        img = img.expand(self.vocab_size, img.size(0), img.size(1), img.size(2))
        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size * self.vocab_size, -1, self.mlp_hidden)

        print('label hand : img ', img.size())

        label_embd = self.label_embedding.weight
        label_embd = label_embd.expand(batch_size, label_embd.size(0), label_embd.size(1))
        label_embd = label_embd.contiguous().view(batch_size * self.vocab_size, self.embed_hidden)

        print('label hand : label :', label_embd.size())

        att = self.v_att(img, label_embd)
        v_emb = (att * img).sum(1) # [batch, v_dim]
        print('final label hand embd :', v_emb.size())

        return v_emb

class Role2LabelHandler(nn.Module):
    def __init__(self,
                 label_embedding,
                 embed_hidden=300,
                 mlp_hidden=512):
        super(Role2LabelHandler, self).__init__()

        self.label_embedding = label_embedding
        self.q_word_project = nn.Linear(mlp_hidden * 2, mlp_hidden)
        self.v_att = Attention(mlp_hidden, embed_hidden, mlp_hidden)
        self.vocab_size = self.label_embedding.weight.size(0)
        self.mlp_hidden =mlp_hidden
        self.embed_hidden = embed_hidden


    def forward(self, q):
        batch_size = q.size(0)
        q = self.q_word_project(q)
        q = q.expand(self.vocab_size, q.size(0), q.size(1), q.size(2))
        q = q.transpose(0,1)
        q = q.contiguous().view(batch_size * self.vocab_size, -1, self.mlp_hidden)
        print('q->a hand : q ', q.size())

        label_embd = self.label_embedding.weight
        label_embd = label_embd.expand(batch_size, label_embd.size(0), label_embd.size(1))
        label_embd = label_embd.contiguous().view(batch_size * self.vocab_size, self.embed_hidden)
        print('label hand : label :', label_embd.size())

        att = self.v_att(q, label_embd)
        q_emb = (att * q).sum(1) # [batch, v_dim]
        print('final label hand embd :', q_emb.size())
        return q_emb


class ImSituationHandler(nn.Module):
    def __init__(self,
                 encoder,
                 qword_embeddings,
                 label_embedding,
                 vocab_size,
                 gpu_mode,
                 mlp_hidden=512,
                 embed_hidden=300):
        super(ImSituationHandler, self).__init__()


        self.encoder = encoder
        self.qword_embeddings = qword_embeddings
        self.label_embedding = label_embedding
        self.vocab_size = vocab_size
        self.gpu_mode = gpu_mode
        self.img_q_handler = RoleQHandler()
        self.img_label_handler = LabelHandler(self.label_embedding)
        self.q_label_handler = Role2LabelHandler(self.label_embedding)
        self.c_net = FCNet([mlp_hidden*3, mlp_hidden])
        self.q_net = FCNet([mlp_hidden, mlp_hidden])
        self.a_net = FCNet([embed_hidden, mlp_hidden])
        self.sim_scorer = SimpleClassifier(
            mlp_hidden, mlp_hidden, 1, 0.5)

        self.mlp_hidden = mlp_hidden

    def forward(self, img, verb):
        batch_size = verb.size(0)

        role_qs, _ = self.encoder.get_role_questions_batch(verb)
        max_role_count = role_qs.size(1)
        #roles = self.encoder.get_role_ids_batch(verb)

        if self.gpu_mode >= 0:
            role_qs = role_qs.to(torch.device('cuda'))
            #roles = roles.to(torch.device('cuda'))
        role_qs = role_qs.view(-1, role_qs.size(-1))

        qword_embd, q_emb, vq_emb = self.img_q_handler(img, self.qword_embeddings(role_qs))#expand to vocab size
        print('out from img q:', qword_embd.size(), q_emb.size(), vq_emb.size())
        va_emb = self.img_label_handler(img)
        ans_emb = self.label_embedding.weight
        qa_emb = self.q_label_handler(qword_embd)

        #expand q and qv
        vq_emb = vq_emb.expand(self.vocab_size, vq_emb.size(0), vq_emb.size(1))
        vq_emb = vq_emb.transpose(0,1)
        vq_emb = vq_emb.contiguous().view(va_emb.size(0), self.mlp_hidden)

        ans_suit = torch.cat([vq_emb, va_emb, qa_emb],1)

        context_repr = self.c_net(ans_suit)
        q_repr = self.q_net(q_emb)

        q_repr = q_repr.expand(self.vocab_size, q_repr.size(0), q_repr.size(1))
        q_repr = q_repr.transpose(0,1)
        q_repr = q_repr.contiguous().view(va_emb.size(0), self.mlp_hidden)

        a_repr = self.a_net(ans_emb)

        a_repr = a_repr.expand(batch_size*max_role_count, a_repr.size(0), a_repr.size(1))
        a_repr = a_repr.contiguous().view(va_emb.size(0), self.mlp_hidden)

        joint_repr = context_repr* q_repr * a_repr

        print('join rep size :', joint_repr.size())
        sim_score = self.sim_scorer(joint_repr)
        print('sim_score :', sim_score.size())
        logits = sim_score.view(-1, self.vocab_size)
        print('logits :', logits.size(), logits[0])

        return logits


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
        self.vocab_size = 50
        self.max_role_count = self.encoder.get_max_role_count()
        self.n_role_q_vocab = len(self.encoder.question_words)

        self.conv = vgg16_modified()

        #self.role_lookup = nn.Embedding(self.n_roles +1, embed_hidden, padding_idx=self.n_roles)
        self.ans_lookup = nn.Embedding(self.vocab_size, embed_hidden)
        self.w_emb = nn.Embedding(self.n_role_q_vocab + 1, embed_hidden, padding_idx=self.n_role_q_vocab)

        self.vsrl_model = ImSituationHandler(self.encoder, self.w_emb, self.ans_lookup, self.vocab_size,
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
        batch_size, n_channel, conv_h, conv_w = img_features.size()
        img = img_features.view(batch_size, n_channel, -1)
        img = img.permute(0, 2, 1)

        img = img.expand(self.max_role_count,img.size(0), img.size(1), img.size(2))
        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size* self.max_role_count, -1, self.mlp_hidden)

        role_pred = self.vsrl_model(img, verb)
        role_pred = role_pred.contiguous().view(batch_size, -1, self.vocab_size)

        #print('ans sizes :', verb_pred.size(), role_pred.size())

        return role_pred

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

