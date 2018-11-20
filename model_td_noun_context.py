import torch
import torch.nn as nn
from attention import Attention, NewAttention, RoleWeightAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet
import torch.nn.functional as F
import torchvision as tv
import utils


class vgg16_modified(nn.Module):
    def __init__(self):
        super(vgg16_modified, self).__init__()
        vgg = tv.models.vgg16(pretrained=True)
        self.vgg_features = vgg.features
        self.out_features = vgg.classifier[6].in_features
        features = list(vgg.classifier.children())[:-1] # Remove last layer
        self.vgg_classifier = nn.Sequential(*features) # Replace the model classifier
        #print(self.vgg_classifier)

    def rep_size(self):
        return 1024

    def base_size(self):
        return 512

    def forward(self,x):
        #return self.dropout2(self.relu2(self.lin2(self.dropout1(self.relu1(self.lin1(self.vgg_features(x).view(-1, 512*7*7)))))))
        features = self.vgg_features(x)
        y =  self.vgg_classifier(features.view(-1, 512*7*7))
        #print('y size :',  y.size())
        return features, y

class TopDownWithContext(nn.Module):
    def __init__(self, encoder,
                 embed_hidden=300,
                 mlp_hidden=512,
                 time_steps=3):
        super(TopDownWithContext, self).__init__()

        self.encoder = encoder
        self.n_verbs = self.encoder.get_num_verbs()
        self.vocab_size = self.encoder.get_num_labels()
        self.max_role_count = self.encoder.get_max_role_count()
        self.n_role_q_vocab = len(self.encoder.question_words)

        self.w_emb = nn.Embedding(self.n_role_q_vocab + 1, embed_hidden, padding_idx=self.n_role_q_vocab)
        self.q_emb = nn.LSTM(embed_hidden, mlp_hidden,
                             batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(mlp_hidden * 2, mlp_hidden)
        self.v_att = Attention(mlp_hidden, mlp_hidden, mlp_hidden)
        self.ctx_att = Attention(mlp_hidden, mlp_hidden, mlp_hidden)
        self.context = FCNet([mlp_hidden*2, mlp_hidden])
        #self.role_weight = RoleWeightAttention(mlp_hidden, mlp_hidden, mlp_hidden)
        self.detailedq = FCNet([mlp_hidden*2, mlp_hidden])
        self.concat = FCNet([mlp_hidden*2, mlp_hidden])
        self.q_net = FCNet([mlp_hidden, mlp_hidden])
        self.v_net = FCNet([mlp_hidden, mlp_hidden])
        self.classifier = SimpleClassifier(
            mlp_hidden, 2 * mlp_hidden, self.vocab_size, 0.5)
        self.time_steps = time_steps

    def forward(self, img_feat, questions, mask):
        batch_size = questions.size(0)
        w_emb = self.w_emb(questions)
        lstm_out, (h, _) = self.q_emb(w_emb)
        q_emb = h.permute(1, 0, 2).contiguous().view(batch_size, -1)
        q_emb = self.lstm_proj(q_emb)

        att = self.v_att(img_feat, q_emb)
        v_emb = (att * img_feat).sum(1)

        current_results = [v_emb]

        #use context made of surrounding roles
        for t in range(self.time_steps-1):
            role_label = torch.cat([q_emb,current_results[-1]],1)
            context = self.context(role_label)
            context = context.view(-1, mask.size(1), context.size(-1))
            context_updated = context.unsqueeze(0)
            context_updated = context_updated.expand(mask.size(1), context.size(0), context.size(1), context.size(2))
            context = context_updated.transpose(0,1)
            masked_context = mask * context
            #role_weights = self.role_weight(masked_context, q_emb)
            #final_context = (role_weights * masked_context).sum(2) #get weighted sum
            final_context = masked_context.sum(2)
            final_context = final_context.view(-1, final_context.size(-1))
            #gets context relavant features
            att = self.ctx_att(img_feat, final_context)
            v_new = att * img_feat
            #detailed_q = torch.cat([final_context, q_emb],1)
            #projectedq = self.detailedq(detailed_q)
            att = self.v_att(v_new, q_emb)
            v_emb = (att * v_new).sum(1)

            v_emb = self.concat(torch.cat([v_emb, current_results[-1]], 1))
            #todo:gating and dense connections


            current_results.append(v_emb)

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(current_results[-1])
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)

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
        self.max_role_count = self.encoder.get_max_role_count()
        self.vocab_size = self.encoder.get_num_labels()
        self.gpu_mode = gpu_mode

        self.conv = vgg16_modified()
        self.role_labeller = TopDownWithContext(self.encoder)

        self.conv_hidden = self.conv.base_size()
        self.mlp_hidden = mlp_hidden
        self.embed_hidden = embed_hidden

    def train_preprocess(self):
        return self.train_transform

    def dev_preprocess(self):
        return self.dev_transform

    def forward(self, img, verb, role_q):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        #print('role size ', role_q.size())

        img_features, conv = self.conv(img)
        batch_size, n_channel, conv_h, conv_w = img_features.size()
        role_q = role_q.view(batch_size*self.max_role_count, -1)
        img = img_features.view(batch_size, n_channel, -1)
        img = img.permute(0, 2, 1)

        img = img.expand(self.max_role_count,img.size(0), img.size(1), img.size(2))
        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size* self.max_role_count, -1, self.mlp_hidden)

        context_mask = self.encoder.get_adj_matrix_noself_expanded(verb, self.mlp_hidden)
        if self.gpu_mode >= 0:
            context_mask = context_mask.to(torch.device('cuda'))

        logits = self.role_labeller(img, role_q, context_mask)

        role_label_pred = logits.contiguous().view(batch_size, -1, self.vocab_size)
        return role_label_pred

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


def build_baseline0(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)


def build_baseline0_newatt(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)
