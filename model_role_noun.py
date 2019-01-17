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


        self.v_att = Attention(mlp_hidden, mlp_hidden, mlp_hidden)


    def forward(self, img, q_emb):


        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1) # [batch, v_dim]

        return v_emb

class RoleQHandler(nn.Module):
    def __init__(self,
                 embed_hidden=300,
                 mlp_hidden=512):
        super(RoleQHandler, self).__init__()

        self.q_emb = nn.LSTM(embed_hidden, mlp_hidden,
                             batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(mlp_hidden * 2, mlp_hidden)

        self.vqa_model = TopDown()



    def forward(self, img, q):
        batch_size = q.size(0)
        w_emb = q
        lstm_out, (h, _) = self.q_emb(w_emb)
        q_emb = h.permute(1, 0, 2).contiguous().view(batch_size, -1)
        q_emb = self.lstm_proj(q_emb)
        v_emb = self.vqa_model(img, q_emb)
        return q_emb, v_emb

class RolePredictor(nn.Module):
    def __init__(self,
                 embed_hidden=300,
                 mlp_hidden=512):
        super(RolePredictor, self).__init__()

        self.inv_roleq = nn.Linear(mlp_hidden + embed_hidden, mlp_hidden)

        self.vqa_model = TopDown()



    def forward(self, img, q):

        q_emb = self.inv_roleq(q)
        v_emb = self.vqa_model(img, q_emb)
        return q_emb, v_emb


class ImSituationHandler(nn.Module):
    def __init__(self,
                 encoder,
                 qword_embeddings,
                 verb_lookup,
                 n_roles,
                 vocab_size,
                 gpu_mode,
                 mlp_hidden=512,
                 embd_hidden=300):
        super(ImSituationHandler, self).__init__()


        self.encoder = encoder
        self.qword_embeddings = qword_embeddings
        self.verb_lookup = verb_lookup
        self.role_size = n_roles
        self.vocab_size = vocab_size
        self.gpu_mode = gpu_mode
        self.mlp_hidden = mlp_hidden
        self.embd_hidden = embd_hidden
        self.role_handler = RoleQHandler()
        self.role_predictor = RolePredictor()
        self.noun_q_net = FCNet([mlp_hidden, mlp_hidden])
        self.noun_v_net = FCNet([mlp_hidden, mlp_hidden])
        self.noun_classifier = SimpleClassifier(
            mlp_hidden, 2 * mlp_hidden, self.vocab_size, 0.5)

        self.role_q_net = FCNet([mlp_hidden, mlp_hidden])
        self.role_v_net = FCNet([mlp_hidden, mlp_hidden])
        self.role_classifier = SimpleClassifier(
            mlp_hidden, 2 * mlp_hidden, self.role_size, 0.5)


    def forward(self, img, verb):

        role_qs, _ = self.encoder.get_role_questions_batch(verb)
        #roles = self.encoder.get_role_ids_batch(verb)

        if self.gpu_mode >= 0:
            role_qs = role_qs.to(torch.device('cuda'))
            #roles = roles.to(torch.device('cuda'))
        role_qs = role_qs.view(-1, role_qs.size(-1))

        noun_q_emb, noun_v_emb = self.role_handler(img, self.qword_embeddings(role_qs))

        noun_q_repr = self.noun_q_net(noun_q_emb)
        noun_v_repr = self.noun_v_net(noun_v_emb)
        joint_noun_repr = noun_q_repr * noun_v_repr
        noun_logits = self.noun_classifier(joint_noun_repr)

        #role_pred
        verb_embd = self.verb_lookup(verb)
        verb_embed_expand = verb_embd.expand(6, verb_embd.size(0), verb_embd.size(1))
        verb_embed_expand = verb_embed_expand.transpose(0,1)
        verb_embed_expand = verb_embed_expand.contiguous().view(-1, self.embd_hidden)
        role_search_q = torch.cat([verb_embed_expand,joint_noun_repr],1)

        role_q_emb, role_v_emb = self.role_predictor(img, role_search_q)

        role_q_repr = self.role_q_net(role_q_emb)
        role_v_repr = self.role_v_net(role_v_emb)
        joint_role_repr = role_q_repr * role_v_repr
        role_logits = self.role_classifier(joint_role_repr)


        return noun_logits, role_logits


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
        self.verb_lookup = nn.Embedding(self.n_verbs, embed_hidden)
        self.ans_lookup = nn.Embedding(self.vocab_size + 1, embed_hidden, padding_idx=self.vocab_size)
        self.w_emb = nn.Embedding(self.n_role_q_vocab + 1, embed_hidden, padding_idx=self.n_role_q_vocab)

        self.vsrl_model = ImSituationHandler(self.encoder, self.w_emb, self.verb_lookup, self.n_roles, self.vocab_size,
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

        noun_pred, role_pred = self.vsrl_model(img, verb)
        noun_pred = noun_pred.contiguous().view(batch_size, -1, self.vocab_size)
        role_pred = role_pred.contiguous().view(batch_size, -1, self.n_roles)
        #print('ans sizes :', verb_pred.size(), role_pred.size())

        return noun_pred, role_pred

    def calculate_noun_loss(self, gt_verbs, role_label_pred, gt_labels,args):

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

    def calculate_role_loss(self, gt_verbs, role_pred, gt_role,args):

        batch_size = role_pred.size()[0]
        if args.train_all:
            loss = 0
            for i in range(batch_size):
                frame_loss = 0
                for j in range(0, self.max_role_count):
                    frame_loss += utils.cross_entropy_loss(role_pred[i][j], gt_role[i,j] ,self.n_roles)
                frame_loss = frame_loss/len(self.encoder.verb2_role_dict[self.encoder.verb_list[gt_verbs[i]]])
                loss += frame_loss


        final_loss = loss/batch_size
        #print('loss :', final_loss)
        return final_loss
