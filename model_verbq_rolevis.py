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
import model_verbq_0
import model_roles_recqa_noself

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
                 vocab_size,
                 embed_hidden=300,
                 mlp_hidden=512):
        super(TopDown, self).__init__()

        self.vocab_size = vocab_size

        self.q_emb = nn.LSTM(embed_hidden, mlp_hidden,
                             batch_first=True, bidirectional=True)
        self.q_prep = FCNet([mlp_hidden, mlp_hidden])
        self.lstm_proj = nn.Linear(mlp_hidden * 2, mlp_hidden)
        self.verb_transform = nn.Linear(embed_hidden, mlp_hidden)
        self.v_att = Attention(mlp_hidden, mlp_hidden, mlp_hidden)
        self.q_net = FCNet([mlp_hidden, mlp_hidden])
        self.v_net = FCNet([mlp_hidden, mlp_hidden])



    def forward(self, img, q):
        batch_size = img.size(0)
        w_emb = q
        self.q_emb.flatten_parameters()
        lstm_out, (h, _) = self.q_emb(w_emb)
        q_emb = h.permute(1, 0, 2).contiguous().view(batch_size, -1)
        q_emb = self.lstm_proj(q_emb)

        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr

        return joint_repr, v_repr

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
        self.verbq_word_count = len(self.encoder.verb_question_words)

        self.role_module = model_roles_recqa_noself.BaseModel(self.encoder, self.gpu_mode)
        self.role_module.eval()

        self.verb_vqa = model_verbq_0.TopDown(self.n_verbs)
        self.verb_q_emb = nn.Embedding(self.verbq_word_count + 1, embed_hidden, padding_idx=self.verbq_word_count)
        self.verb_last_class = nn.Linear(mlp_hidden*8, self.n_verbs)

        self.verb_role_maker = nn.Linear(mlp_hidden, mlp_hidden)
        self.verb_real_comb_concat = nn.Linear(mlp_hidden * 2, mlp_hidden)

        #self.conv_hidden = self.conv.base_size()
        self.mlp_hidden = mlp_hidden
        self.embed_hidden = embed_hidden

    def train_preprocess(self):
        return self.train_transform

    def dev_preprocess(self):
        return self.dev_transform

    def forward(self, img, verb, labels):


        _, vis_rep = self.role_module(img, verb)
        vis_rep = vis_rep.contiguous().view(-1, self.mlp_hidden)

        img_features = self.role_module.conv(img)
        batch_size, n_channel, conv_h, conv_w = img_features.size()
        img_ft = img_features.view(batch_size, n_channel, -1)
        img_ft = img_ft.permute(0, 2, 1)

        img_exp = img_ft.expand(self.max_role_count,img_ft.size(0), img_ft.size(1), img_ft.size(2))
        img_exp = img_exp.transpose(0,1)
        img_exp = img_exp.contiguous().view(batch_size* self.max_role_count, -1, self.mlp_hidden)

        verb_q_idx = self.encoder.get_common_verbq(img.size(0))

        if self.gpu_mode >= 0:
            verb_q_idx = verb_q_idx.to(torch.device('cuda'))

        role_values = self.verb_role_maker(vis_rep).unsqueeze(1)

        #print(role_values[0])
        rolewise = role_values * img_exp
        added_all = torch.sum(rolewise.view(-1,self.max_role_count, rolewise.size(1), rolewise.size(2) ), 1)
        joined = torch.cat([added_all, img_ft], 2)
        combo = self.verb_real_comb_concat(joined)

        qw_emb_i1 = self.verb_q_emb(verb_q_idx)

        verb_pred_logit_i1 = self.verb_vqa(combo, qw_emb_i1)
        verb_pred_i1 = self.verb_last_class(verb_pred_logit_i1)

        #role_label_pred = self.role_classifier(rep).contiguous().view(batch_size, -1, self.vocab_size)
        return verb_pred_i1


    def calculate_loss(self, verb_pred, gt_verbs):

        batch_size = verb_pred.size()[0]
        loss = 0
        #print('eval pred verbs :', pred_verbs)
        for i in range(batch_size):
            verb_loss = 0
            verb_loss += utils.cross_entropy_loss(verb_pred[i], gt_verbs[i])
            loss += verb_loss


        final_loss = loss/batch_size
        #print('loss :', final_loss)
        return final_loss