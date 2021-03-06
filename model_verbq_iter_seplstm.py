import torch
import torch.nn as nn
from attention import Attention, NewAttention, PropAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet
import torch.nn.functional as F
import torchvision as tv
import utils
import numpy as np
import model_roles_recqa_noself_4others
from torch.nn.init import kaiming_uniform_, xavier_uniform_, normal

'''def linear(in_dim, out_dim, bias=True):
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()

    return lin'''

class vgg16_modified(nn.Module):
    def __init__(self):
        super(vgg16_modified, self).__init__()
        vgg = tv.models.vgg16_bn(pretrained=True)
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
                 embed_hidden=768,
                 mlp_hidden=512):
        super(TopDown, self).__init__()

        self.vocab_size = vocab_size

        self.v_att = PropAttention(mlp_hidden, mlp_hidden, mlp_hidden)
        '''self.q_net = FCNet([mlp_hidden, mlp_hidden])
        self.v_net = FCNet([mlp_hidden, mlp_hidden])
        self.classifier = SimpleClassifier(
            mlp_hidden, 2 * mlp_hidden, self.vocab_size, 0.5)'''



    def forward(self, img, q):
        batch_size = img.size(0)
        w_emb = q

        att = self.v_att(img, q)
        v_emb = (att * img)
        v_emb = v_emb.permute(0, 2, 1)
        v_emb = v_emb.contiguous().view(-1, 512*7*7)
        v_emb_with_q = torch.cat([v_emb, q], -1)

        return v_emb_with_q

class BaseModel(nn.Module):
    def __init__(self, encoder,
                 gpu_mode,
                 embed_hidden=300,
                 mlp_hidden = 512
                 ):
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
        self.mlp_hidden = mlp_hidden
        self.verbq_word_count = len(self.encoder.verb_question_words)
        self.n_verbs = self.encoder.get_num_verbs()


        self.conv = vgg16_modified()

        self.verb_vqa = TopDown(self.n_verbs)
        self.verb_q_emb = nn.Embedding(self.verbq_word_count, embed_hidden)

        #xavier_uniform_(self.verb_q_emb.weight)

        self.role_module = model_roles_recqa_noself_4others.BaseModel(self.encoder, self.gpu_mode)

        self.q_emb1 = nn.LSTM(embed_hidden, mlp_hidden,
                              batch_first=True, bidirectional=True)
        self.lstm_proj1 = nn.Linear(mlp_hidden * 2, mlp_hidden)

        self.role_proj = nn.Linear(mlp_hidden, mlp_hidden)
        self.q_emb2 = nn.LSTM(mlp_hidden, mlp_hidden,
                              batch_first=True, bidirectional=True)
        self.lstm_proj2 = nn.Linear(mlp_hidden * 2, mlp_hidden)

        #self.lstm_init(self.lstm_proj1)
        #self.lstm_init(self.lstm_proj2)


        self.last_class = nn.Sequential(
            nn.Linear(mlp_hidden * 7 *7 + mlp_hidden, mlp_hidden*8),
            nn.BatchNorm1d(mlp_hidden*8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden * 8, mlp_hidden*8),
            nn.BatchNorm1d(mlp_hidden*8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.mlp_hidden*8, self.n_verbs)
        )

        self.dropout = nn.Dropout(0.3)

    def lstm_init(self, lstm):
        for name, param in lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                xavier_uniform_(param)


    def init_verbq_embd(self):
        #load word embeddings
        verbq_embd = np.load('imsitu_data/verbq_embd.npy')
        word_emds = torch.from_numpy(verbq_embd)
        pad_data = self.verb_q_emb.weight.data[-1]

        final_embeddings = torch.cat([word_emds,pad_data.unsqueeze(0)],0)
        self.verb_q_emb.weight.data.copy_(final_embeddings)

    def train_preprocess(self):
        return self.train_transform

    def dev_preprocess(self, ):
        return self.dev_transform

    def forward(self, img, verbs=None, labels=None):

        verb_q_idx = self.encoder.get_common_verbq(img.size(0))

        if self.gpu_mode >= 0:
            verb_q_idx = verb_q_idx.to(torch.device('cuda'))

        img_embd = self.conv(img)
        batch_size, n_channel, conv_h, conv_w = img_embd.size()
        img_embd = img_embd.view(batch_size, n_channel, -1)
        img_embd = img_embd.permute(0, 2, 1)

        q_emb = self.verb_q_emb(verb_q_idx)

        self.q_emb1.flatten_parameters()
        lstm_out, (h, _) = self.q_emb1(q_emb)
        q_emb = h.permute(1, 0, 2).contiguous().view(batch_size, -1)
        q_emb = self.lstm_proj1(q_emb)

        verb_pred_rep_prev = self.verb_vqa(img_embd, q_emb)
        verb_pred_prev = self.last_class(verb_pred_rep_prev)

        loss1 = self.calculate_loss(verb_pred_prev, verbs)

        sorted_idx = torch.sort(verb_pred_prev, 1, True)[1]
        verbs = sorted_idx[:,0]
        role_pred, int_rep = self.role_module(img, verbs)

        img_embd = self.conv(img)
        batch_size, n_channel, conv_h, conv_w = img_embd.size()
        img_embd = img_embd.view(batch_size, n_channel, -1)
        img_embd = img_embd.permute(0, 2, 1)

        updated_roleq = torch.cat([self.role_proj(int_rep), q_emb.unsqueeze(1)], 1)
        self.q_emb2.flatten_parameters()
        lstm_out, (h, _) = self.q_emb2(updated_roleq)
        q_emb_up = h.permute(1, 0, 2).contiguous().view(batch_size, -1)
        q_emb_up = self.lstm_proj2(q_emb_up)

        verb_pred_rep = self.verb_vqa(img_embd, q_emb_up)
        combined = verb_pred_rep_prev + self.dropout(verb_pred_rep)
        verb_pred = self.last_class(combined)

        loss2 = self.calculate_loss(verb_pred, verbs)

        sum_losses = loss1 + loss2
        batch_avg_loss = sum_losses / 2
        loss = batch_avg_loss

        return verb_pred, loss

    def forward_eval(self, img, verbs=None, labels=None):
        verb_q_idx = self.encoder.get_common_verbq(img.size(0))

        if self.gpu_mode >= 0:
            verb_q_idx = verb_q_idx.to(torch.device('cuda'))

        img_embd = self.conv(img)
        batch_size, n_channel, conv_h, conv_w = img_embd.size()
        img_embd = img_embd.view(batch_size, n_channel, -1)
        img_embd = img_embd.permute(0, 2, 1)

        q_emb = self.verb_q_emb(verb_q_idx)

        self.q_emb1.flatten_parameters()
        lstm_out, (h, _) = self.q_emb1(q_emb)
        q_emb = h.permute(1, 0, 2).contiguous().view(batch_size, -1)
        q_emb = self.lstm_proj1(q_emb)

        verb_pred_rep_prev = self.verb_vqa(img_embd, q_emb)
        verb_pred_prev = self.last_class(verb_pred_rep_prev)

        sorted_idx = torch.sort(verb_pred_prev, 1, True)[1]
        verbs = sorted_idx[:,0]
        role_pred, int_rep = self.role_module(img, verbs)

        img_embd = self.conv(img)
        batch_size, n_channel, conv_h, conv_w = img_embd.size()
        img_embd = img_embd.view(batch_size, n_channel, -1)
        img_embd = img_embd.permute(0, 2, 1)

        updated_roleq = torch.cat([self.role_proj(int_rep), q_emb.unsqueeze(1)], 1)
        self.q_emb2.flatten_parameters()
        lstm_out, (h, _) = self.q_emb2(updated_roleq)
        q_emb_up = h.permute(1, 0, 2).contiguous().view(batch_size, -1)
        q_emb_up = self.lstm_proj2(q_emb_up)

        verb_pred_rep = self.verb_vqa(img_embd, q_emb_up)
        combined = verb_pred_rep_prev + self.dropout(verb_pred_rep)
        verb_pred = self.last_class(combined)

        verb_pred = verb_pred.contiguous().view(batch_size, -1, self.n_verbs)

        return verb_pred


    def calculate_loss(self, verb_pred, gt_verbs):

        batch_size = verb_pred.size()[0]
        loss = 0
        #print('eval pred verbs :', pred_verbs)
        for i in range(batch_size):
            verb_loss = 0
            verb_loss += utils.cross_entropy_loss(verb_pred[i], gt_verbs[i])
            loss += verb_loss

        final_loss = loss/batch_size
        return final_loss