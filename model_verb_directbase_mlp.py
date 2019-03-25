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
import model_verb_directcnn
import model_roles_verbrole_only

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
                 embed_hidden=300,
                 mlp_hidden=512):
        super(TopDown, self).__init__()

        self.vocab_size = vocab_size

        self.q_emb = nn.LSTM(embed_hidden, mlp_hidden,
                             batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(mlp_hidden * 2, mlp_hidden)
        self.verb_transform = nn.Linear(embed_hidden, mlp_hidden)
        self.v_att = NewAttention(mlp_hidden, mlp_hidden, mlp_hidden)
        '''self.q_net = FCNet([mlp_hidden, mlp_hidden])
        self.v_net = FCNet([mlp_hidden, mlp_hidden])
        self.classifier = SimpleClassifier(
            mlp_hidden, 2 * mlp_hidden, self.vocab_size, 0.5)'''



    def forward(self, img, q):
        batch_size = img.size(0)
        w_emb = q
        self.q_emb.flatten_parameters()
        lstm_out, (h, _) = self.q_emb(w_emb)
        q_emb = h.permute(1, 0, 2).contiguous().view(batch_size, -1)
        q_emb = self.lstm_proj(q_emb)

        att = self.v_att(img, q_emb)
        v_emb = (att * img)
        v_emb = v_emb.permute(0, 2, 1)
        v_emb = v_emb.contiguous().view(-1, 512*7*7)
        v_emb_with_q = torch.cat([v_emb, q_emb], -1)
        logits = self.classifier(v_emb_with_q)

        return logits

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
        self.verbq_word_count = len(self.encoder.verb_q_words)
        self.n_verbs = self.encoder.get_num_verbs()

        self.verb_module = model_verb_directcnn.BaseModel(self.encoder, self.gpu_mode)
        self.role_module = model_roles_verbrole_only.BaseModel(self.encoder, self.gpu_mode)

        self.conv = vgg16_modified()

        self.classifier = nn.Sequential(
            nn.Linear(mlp_hidden * 49, mlp_hidden*8),
            nn.BatchNorm1d(mlp_hidden*8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden * 8, mlp_hidden*8),
            nn.BatchNorm1d(mlp_hidden*8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden * 8, self.n_verbs)
        )



    def train_preprocess(self):
        return self.train_transform

    def dev_preprocess(self, ):
        return self.dev_transform

    def forward(self, img, verb, labels):

        if self.training:

            img_embd = self.conv(img)
            batch_size, n_channel, conv_h, conv_w = img_embd.size()
            img_embd = img_embd.view(batch_size, n_channel, -1)
            img_embd = img_embd.permute(0, 2, 1)

            #_, pred_rep = self.role_module(img, verb)

            #contexted_img = torch.cat([img_embd,pred_rep], 1)

            verb_pred = self.classifier(img_embd.contiguous().view(-1, 512*49))

        else:
            '''verb_pred_prev = self.verb_module(img)

            sorted_idx = torch.sort(verb_pred_prev, 1, True)[1]
            verbs = sorted_idx[:,0]
            _, pred_rep = self.role_module(img, verbs)'''

            img_embd = self.conv(img)
            batch_size, n_channel, conv_h, conv_w = img_embd.size()
            img_embd = img_embd.view(batch_size, n_channel, -1)
            img_embd = img_embd.permute(0, 2, 1)

            #_, pred_rep = self.role_module(img, verb)

            #contexted_img = torch.cat([img_embd,pred_rep], 1)

            verb_pred = self.classifier(img_embd.contiguous().view(-1, 512*49))

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
        #print('loss :', final_loss)
        return final_loss