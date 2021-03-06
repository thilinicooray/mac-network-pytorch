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
import model_roles_independent

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
        self.q_prep = FCNet([mlp_hidden, mlp_hidden])
        self.lstm_proj = nn.Linear(mlp_hidden * 2, mlp_hidden)
        self.verb_transform = nn.Linear(embed_hidden, mlp_hidden)
        self.v_att = Attention(mlp_hidden, mlp_hidden, mlp_hidden)
        self.q_net = FCNet([mlp_hidden, mlp_hidden])
        self.v_net = FCNet([mlp_hidden, mlp_hidden])
        self.classifier = SimpleClassifier(
            mlp_hidden, 2 * mlp_hidden, self.vocab_size, 0.5)


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
        logits = self.classifier(joint_repr)

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
        self.role_module = model_roles_independent.BaseModel(self.encoder, self.gpu_mode)
        self.conv = vgg16_modified()

        self.verb_module.eval()
        self.role_module.eval()

        '''for param in self.verb_module.parameters():
            param.require_grad = False

        for param in self.role_module.parameters():
            param.require_grad = False
        
        for param in self.conv.parameters():
            param.require_grad = False'''
        self.verb_vqa = TopDown(self.n_verbs)
        self.verb_q_emb = nn.Embedding(self.verbq_word_count + 1, embed_hidden, padding_idx=self.verbq_word_count)


    def train_preprocess(self):
        return self.train_transform

    def dev_preprocess(self, ):
        return self.dev_transform

    def forward(self, img, verb, labels):

        if self.training:
            verb_q_idx_1 = self.encoder.get_verbq_idx(verb, labels[:,0,:]).unsqueeze(1)
            verb_q_idx_2 = self.encoder.get_verbq_idx(verb, labels[:,1,:]).unsqueeze(1)
            verb_q_idx_3 = self.encoder.get_verbq_idx(verb, labels[:,2,:]).unsqueeze(1)
            verb_q_idx = torch.cat([verb_q_idx_1, verb_q_idx_2, verb_q_idx_3], 1)

            if self.gpu_mode >= 0:
                verb_q_idx = verb_q_idx.to(torch.device('cuda'))

            img_embd = self.conv(img)
            batch_size, n_channel, conv_h, conv_w = img_embd.size()
            img_embd = img_embd.view(batch_size, n_channel, -1)
            img_embd = img_embd.permute(0, 2, 1)
            img_embd = img_embd.expand(3,img_embd.size(0), img_embd.size(1), img_embd.size(2))
            img_embd = img_embd.transpose(0,1)
            img_embd = img_embd.contiguous().view(batch_size* 3, -1, self.mlp_hidden)

            verb_q_idx = verb_q_idx.view(batch_size*3, -1)
            q_emb = self.verb_q_emb(verb_q_idx)

            verb_pred = self.verb_vqa(img_embd, q_emb)

            verb_pred = verb_pred.contiguous().view(batch_size, -1, self.n_verbs)


        else:
            verb_pred_prev = self.verb_module(img)

            sorted_idx = torch.sort(verb_pred_prev, 1, True)[1]
            verbs = sorted_idx[:,0]
            role_pred = self.role_module(img, verbs)
            label_idx = torch.max(role_pred,-1)[1]

            verb_q_idx = self.encoder.get_verbq_idx(verbs, label_idx)

            if self.gpu_mode >= 0:
                verb_q_idx = verb_q_idx.to(torch.device('cuda'))

            img_embd = self.conv(img)
            batch_size, n_channel, conv_h, conv_w = img_embd.size()
            img_embd = img_embd.view(batch_size, n_channel, -1)
            img_embd = img_embd.permute(0, 2, 1)

            q_emb = self.verb_q_emb(verb_q_idx)

            verb_pred = self.verb_vqa(img_embd, q_emb) + verb_pred_prev

        return verb_pred

    def calculate_loss(self, verb_pred, gt_verbs):

        batch_size = verb_pred.size()[0]
        verb_ref = verb_pred.size(1)
        loss = 0
        #print('eval pred verbs :', pred_verbs)
        for i in range(batch_size):
            verb_loss = 0
            for r in range(verb_ref):
                verb_loss += utils.cross_entropy_loss(verb_pred[i][r], gt_verbs[i])
            loss += verb_loss


        final_loss = loss/batch_size
        #print('loss :', final_loss)
        return final_loss