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
                 vocab_size,
                 embed_hidden=300,
                 mlp_hidden=512):
        super(TopDown, self).__init__()

        self.vocab_size = vocab_size



        self.v_att = Attention(mlp_hidden, mlp_hidden, mlp_hidden)
        self.q_net = FCNet([mlp_hidden, mlp_hidden])
        self.v_net = FCNet([mlp_hidden, mlp_hidden])



    def forward(self, img, q):
        batch_size = img.size(0)
        q_emb = q


        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr

        return joint_repr

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
        self.det_obj_label_count = self.encoder.total_det_objcount

        self.conv = vgg16_modified()
        self.verb_lookup = nn.Embedding(self.n_verbs, embed_hidden)
        self.w_emb = nn.Embedding(self.n_role_q_vocab + 1, embed_hidden, padding_idx=self.n_role_q_vocab)
        self.det_obj_emb = nn.Embedding(self.det_obj_label_count + 1, embed_hidden, padding_idx=self.det_obj_label_count)
        self.q_emb1 = nn.LSTM(embed_hidden, mlp_hidden,
                              batch_first=True, bidirectional=True)
        self.lstm_proj1 = nn.Linear(mlp_hidden * 2, mlp_hidden)
        '''self.q_emb2 = nn.LSTM(mlp_hidden, mlp_hidden,
                              batch_first=True, bidirectional=True)
        self.lstm_proj2 = nn.Linear(mlp_hidden * 2, mlp_hidden)'''
        self.labelrep_proj = nn.Linear(mlp_hidden, embed_hidden)
        self.roles = TopDown(self.vocab_size)
        self.classifier = SimpleClassifier(
            mlp_hidden, 2 * mlp_hidden, self.vocab_size, 0.5)

        self.conv_hidden = self.conv.base_size()
        self.mlp_hidden = mlp_hidden
        self.embed_hidden = embed_hidden
        self.dropout = nn.Dropout(0.3)
        self.num_steps = 3

    def train_preprocess(self):
        return self.train_transform

    def dev_preprocess(self):
        return self.dev_transform

    def forward(self, img, verb):

        img_features = self.conv(img)
        batch_size, n_channel, conv_h, conv_w = img_features.size()
        img = img_features.view(batch_size, n_channel, -1)
        img = img.permute(0, 2, 1)

        img_updated = img

        img_updated = img_updated.expand(self.max_role_count,img_updated.size(0), img_updated.size(1), img_updated.size(2))
        img_updated = img_updated.transpose(0,1)
        img_updated = img_updated.contiguous().view(batch_size* self.max_role_count, -1, self.mlp_hidden)
        verb_embd = self.verb_lookup(verb)
        verb_embed_expand = verb_embd.expand(self.max_role_count, verb_embd.size(0), verb_embd.size(1))
        verb_embed_expand = verb_embed_expand.transpose(0,1)
        verb_embed_expand = verb_embed_expand.contiguous().view(-1, self.embed_hidden)

        role_qs, _ = self.encoder.get_role_questions_batch(verb)
        if self.gpu_mode >= 0:
            role_qs = role_qs.to(torch.device('cuda'))


        role_qs = role_qs.view(batch_size*self.max_role_count, -1)
        embed_qs = torch.cat([ self.w_emb(role_qs),verb_embed_expand.unsqueeze(1)],1)
        self.q_emb1.flatten_parameters()
        lstm_out, (h, _) = self.q_emb1(embed_qs)
        q_emb = h.permute(1, 0, 2).contiguous().view(batch_size*self.max_role_count, -1)
        q_emb = self.lstm_proj1(q_emb)

        rep = self.roles(img_updated, q_emb)

        for i in range(self.num_steps):

            labelrep = rep.contiguous().view(batch_size, -1, self.mlp_hidden)
            labelrep_expand = labelrep.expand(self.max_role_count, labelrep.size(0), labelrep.size(1), labelrep.size(2))
            labelrep_expand = labelrep_expand.transpose(0,1)
            labelrep_expand_new = torch.zeros([batch_size, self.max_role_count, self.max_role_count-1, self.mlp_hidden])
            for i in range(self.max_role_count):
                if i == 0:
                    labelrep_expand_new[:,i] = labelrep_expand[:,i,1:]
                elif i == self.max_role_count -1:
                    labelrep_expand_new[:,i] = labelrep_expand[:,i,:i]
                else:
                    labelrep_expand_new[:,i] = torch.cat([labelrep_expand[:,i,:i], labelrep_expand[:,i,i+1:]], 1)

            if self.gpu_mode >= 0:
                labelrep_expand_new = labelrep_expand_new.to(torch.device('cuda'))

            labelrep_expand = labelrep_expand_new.contiguous().view(-1, self.max_role_count-1, self.mlp_hidden)

            labelrep_expand_proj = self.labelrep_proj(labelrep_expand)

            updated_roleq = torch.cat([labelrep_expand_proj, embed_qs], 1)
            self.q_emb1.flatten_parameters()
            lstm_out, (h, _) = self.q_emb1(updated_roleq)
            q_emb_up = h.permute(1, 0, 2).contiguous().view(batch_size*self.max_role_count, -1)
            q_emb_up = self.lstm_proj1(q_emb_up)

            rep2 = self.roles(img_updated, q_emb_up)

            rep = rep + self.dropout(rep2)
            #rep = self.rep_proj(torch.cat([rep2, rep], -1))
            #rep = rep * rep2

        role_label_pred = self.classifier(rep)
        role_label_pred = role_label_pred.contiguous().view(batch_size, -1, self.vocab_size)


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