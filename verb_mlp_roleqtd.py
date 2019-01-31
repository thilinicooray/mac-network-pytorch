import torch
import torch.nn as nn
from attention import Attention, BigAttention
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
        self.q_net = FCNet([mlp_hidden, mlp_hidden])
        self.v_net = FCNet([mlp_hidden, mlp_hidden])


    def forward(self, img, q):
        batch_size = img.size(0)
        w_emb = q
        lstm_out, (h, _) = self.q_emb(w_emb)
        q_emb = h.permute(1, 0, 2).contiguous().view(batch_size, -1)
        q_emb = self.lstm_proj(q_emb)

        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr

        return joint_repr

class RoleNode(nn.Module):
    def __init__(self,
                 encoder,
                 gpu_mode,
                 q_word_embeddings,
                 num_roles,
                 num_labels, mlp_hidden,
                 embd_hidden, vqa_model):
        super(RoleNode, self).__init__()

        self.encoder = encoder
        self.gpu_mode = gpu_mode

        self.q_word_embeddings = q_word_embeddings
        self.num_roles = num_roles
        self.num_labels = num_labels
        self.mlp_hidden = mlp_hidden
        self.embd_hidden = embd_hidden
        self.vqa_model = vqa_model

        self.label_classifier = SimpleClassifier(
            mlp_hidden, 2 * mlp_hidden, self.num_labels + 1, 0.5)

    def forward(self, img, roleq):


        roleq = self.q_word_embeddings(roleq)
        batch_size = img.size(0)
        max_role_count = self.encoder.get_max_role_count()

        roleq = roleq.view(-1, roleq.size(-2), roleq.size(-1))

        img = img.expand(max_role_count,img.size(0), img.size(1), img.size(2))
        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size* max_role_count, -1, self.mlp_hidden)

        '''verb_expanded = self.verb_project(verb)
        word_count = roleq.size(-2)
        verb_set = verb_expanded.expand(word_count, verb_expanded.size(0), verb_expanded.size(1))
        verb_set = verb_set.transpose(0,1)

        roleq = torch.cat([roleq, verb_set],-1)'''

        ans_rep = self.vqa_model(img, roleq)
        label_logits = self.label_classifier(ans_rep)

        label_logits = label_logits.view(batch_size, max_role_count, self.num_labels + 1)

        return ans_rep, label_logits


class RecursiveGraph(nn.Module):
    def __init__(self,
                 encoder,
                 q_word_embeddings,
                 num_verbs, num_roles, num_labels,
                 embd_hidden,
                 mlp_hidden,
                 max_roles,
                 gpu_mode):
        super(RecursiveGraph, self).__init__()

        self.encoder = encoder
        self.q_word_embeddings = q_word_embeddings
        self.num_verbs = num_verbs
        self.num_roles = num_roles
        self.num_labels = num_labels
        self.emd_hidden = embd_hidden
        self.mlp_hidden = mlp_hidden
        self.max_roles = max_roles
        self.gpu_mode = gpu_mode
        self.vqa_model = TopDown()
        '''self.verb_node = VerbNode(self.encoder, self.gpu_mode, self.verb_lookup, self.q_word_embeddings,
                                  self.num_verbs, self.vqa_model, self.mlp_hidden, self.emd_hidden)'''
        self.verb_mlp = nn.Sequential(
            nn.Linear(mlp_hidden*8, mlp_hidden*2),
            nn.BatchNorm1d(mlp_hidden*2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden*2, self.num_verbs),
        )
        self.role_node = RoleNode(self.encoder, self.gpu_mode, self.q_word_embeddings,
                                  self.num_roles, self.num_labels, self.mlp_hidden,
                                  self.emd_hidden, self.vqa_model)

    def forward(self, img, img_cls, verbq, gt_verb):
        verb_pred = self.verb_mlp(img_cls)

        role_qs = self.encoder.get_role_questions_batch(gt_verb)
        if self.gpu_mode >= 0:
            role_qs = role_qs.to(torch.device('cuda'))

        label_soft_ans, label_logits = self.role_node(img, role_qs)

        return verb_pred, label_logits


    def forward_eval(self, img, img_cls, verbq):

        verb_pred = self.verb_mlp(img_cls)
        sorted_idx = torch.sort(verb_pred, 1, True)[1]
        verb = sorted_idx[:,0]
        role_qs = self.encoder.get_role_questions_batch(verb)
        if self.gpu_mode >= 0:
            role_qs = role_qs.to(torch.device('cuda'))

        label_soft_ans, label_logits  = self.role_node(img, role_qs)

        return verb_pred, label_logits



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
        self.w_emb = nn.Embedding(self.n_role_q_vocab + 2, embed_hidden, padding_idx=self.n_role_q_vocab)

        self.vsrl_model = RecursiveGraph(self.encoder,
                                         self.w_emb,
                                         self.n_verbs, self.n_roles, self.vocab_size,
                                         embed_hidden,
                                         mlp_hidden,
                                         self.max_role_count,
                                         self.gpu_mode
                                         )


        self.conv_hidden = self.conv.base_size()
        self.mlp_hidden = mlp_hidden
        self.embed_hidden = embed_hidden

    def train_preprocess(self):
        return self.train_transform

    def dev_preprocess(self):
        return self.dev_transform

    def forward(self, img, verbq, verb):

        img_features, img_cls = self.conv(img)
        batch_size, n_channel, conv_h, conv_w = img_features.size()
        img = img_features.view(batch_size, n_channel, -1)
        img = img.permute(0, 2, 1)

        if self.training:
            verb_pred, label_pred = self.vsrl_model(img, img_cls, verbq, verb)
        else:
            verb_pred, label_pred = self.vsrl_model.forward_eval(img, img_cls, verbq)


        return verb_pred, label_pred

    def calculate_loss(self, verb_pred, gt_verbs, role_label_pred, gt_labels,args):

        batch_size = verb_pred.size()[0]
        if args.train_all:
            loss = 0
            for i in range(batch_size):
                for index in range(gt_labels.size()[1]):
                    frame_loss = 0
                    verb_loss = utils.cross_entropy_loss(verb_pred[i], gt_verbs[i])
                    #frame_loss = criterion(role_label_pred[i], gt_labels[i,index])
                    for j in range(0, self.max_role_count):
                        frame_loss += utils.cross_entropy_loss(role_label_pred[i][j], gt_labels[i,index,j] ,self.vocab_size)
                    frame_loss = (verb_loss) + frame_loss/len(self.encoder.verb2_role_dict[self.encoder.verb_list[gt_verbs[i]]])
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


