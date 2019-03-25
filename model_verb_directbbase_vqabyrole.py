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
import model_roles_verbcatrole2img

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

        self.v_att = Attention(mlp_hidden, mlp_hidden, mlp_hidden)
        self.q_net = FCNet([mlp_hidden, mlp_hidden])
        self.v_net = FCNet([mlp_hidden, mlp_hidden])
        self.classifier = SimpleClassifier(
            mlp_hidden, 2 * mlp_hidden, self.vocab_size, 0.5)
        self.mlp_hidden = mlp_hidden



    def forward(self, img, q):
        batch_size = img.size(0)
        q_emb = q
        print('sizes :', img.size(), q_emb.size())
        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr_new = q_repr * v_repr

        joint_rep = joint_repr_new.view(-1, 6, self.mlp_hidden)
        v_emb_with_q = torch.sum(joint_rep, 1)

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
        self.role_module = model_roles_verbcatrole2img.BaseModel(self.encoder, self.gpu_mode)
        self.verb_module.eval()
        self.role_module.eval()

        self.conv = vgg16_modified()
        self.verbqa = TopDown(self.n_verbs)

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


            img_embd = img_embd.expand(self.role_module.max_role_count,img_embd.size(0), img_embd.size(1), img_embd.size(2))
            img_embd = img_embd.transpose(0,1)
            img_embd = img_embd.contiguous().view(batch_size* self.role_module.max_role_count, -1, self.mlp_hidden)

            _, pred_rep = self.role_module(img, verb)

            pred_rep_exp = pred_rep.contiguous().view(batch_size* self.role_module.max_role_count, -1)

            verb_pred = self.verbqa(img_embd, pred_rep_exp)

        else:
            verb_pred_prev = self.verb_module(img)

            sorted_idx = torch.sort(verb_pred_prev, 1, True)[1]
            verbs = sorted_idx[:,0]
            _, pred_rep = self.role_module(img, verbs)

            img_embd = self.conv(img)
            batch_size, n_channel, conv_h, conv_w = img_embd.size()
            img_embd = img_embd.view(batch_size, n_channel, -1)
            img_embd = img_embd.permute(0, 2, 1)
            img_embd = img_embd.expand(self.role_module.max_role_count,img_embd.size(0), img_embd.size(1), img_embd.size(2))
            img_embd = img_embd.transpose(0,1)
            img_embd = img_embd.contiguous().view(batch_size* self.role_module.max_role_count, -1, self.mlp_hidden)

            pred_rep_exp = pred_rep.contiguous().view(batch_size* self.role_module.max_role_count, -1)

            verb_pred = self.verbqa(img_embd, pred_rep_exp)

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