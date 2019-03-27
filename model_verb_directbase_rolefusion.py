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
    def __init__(self, num_classes):
        super(vgg16_modified, self).__init__()
        vgg = tv.models.vgg16_bn(pretrained=True)
        self.vgg_features = vgg.features

        '''for param in self.vgg_features.parameters():
            param.require_grad = False'''

        self.out_features = vgg.classifier[6].in_features
        features = list(vgg.classifier.children())[:-1] # Remove last layer
        features.extend([nn.Linear(self.out_features, num_classes)])
        self.vgg_classifier = nn.Sequential(*features) # Replace the model classifier
        #print(self.vgg_classifier)

    def rep_size(self):
        return 1024

    def base_size(self):
        return 512

    def forward_features(self,x):
        y = self.vgg_features(x)
        #print('y size :',  y.size())
        return y

    def forward_classify(self,x):
        y =  self.vgg_classifier(x.view(-1, 512*7*7))
        #print('y size :',  y.size())
        return y

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

        self.conv = vgg16_modified(self.n_verbs)

        #self.convtry = nn.Conv2d(mlp_hidden*2, mlp_hidden, [1, 1], 1, 0, bias=False)

    def train_preprocess(self):
        return self.train_transform

    def dev_preprocess(self, ):
        return self.dev_transform

    def forward_new(self, img, verb, labels):

        verb_pred_prev = self.verb_module(img)

        sorted_idx = torch.sort(verb_pred_prev, 1, True)[1]
        verbs = sorted_idx[:,0]
        _, role_attented_img = self.role_module(img, verbs) #role_attented_img = bx6x49x512

        tot_role_att_img = torch.sum(role_attented_img,1)
        tot_role_att_img = tot_role_att_img.permute(0, 2, 1)

        img_embd = self.conv.forward_features(img)
        batch_size, n_channel, conv_h, conv_w = img_embd.size()
        context_embed = tot_role_att_img.view(batch_size, n_channel, conv_h, conv_w)

        contexted_img = torch.cat([img_embd,context_embed], 1)

        reduced_contexted_img = self.convtry(contexted_img)

        verb_pred = self.conv.forward_classify(reduced_contexted_img)

        return verb_pred

    def forward(self, img, verb, labels):


        img_embd = self.conv.forward_features(img)

        verb_pred = self.conv.forward_classify(img_embd)

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