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
import model_verbq_final_addpreveval
import model_roles_independent

class BaseModel(nn.Module):
    def __init__(self, encoder,
                 gpu_mode,
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

        self.verb_module = model_verbq_final_addpreveval.BaseModel(self.encoder, self.gpu_mode)
        self.role_module = model_roles_independent.BaseModel(self.encoder, self.gpu_mode)


    def train_preprocess(self):
        return self.train_transform

    def dev_preprocess(self, ):
        return self.dev_transform

    def forward(self, img, verb, labels, beam=10, topk=5):

        role_pred_topk = None

        verb_pred = self.verb_module(img, verb, labels)

        verb_pred = F.log_softmax(verb_pred, dim=-1)

        sorted_val, sorted_idx = torch.sort(verb_pred, 1, True)
        verbs = sorted_idx[:,:beam]
        selected_verb_val = sorted_val[:, : beam]

        for k in range(0,beam):
            role_pred = self.role_module(img, verbs[:,k])
            role_pred = F.log_softmax(role_pred, dim=-1)

            if k == 0:
                val, idx = torch.max(role_pred,-1)
                role_pred_topk = idx.unsqueeze(1)
            else:
                val,idx = torch.max(role_pred,-1)
                role_pred_topk = torch.cat((role_pred_topk.clone(), idx.unsqueeze(1)), 1)
            selected_verb_val[:, k] += torch.sum(val,1)

            if self.gpu_mode >= 0:
                torch.cuda.empty_cache()

        b_sorted_val, beam_sorted_idx = torch.sort(selected_verb_val, 1, True)


        verb_beam = verbs[0][beam_sorted_idx[0]].unsqueeze(0)

        roles_beam = role_pred_topk[0][beam_sorted_idx[0]].unsqueeze(0)

        for i in range(1,img.size(0)):
            verb_beam = torch.cat((verb_beam.clone(), verbs[i][beam_sorted_idx[i]].unsqueeze(0)), 0)
            roles_beam = torch.cat((roles_beam.clone(), role_pred_topk[i][beam_sorted_idx[i]].unsqueeze(0)), 0)


        return verb_beam[:, :topk], roles_beam[:,:topk]