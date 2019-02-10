import torch
from torch import nn
import torch.nn.functional as F
import utils
import torchvision as tv
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
        y =  self.vgg_classifier(self.vgg_features(x).view(-1, 512*7*7))
        #print('y size :',  y.size())
        return y

class BaseModel(nn.Module):
    def __init__(
            self,
            encoder,
            gpu_mode,
            conv_hidden=24,
            embed_hidden=300,
            lstm_hidden=300,
            mlp_hidden=512
    ):
        super(BaseModel, self).__init__()

        self.normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.train_transform = tv.transforms.Compose([
            tv.transforms.RandomResizedCrop(224),
            tv.transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
        ])

        self.dev_transform = tv.transforms.Compose([
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(224),
            lambda x: np.asarray(x),
        ])

        self.encoder = encoder
        self.gpu_mode = gpu_mode
        self.n_verbs = self.encoder.get_num_verbs()

        self.conv = vgg16_modified()

        self.rotation = nn.Sequential(
            nn.Linear(mlp_hidden*8, mlp_hidden*2),
            nn.BatchNorm1d(mlp_hidden*2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden*2, self.n_verbs),
        )


    def train_preprocess(self):
        return self.train_transform

    def dev_preprocess(self):
        return self.dev_transform

    def forward(self, image):

        conv = self.conv(image)

        #verb pred
        rot_pred = self.rotation(conv)

        return rot_pred


    def calculate_loss(self, rot_pred, gt_labels):

        batch_size = rot_pred.size()[0]
        loss = 0
        for i in range(batch_size):
            verb_loss = utils.cross_entropy_loss(rot_pred[i], gt_labels[i])
            loss += verb_loss

        final_loss = loss/batch_size
        return final_loss
