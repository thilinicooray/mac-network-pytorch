import torch
import torch.nn as nn
import torchvision as tv
from utils import init_weight

class vgg16_modified(nn.Module):
    def __init__(self, num_classes):
        super(vgg16_modified, self).__init__()
        vgg = tv.models.vgg16(pretrained=True)
        self.vgg_features = vgg.features

        self.out_features = vgg.classifier[6].in_features
        features = list(vgg.classifier.children())[:-1] # Remove last layer
        features.extend([nn.Linear(self.out_features, num_classes)])
        self.vgg_classifier = nn.Sequential(*features) # Replace the model classifier
        #print(self.vgg_classifier)

    def rep_size(self):
        return 1024

    def base_size(self):
        return 512

    def forward(self,x):
        feats = self.vgg_features(x).view(-1, 512*7*7)
        y =  self.vgg_classifier[:-1](feats)
        print('y size :',  y.size())
        return y



class BaseModel(nn.Module):
    def __init__(self, encoder, gpu_mode):
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
        self.num_labels = self.encoder.get_num_labels()
        self.num_verbs = self.encoder.get_num_verbs()
        self.pos_weights = self.encoder.get_obj_weights()

        self.conv_nouns = vgg16_modified(self.num_labels)
        self.conv_verbs = vgg16_modified(self.num_verbs)

    def train_preprocess(self):
        return self.train_transform

    def dev_preprocess(self):
        return self.dev_transform

    def forward(self, img):

        pred_nouns_feat = self.conv_nouns(img)
        pred_verb_feat = self.conv_verbs(img)

        return pred_verb_feat, pred_nouns_feat

    def calculate_loss(self, pred, gt_labels):

        if self.gpu_mode >= 0:
            self.pos_weights = self.pos_weights.to(torch.device('cuda'))

        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)

        final_loss = criterion(pred, gt_labels)
        return final_loss