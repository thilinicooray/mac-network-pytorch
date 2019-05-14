import torch
import torch.nn as nn
import torchvision as tv
from utils import init_weight

class resnet_modified_large(nn.Module):
    def __init__(self, num_classes):
        super(resnet_modified_large, self).__init__()
        self.resnet = tv.models.resnet152(pretrained=True)
        #probably want linear, relu, dropout
        self.out = nn.Linear(2048, num_classes)
        init_weight(self.out)

    def base_size(self): return 2048
    def rep_size(self): return 1024

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x= self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)

        #print x.size()
        return x

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
        y =  self.vgg_classifier(self.vgg_features(x).view(-1, 512*7*7))
        #print('y size :',  y.size())
        return y

class vgg16_addedlayer(nn.Module):
    def __init__(self, num_classes):
        super(vgg16_addedlayer, self).__init__()
        vgg = tv.models.vgg16(pretrained=True)
        self.vgg_features = vgg.features

        self.conv_exp = nn.Sequential(
            nn.Conv2d(512, 2048, [1, 1], 1, 0, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU()
        )

        self.out_features = vgg.classifier[6].in_features
        features = [nn.Linear(2048*7*7, 4096)]
        features1 = list(vgg.classifier.children())[1:-1] # Remove first and last layer
        features.extend(features1)
        features.extend([nn.Linear(self.out_features, num_classes)])
        self.vgg_classifier = nn.Sequential(*features) # Replace the model classifier
        print(self.vgg_classifier)

    def rep_size(self):
        return 1024

    def base_size(self):
        return 512

    def forward(self,x):
        y =  self.vgg_classifier(self.conv_exp(self.vgg_features(x)).view(-1, 2048*7*7))
        #print('y size :',  y.size())
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
        self.pos_weights = self.encoder.get_obj_weights()

        self.conv = vgg16_addedlayer(self.num_labels)

    def train_preprocess(self):
        return self.train_transform

    def dev_preprocess(self):
        return self.dev_transform

    def forward(self, img):

        pred = self.conv(img)

        return pred

    def calculate_loss(self, pred, gt_labels):

        if self.gpu_mode >= 0:
            self.pos_weights = self.pos_weights.to(torch.device('cuda'))

        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)

        final_loss = criterion(pred, gt_labels)
        return final_loss