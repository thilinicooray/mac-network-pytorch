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


class BaseModel(nn.Module):
    def __init__(self, encoder):
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
        self.num_labels = self.encoder.get_num_labels()
        self.pos_weights = self.encoder.get_obj_weights()

        self.conv = resnet_modified_large(self.num_labels)

    def train_preprocess(self):
        return self.train_transform

    def dev_preprocess(self):
        return self.dev_transform

    def forward(self, img):

        pred = self.conv(img)

        return pred

    def calculate_loss(self, pred, gt_labels):

        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)

        final_loss = criterion(pred, gt_labels)
        return final_loss