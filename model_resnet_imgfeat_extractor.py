import torch
import torch.nn as nn
import torchvision as tv

class resnet_152_features(nn.Module):
    def __init__(self):
        super(resnet_152_features, self).__init__()
        self.resnet = tv.models.resnet152(pretrained=True)

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

        return x

class BaseModel(nn.Module):
    def __init__(self):
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

        self.conv = resnet_152_features()

    def train_preprocess(self):
        return self.train_transform

    def dev_preprocess(self):
        return self.dev_transform

    def forward(self, img):

        img_features = self.conv(img)
        batch_size, n_channel, conv_h, conv_w = img_features.size()
        img = img_features.view(batch_size, n_channel, -1)
        img = img.permute(0, 2, 1)

        return img