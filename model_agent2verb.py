import torch
from torch import nn
import torch.nn.functional as F
import utils
import torchvision as tv

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

class vgg_modified(nn.Module):
    def __init__(self):
        super(vgg_modified,self).__init__()
        self.vgg = tv.models.vgg16(pretrained=True)
        self.vgg_features = self.vgg.features
        #self.classifier = nn.Sequential(
        #nn.Dropout(),
        '''self.lin1 = nn.Linear(512 * 7 * 7, 1024)
        self.relu1 = nn.ReLU(True)
        self.dropout1 = nn.Dropout()
        self.lin2 =  nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU(True)
        self.dropout2 = nn.Dropout()

        utils.initLinear(self.lin1)
        utils.initLinear(self.lin2)'''

    def rep_size(self):
        return 1024

    def base_size(self):
        return 512

    def forward(self,x):
        #return self.dropout2(self.relu2(self.lin2(self.dropout1(self.relu1(self.lin1(self.vgg_features(x).view(-1, 512*7*7)))))))
        return self.vgg_features(x)


class resnet_modified_small(nn.Module):
    def __init__(self):
        super(resnet_modified_small, self).__init__()
        self.resnet = tv.models.resnet34(pretrained=False)
        self.dropout2d = nn.Dropout2d(0.2)
        #probably want linear, relu, dropout
        '''self.linear = nn.Linear(7*7*512, 1024)
        self.dropout2d = nn.Dropout2d(.5)
        self.dropout = nn.Dropout(.5)
        self.relu = nn.LeakyReLU()
        utils.init_weight(self.linear)'''

    def base_size(self):
        return 512

    def rep_size(self):
        return 1024

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.dropout2d(x)

        #return self.dropout(self.relu(self.linear(x.view(-1, 7*7*self.base_size()))))
        return x


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
        #self.vocab_size = self.encoder.get_num_labels()
        self.n_verbs = self.encoder.get_num_verbs()

        self.conv = vgg16_modified()

        self.agent = nn.Sequential(
            nn.Linear(mlp_hidden*8, mlp_hidden*2),
            nn.BatchNorm1d(mlp_hidden*2),
            nn.ReLU()
        )

        self.verb = nn.Sequential(
            nn.Linear(mlp_hidden*2, mlp_hidden*2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden*2, self.n_verbs)
        )
    def train_preprocess(self):
        return self.train_transform

    def dev_preprocess(self):
        return self.dev_transform

    def forward(self, image):

        '''print('testing 123')
        x = torch.tensor([[1, 2, 3],[4,5,6]])
        print('original', x.size())
        x = x.repeat(1,2)
        print('xxxxxx', x, x.view(-1,3), x.size())'''

        conv = self.conv(image)

        #verb pred
        agent_feat = self.agent(conv)

        verb_pred = self.verb(agent_feat)

        return verb_pred


    def calculate_loss(self, verb_pred, gt_verbs):

        batch_size = verb_pred.size()[0]
        loss = 0
        #print('eval pred verbs :', pred_verbs)
        for i in range(batch_size):
            verb_loss = utils.cross_entropy_loss(verb_pred[i], gt_verbs[i])
            loss += verb_loss
            '''for index in range(gt_labels.size()[1]):
                frame_loss = 0
                verb_loss = utils.cross_entropy_loss(verb_pred[i], gt_verbs[i])


                #frame_loss += verb_loss
                #print('frame loss', frame_loss)
                loss += verb_loss'''


        final_loss = loss/batch_size
        #print('loss :', final_loss)
        return final_loss



