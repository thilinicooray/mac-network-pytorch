import torch
from torch import nn
import torch.nn.functional as F
import utils
import torchvision as tv
from torch.autograd import Function

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

class RelationNetworks(nn.Module):
    def __init__(
            self,
            encoder,
            gpu_mode,
            conv_hidden=24,
            embed_hidden=300,
            lstm_hidden=300,
            mlp_hidden=512
    ):
        super(RelationNetworks, self).__init__()

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

        self.conv = vgg16_modified()
        self.verb_lookup = nn.Embedding(self.n_verbs, embed_hidden)

        self.verb_transform = nn.Linear(embed_hidden, mlp_hidden*2)

        self.verb = nn.Sequential(
            nn.Linear(mlp_hidden*8, mlp_hidden*2),
            nn.BatchNorm1d(mlp_hidden*2),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Linear(mlp_hidden*2, self.n_verbs)


    def train_preprocess(self):
        return self.train_transform

    def dev_preprocess(self):
        return self.dev_transform

    def forward(self, image, verb_id):

        '''print('testing 123')
        x = torch.tensor([[1, 2, 3],[4,5,6]])
        print('original', x.size())
        x = x.repeat(1,2)
        print('xxxxxx', x, x.view(-1,3), x.size())'''

        conv = self.conv(image)


        #verb pred
        verb_rep = self.verb(conv)
        verb_embedding = self.verb_transform(self.verb_lookup(verb_id))

        return verb_rep, verb_embedding

    def classifier_forward(self, image):

        conv = self.conv(image)
        #verb pred
        verb_rep = self.verb(conv)
        verb_pred = self.classifier(verb_rep)

        return verb_pred

    def classifier_eval(self, image):

        conv = self.conv(image)
        #verb pred
        verb_rep = self.verb(conv)
        verb_pred = self.classifier(verb_rep)

        return verb_pred


    def calculate_loss(self, verb_pred, gt_verbs):

        batch_size = verb_pred.size()[0]
        loss = 0
        #print('eval pred verbs :', pred_verbs)
        for i in range(batch_size):
            verb_loss = utils.cross_entropy_loss(verb_pred[i], gt_verbs[i])
            loss += verb_loss

        final_loss = loss/batch_size
        return final_loss

    def triplet_loss(self, p_img_rep, p_verb_rep, n_img_rep, n_verb_rep, margin=0.5):
        d_p_img = self.pdist(p_img_rep, p_verb_rep)
        d_n_img = self.pdist(p_img_rep, n_verb_rep)

        dist_hinge1 = torch.clamp(margin + d_p_img - d_n_img, min=0.0)

        d_p_label = self.pdist(p_verb_rep, p_img_rep)
        d_n_label = self.pdist(p_verb_rep, n_img_rep)

        dist_hinge2 = torch.clamp(margin + d_p_label - d_n_label, min=0.0)
        loss = torch.mean(dist_hinge1 + dist_hinge2)
        return loss

    def pdist(self, x1, x2, norm=2):
        assert x1.size() == x2.size()
        eps = 1e-4 / x1.size(1)
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, norm).sum(dim=1)
        return torch.pow(out + eps, 1. / norm)




