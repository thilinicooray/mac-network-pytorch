import torch
from torch import nn
import torch.nn.functional as F
import utils
import torchvision as tv

from attention import Attention
from classifier import SimpleClassifier
from fc import FCNet

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
        feat =  self.vgg_features(x)
        y =  self.vgg_classifier(feat.view(-1, 512*7*7))
        #print('y size :',  y.size())
        return feat, y

class verb_module(nn.Module):
    def __init__(
            self,
            n_verbs,
            mlp_hidden=512
    ):
        super(verb_module, self).__init__()
        self.conv = vgg16_modified()

        self.verb = nn.Sequential(
            nn.Linear(mlp_hidden*8, mlp_hidden*2),
            nn.BatchNorm1d(mlp_hidden*2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden*2, n_verbs),
        )

    def forward(self, image):

        _, conv = self.conv(image)

        #verb pred
        verb_pred = self.verb(conv)

        return verb_pred

class role_module(nn.Module):
    def __init__(self, encoder,
                 gpu_mode,
                 embed_hidden=300,
                 mlp_hidden=512):
        super(role_module, self).__init__()

        self.encoder = encoder
        self.gpu_mode = gpu_mode
        self.n_roles = self.encoder.get_num_roles()
        self.n_verbs = self.encoder.get_num_verbs()
        self.vocab_size = self.encoder.get_num_labels()
        self.max_role_count = self.encoder.get_max_role_count()
        self.n_role_q_vocab = len(self.encoder.question_words)

        self.conv = vgg16_modified()
        self.verb_lookup = nn.Embedding(self.n_verbs, embed_hidden)
        self.w_emb = nn.Embedding(self.n_role_q_vocab + 1, embed_hidden, padding_idx=self.n_role_q_vocab)
        self.q_emb = nn.LSTM(embed_hidden, mlp_hidden,
                             batch_first=True, bidirectional=True)
        self.q_prep = FCNet([mlp_hidden, mlp_hidden])
        self.lstm_proj = nn.Linear(mlp_hidden * 2, mlp_hidden)
        self.verb_transform = nn.Linear(embed_hidden, mlp_hidden)
        self.v_att = Attention(mlp_hidden, mlp_hidden, mlp_hidden)
        self.q_net = FCNet([mlp_hidden, mlp_hidden])
        self.v_net = FCNet([mlp_hidden, mlp_hidden])
        self.classifier = SimpleClassifier(
            mlp_hidden, 2 * mlp_hidden, self.vocab_size, 0.5)

        self.conv_hidden = self.conv.base_size()
        self.mlp_hidden = mlp_hidden
        self.embed_hidden = embed_hidden

    def forward(self, img, verb, role_q):

        img_features, _ = self.conv(img)
        batch_size, n_channel, conv_h, conv_w = img_features.size()
        role_q = role_q.view(batch_size*self.max_role_count, -1)
        img = img_features.view(batch_size, n_channel, -1)
        img = img.permute(0, 2, 1)
        w_emb = self.w_emb(role_q)

        lstm_out, (h, _) = self.q_emb(w_emb)

        q_emb = h.permute(1, 0, 2).contiguous().view(batch_size*self.max_role_count, -1)
        q_emb = self.lstm_proj(q_emb)
        verb_embd = self.verb_transform(self.verb_lookup(verb))
        verb_embed_expand = verb_embd.expand(self.max_role_count, verb_embd.size(0), verb_embd.size(1))
        verb_embed_expand = verb_embed_expand.transpose(0,1)
        verb_embed_expand = verb_embed_expand.contiguous().view(-1, self.mlp_hidden)
        q_emb = self.q_prep(q_emb * verb_embed_expand)

        img = img.expand(self.max_role_count,img.size(0), img.size(1), img.size(2))
        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size* self.max_role_count, -1, self.mlp_hidden)

        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)

        role_label_pred = logits.contiguous().view(batch_size, -1, self.vocab_size)
        return role_label_pred


class BaseModel(nn.Module):
    def __init__(
            self,
            encoder,
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
        self.n_verbs = self.encoder.get_num_verbs()

        self.verb = verb_module(self.n_verbs)
        self.roles = role_module(self.encoder, self.gpu_mode)

    def train_preprocess(self):
        return self.train_transform

    def dev_preprocess(self):
        return self.dev_transform

    def forward(self, image):

        verb_pred = self.verb(image)

        sorted_idx = torch.sort(verb_pred, 1, True)[1]
        verb = sorted_idx[:,0]
        role_qs = self.encoder.get_role_questions_batch(verb)
        if self.gpu_mode >= 0:
            role_qs = role_qs.to(torch.device('cuda'))

        role_pred = self.roles(image, verb, role_qs)

        return verb_pred, role_pred


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



