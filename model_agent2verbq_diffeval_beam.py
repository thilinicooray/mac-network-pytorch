import torch
from torch import nn
import torch.nn.functional as F
import utils
import torchvision as tv
from classifier import SimpleClassifier
from fc import FCNet
from attention import Attention

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

class vgg16_modified_feat(nn.Module):
    def __init__(self):
        super(vgg16_modified_feat, self).__init__()
        vgg = tv.models.vgg16(pretrained=True)
        self.vgg_features = vgg.features

    def rep_size(self):
        return 1024

    def base_size(self):
        return 512

    def forward(self,x):
        #return self.dropout2(self.relu2(self.lin2(self.dropout1(self.relu1(self.lin1(self.vgg_features(x).view(-1, 512*7*7)))))))
        features = self.vgg_features(x)

        return features


class TopDown(nn.Module):
    def __init__(self,
                 embed_hidden=300,
                 mlp_hidden=512):
        super(TopDown, self).__init__()

        self.q_emb = nn.LSTM(embed_hidden, mlp_hidden,
                             batch_first=True, bidirectional=True)
        self.q_prep = FCNet([mlp_hidden, mlp_hidden])
        self.lstm_proj = nn.Linear(mlp_hidden * 2, mlp_hidden)
        self.verb_transform = nn.Linear(embed_hidden, mlp_hidden)
        self.v_att = Attention(mlp_hidden, mlp_hidden, mlp_hidden)


    def forward(self, img, q):
        batch_size = q.size(0)
        w_emb = q
        lstm_out, (h, _) = self.q_emb(w_emb)
        q_emb = h.permute(1, 0, 2).contiguous().view(batch_size, -1)
        q_emb = self.lstm_proj(q_emb)

        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1) # [batch, v_dim]

        return q_emb, v_emb


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
        self.mlp_hidden = mlp_hidden
        #self.vocab_size = self.encoder.get_num_labels()
        self.n_verbs = self.encoder.get_num_verbs()
        self.vocab_size = self.encoder.get_num_labels()

        self.q_word_count = len(self.encoder.question_words)


        self.conv_agent = vgg16_modified()
        self.conv_verb = vgg16_modified_feat()

        self.agent = nn.Sequential(
            nn.Linear(mlp_hidden*8, mlp_hidden*2),
            nn.BatchNorm1d(mlp_hidden*2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden*2, self.vocab_size),
        )

        self.w_emb = nn.Embedding(self.q_word_count, embed_hidden)
        self.verb = TopDown()

        self.q_net = FCNet([mlp_hidden, mlp_hidden])
        self.v_net = FCNet([mlp_hidden, mlp_hidden])

        self.classifier = SimpleClassifier(
            mlp_hidden, 2 * mlp_hidden, self.n_verbs, 0.5)

    def train_preprocess(self):
        return self.train_transform

    def dev_preprocess(self):
        return self.dev_transform

    def forward(self, image, img_id):

        verb_qs = self.encoder.get_verb_questions_batch(img_id)
        if self.gpu_mode >= 0:
            verb_qs = verb_qs.to(torch.device('cuda'))

        verb_q_count = verb_qs.size(1)
        verb_qs = verb_qs.view(-1, verb_qs.size(-1))

        img_feat = self.conv_verb(image)
        batch_size, n_channel, conv_h, conv_w = img_feat.size()
        img = img_feat.view(batch_size, n_channel, -1)
        img = img.permute(0, 2, 1)

        img = img.expand(verb_q_count,img.size(0), img.size(1), img.size(2))
        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size* verb_q_count, -1, self.mlp_hidden)

        q_emb, v_emb = self.verb(img, self.w_emb(verb_qs))

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        verb_pred = self.classifier(joint_repr)

        verb_pred = verb_pred.contiguous().view(batch_size, -1, self.n_verbs)

        return verb_pred

    def forward_eval(self, image, topk=10):

        conv_agent = self.conv_agent(image)

        agent_logit = self.agent(conv_agent)
        sorted_idx = torch.sort(agent_logit, 1, True)[1]
        #print('sorted ', sorted_idx.size())
        current_agents = sorted_idx[:,:topk]

        img_feat = self.conv_verb(image)
        batch_size, n_channel, conv_h, conv_w = img_feat.size()
        img = img_feat.view(batch_size, n_channel, -1)
        img = img.permute(0, 2, 1)

        q_word_idx = self.encoder.get_qword_idx_for_agentq(current_agents)
        verbq = self.encoder.common_q_idx
        if self.gpu_mode >= 0:
            q_word_idx = q_word_idx.to(torch.device('cuda'))
            verbq = verbq.to(torch.device('cuda'))

        verbq = verbq.unsqueeze(0)
        verbq = verbq.expand(topk, batch_size, verbq.size(1))
        verbq = verbq.view(topk * batch_size, -1)

        q_word_idx = q_word_idx.view(topk * batch_size, -1)

        verb_q = torch.cat([self.w_emb(verbq[:,:3]), self.w_emb(q_word_idx), self.w_emb(verbq[:,3:])], 1)

        #print('verbq :', verb_q.size())

        img = img.expand(topk,img.size(0), img.size(1), img.size(2))
        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size* topk, -1, self.mlp_hidden)

        q_emb, v_emb = self.verb(img, verb_q)

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        verb_pred = self.classifier(joint_repr)

        verb_pred = verb_pred.contiguous().view(batch_size, -1, self.n_verbs)
        sorted_verbs = torch.sort(verb_pred, -1, True)

        sorted_verbs_5_val = sorted_verbs[0][:, :, :5]
        sorted_verbs_5_idx = sorted_verbs[1][:, :, :5]
        verb_pred_top5 = self.get_top_5(sorted_verbs_5_val, sorted_verbs_5_idx)

        if self.gpu_mode >= 0:
            verb_pred_top5 = verb_pred_top5.to(torch.device('cuda'))

        return verb_pred_top5

    def get_top_5(self, all_res_val, all_res_idx):

        batch_size, ref, items = all_res_val.size()
        top_res = []
        for b in range(batch_size):
            all_data = {}

            for r in range(ref):
                for i in range(items):
                    all_data[all_res_val[b][r][i]] = all_res_idx[b][r][i]

            sorted_data = sorted(all_data.items(), reverse=True, key=lambda kv: kv[0])
            top5 = sorted_data[:5]
            top5_idx = []

            for t in top5:
                top5_idx.append(t[1])

            top_res.append(torch.tensor(top5_idx, dtype=torch.long))

        return torch.stack(top_res,0)


    def calculate_loss(self, verb_pred, gt_verbs):

        batch_size = verb_pred.size()[0]
        verb_ref = verb_pred.size(1)
        loss = 0
        #print('eval pred verbs :', pred_verbs)
        for i in range(batch_size):
            verb_loss = 0
            for r in range(verb_ref):
                verb_loss += utils.cross_entropy_loss(verb_pred[i][r], gt_verbs[i])
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



