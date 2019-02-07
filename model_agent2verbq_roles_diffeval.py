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
        self.all_nouns_count = len(self.encoder.noun_list)
        self.max_role_count = self.encoder.max_role_count

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

        self.w_emb = nn.Embedding(self.q_word_count+1, embed_hidden, padding_idx=self.q_word_count)
        self.verb = TopDown()
        self.nouns = TopDown()

        self.q_net = FCNet([mlp_hidden, mlp_hidden])
        self.v_net = FCNet([mlp_hidden, mlp_hidden])

        self.classifier = SimpleClassifier(
            mlp_hidden, 2 * mlp_hidden, self.n_verbs, 0.5)

        self.noun_q_net = FCNet([mlp_hidden, mlp_hidden])
        self.noun_v_net = FCNet([mlp_hidden, mlp_hidden])

        self.noun_classifier = SimpleClassifier(
            mlp_hidden, 2 * mlp_hidden, self.all_nouns_count, 0.5)


    def train_preprocess(self):
        return self.train_transform

    def dev_preprocess(self):
        return self.dev_transform

    def forward(self, image, img_id, gt_verb):

        verb_qs = self.encoder.get_verb_questions_batch(img_id)
        if self.gpu_mode >= 0:
            verb_qs = verb_qs.to(torch.device('cuda'))

        verb_q_count = verb_qs.size(1)
        verb_qs = verb_qs.view(-1, verb_qs.size(-1))

        img_feat = self.conv_verb(image)
        batch_size, n_channel, conv_h, conv_w = img_feat.size()
        img_full = img_feat.view(batch_size, n_channel, -1)
        img_full = img_full.permute(0, 2, 1)

        img = img_full.expand(verb_q_count,img_full.size(0), img_full.size(1), img_full.size(2))
        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size* verb_q_count, -1, self.mlp_hidden)

        q_emb, v_emb = self.verb(img, self.w_emb(verb_qs))

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        verb_pred = self.classifier(joint_repr)

        verb_pred = verb_pred.contiguous().view(batch_size, -1, self.n_verbs)

        #role
        role_qs = self.encoder.get_role_questions_batch(gt_verb)
        if self.gpu_mode >= 0:
            role_qs = role_qs.to(torch.device('cuda'))

        role_embd = self.w_emb(role_qs)
        roleq = role_embd.view(-1, role_embd.size(-2), role_embd.size(-1))

        img_noun = img_full.expand(self.max_role_count,img_full.size(0), img_full.size(1), img_full.size(2))
        img_noun = img_noun.transpose(0,1)
        img_noun = img_noun.contiguous().view(batch_size* self.max_role_count, -1, self.mlp_hidden)

        q_emb_noun, v_emb_noun = self.nouns(img_noun, roleq)

        q_repr_noun = self.noun_q_net(q_emb_noun)
        v_repr_noun = self.noun_v_net(v_emb_noun)
        joint_repr_noun = q_repr_noun * v_repr_noun
        noun_pred = self.noun_classifier(joint_repr_noun)
        noun_pred = noun_pred.contiguous().view(batch_size, -1, self.all_nouns_count)

        return verb_pred, noun_pred

    def forward_eval(self, image, topk=5):

        conv_agent = self.conv_agent(image)

        agent_logit = self.agent(conv_agent)
        sorted_idx = torch.sort(agent_logit, 1, True)[1]

        current_agents = sorted_idx[:,:topk]

        img_feat = self.conv_verb(image)
        batch_size, n_channel, conv_h, conv_w = img_feat.size()
        img_full = img_feat.view(batch_size, n_channel, -1)
        img_full = img_full.permute(0, 2, 1)

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

        img = img_full.expand(topk,img_full.size(0), img_full.size(1), img_full.size(2))
        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size* topk, -1, self.mlp_hidden)

        q_emb, v_emb = self.verb(img, verb_q)

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        verb_pred = self.classifier(joint_repr)

        verb_pred_sm = F.log_softmax(verb_pred)
        n = 5
        verb_pred_sm = verb_pred_sm.contiguous().view(batch_size, -1)
        top_10_verbs, top_10_prob = self.get_top10_verbs(verb_pred_sm)

        if self.gpu_mode >= 0:
            top_10_verbs = top_10_verbs.to(torch.device('cuda'))
            top_10_prob = top_10_prob.to(torch.device('cuda'))

        #noun pred
        pred_verbs = top_10_verbs.view(batch_size*n)
        role_qs = self.encoder.get_role_questions_batch(pred_verbs)
        if self.gpu_mode >= 0:
            role_qs = role_qs.to(torch.device('cuda'))

        role_embd = self.w_emb(role_qs)
        roleq = role_embd.view(-1, role_embd.size(-2), role_embd.size(-1))

        img_noun = img_full.expand(self.max_role_count*n,img_full.size(0), img_full.size(1), img_full.size(2))
        img_noun = img_noun.transpose(0,1)
        img_noun = img_noun.contiguous().view(batch_size* self.max_role_count*n, -1, self.mlp_hidden)

        q_emb_noun, v_emb_noun = self.nouns(img_noun, roleq)

        q_repr_noun = self.noun_q_net(q_emb_noun)
        v_repr_noun = self.noun_v_net(v_emb_noun)
        joint_repr_noun = q_repr_noun * v_repr_noun
        noun_pred = self.noun_classifier(joint_repr_noun)

        noun_pred = noun_pred.contiguous().view(batch_size, n, 6, self.all_nouns_count)
        noun_pred_sm = F.log_softmax(noun_pred, dim=-1)
        sorted_nouns_prob = torch.sort(noun_pred_sm, -1, True)[0]
        best_nouns = sorted_nouns_prob[:,:,:,0]

        verb_mask = self.encoder.get_verb_role_mask(pred_verbs)

        if self.gpu_mode >= 0:
            verb_mask = verb_mask.to(torch.device('cuda'))
        verb_mask = verb_mask.view(batch_size, n, -1)

        masked_nouns = best_nouns * verb_mask

        all = torch.cat([top_10_prob.unsqueeze(-1), masked_nouns], -1)
        all_summed = torch.sum(all,-1)
        sorted_tot_idx = torch.sort(all_summed,-1,True)[1]

        final_verb_select = self.get_final_verb_selection(sorted_tot_idx, top_10_verbs)

        if self.gpu_mode >= 0:
            final_verb_select = final_verb_select.to(torch.device('cuda'))

        return final_verb_select

    def get_top10_verbs(self, verb_pred):
        batch_size = verb_pred.size(0)

        sorted_val_all, sorted_idx_all = torch.sort(verb_pred, 1, True)

        sorted_idx_dup = torch.remainder(sorted_idx_all, self.n_verbs)
        top10_idx = []
        top10_prob = []
        for b in range(batch_size):
            sorted_idx = []
            sorted_prob = []

            for q in range(sorted_idx_dup[b].size(0)):
                if sorted_idx_dup[b][q] not in sorted_idx:
                    sorted_idx.append(sorted_idx_dup[b][q].item())
                    sorted_prob.append(sorted_val_all[b][q].item())
                    if len(sorted_idx) == 5:
                        break

            sorted_idx = torch.tensor(sorted_idx, dtype=torch.long)
            sorted_prob = torch.tensor(sorted_prob, dtype=torch.float)

            top10_idx.append(sorted_idx)
            top10_prob.append(sorted_prob)

        return torch.stack(top10_idx,0), torch.stack(top10_prob,0)

    def get_final_verb_selection(self, sorted_idx, top10_verbs):

        batch_size = sorted_idx.size(0)
        final_res = []
        for b in range(batch_size):
            curr_idx = sorted_idx[b]
            prev_verbs = top10_verbs[b]
            top5 = []
            for idx in curr_idx.tolist():
                top5.append(prev_verbs[idx].item())
                if len(top5) == 5:
                    break
            final_res.append(torch.tensor(top5, dtype=torch.long))

        return torch.stack(final_res,0)


    def calculate_loss(self, verb_pred, gt_verbs, role_label_pred, gt_labels):

        batch_size = verb_pred.size()[0]
        verb_ref = verb_pred.size(1)
        loss = 0

        for i in range(batch_size):
            for index in range(verb_ref):
                frame_loss = 0
                verb_loss = utils.cross_entropy_loss(verb_pred[i][index], gt_verbs[i])
                for j in range(0, self.max_role_count):
                    frame_loss += utils.cross_entropy_loss(role_label_pred[i][j], gt_labels[i,index,j] ,self.all_nouns_count)
                frame_loss = verb_loss + frame_loss/len(self.encoder.verb2_role_dict[self.encoder.verb_list[gt_verbs[i]]])
                loss += frame_loss

        final_loss = loss/batch_size
        return final_loss



