import torch
import torch.nn as nn
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet
import torch.nn.functional as F
import torchvision as tv
import math
import utils
import copy

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    #print('inside single att: query', query.size())
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    #print('scores :', scores)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    #print('att ', p_attn)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.rep = FCNet([d_model, d_model])

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.rep(x)

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
        features = self.vgg_features(x)
        y =  self.vgg_classifier(features.view(-1, 512*7*7))
        #print('y size :',  y.size())
        return features, y

class BaseModel(nn.Module):
    def __init__(self, encoder,
                 gpu_mode,
                 embed_hidden=300,
                 mlp_hidden=512):
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
        self.n_roles = self.encoder.get_num_roles()
        self.n_verbs = self.encoder.get_num_verbs()
        self.vocab_size = self.encoder.get_num_labels()
        self.max_role_count = self.encoder.get_max_role_count()
        self.n_role_q_vocab = len(self.encoder.question_words)

        self.conv = vgg16_modified()

        #self.verb_lookup = nn.Embedding(self.n_verbs, embed_hidden)
        self.w_emb = nn.Embedding(self.n_role_q_vocab + 1, embed_hidden, padding_idx=self.n_role_q_vocab)
        self.q_emb = nn.LSTM(embed_hidden, mlp_hidden,
                             batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(mlp_hidden * 2, mlp_hidden)
        #self.verb_transform = nn.Linear(embed_hidden, mlp_hidden)
        self.v_att = Attention(mlp_hidden, mlp_hidden, mlp_hidden)
        self.query_prep = FCNet([mlp_hidden*2, mlp_hidden])
        #self.lnorm = LayerNorm(mlp_hidden)
        self.multihead_att = MultiHeadedAttention(h=4, d_model=mlp_hidden)
        #self.dropout = nn.Dropout(0.1)
        #self.gate = nn.GRUCell(mlp_hidden, mlp_hidden)
        self.q_net = FCNet([mlp_hidden, mlp_hidden])
        self.v_net = FCNet([mlp_hidden, mlp_hidden])
        self.classifier = SimpleClassifier(
            mlp_hidden, 2 * mlp_hidden, self.vocab_size, 0.5)

        self.conv_hidden = self.conv.base_size()
        self.mlp_hidden = mlp_hidden
        self.embed_hidden = embed_hidden

    def train_preprocess(self):
        return self.train_transform

    def dev_preprocess(self):
        return self.dev_transform

    def forward(self, img, verb, role_q):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        #print('role size ', role_q.size())

        img_features, conv = self.conv(img)
        batch_size, n_channel, conv_h, conv_w = img_features.size()
        role_q = role_q.view(batch_size*self.max_role_count, -1)
        img = img_features.view(batch_size, n_channel, -1)
        img = img.permute(0, 2, 1)
        w_emb = self.w_emb(role_q)

        #do this so lstm ignored pads. i can't order by seq length as all roles of image should be together.
        #hence removing, so padded time steps are also included in the prediction
        '''embed = nn.utils.rnn.pack_padded_sequence(embed, question_length,
                                                  batch_first=True)'''
        lstm_out, (h, _) = self.q_emb(w_emb)
        #used to pad back what was removed by packing
        '''lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out,
                                                       batch_first=True)'''
        #q_emb = self.lstm_proj(lstm_out)
        q_emb = h.permute(1, 0, 2).contiguous().view(batch_size*self.max_role_count, -1)
        q_emb = self.lstm_proj(q_emb)
        '''verb_embd = self.verb_transform(self.verb_lookup(verb))
        verb_embed_expand = verb_embd.expand(self.max_role_count, verb_embd.size(0), verb_embd.size(1))
        verb_embed_expand = verb_embed_expand.transpose(0,1)
        verb_embed_expand = verb_embed_expand.contiguous().view(-1, self.mlp_hidden)
        q_emb = F.relu(q_emb * verb_embed_expand)'''
        #q_emb = self.q_emb(w_emb) # [batch, q_dim]

        img = img.expand(self.max_role_count,img.size(0), img.size(1), img.size(2))
        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size* self.max_role_count, -1, self.mlp_hidden)

        init_gate_hidden = torch.zeros(batch_size* self.max_role_count, self.mlp_hidden)
        if self.gpu_mode >= 0:
            init_gate_hidden = init_gate_hidden.to(torch.device('cuda'))
        #ans = [init_gate_hidden]

        mask = self.encoder.get_adj_matrix(verb)
        if self.gpu_mode >= 0:
            mask = mask.to(torch.device('cuda'))

        att = self.v_att(img, q_emb)
        v_emb_org = (att * img).sum(1)
        ans = [v_emb_org]
        '''for j in range(2):
             # [batch, v_dim]
            #v_emb = self.lnorm(v_emb_org)
            v_emb = ans[-1]
            v_emb = v_emb.view(batch_size, self.max_role_count, -1)
            v_emb = self.multihead_att(v_emb, v_emb, v_emb, mask)
            v_emb = v_emb.view(batch_size*self.max_role_count, -1)
            calc_ans = v_emb
            #gated_ans = self.gate(v_emb, ans[-1])
            ans.append(calc_ans)'''

        v_repr = ans[-1]
        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_repr)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)

        role_label_pred = logits.contiguous().view(batch_size, -1, self.vocab_size)
        return role_label_pred

    def calculate_loss(self, gt_verbs, role_label_pred, gt_labels,args):

        batch_size = role_label_pred.size()[0]
        if args.train_all:
            loss = 0
            for i in range(batch_size):
                for index in range(gt_labels.size()[1]):
                    frame_loss = 0
                    #verb_loss = utils.cross_entropy_loss(verb_pred[i], gt_verbs[i])
                    #frame_loss = criterion(role_label_pred[i], gt_labels[i,index])
                    for j in range(0, self.max_role_count):
                        frame_loss += utils.cross_entropy_loss(role_label_pred[i][j], gt_labels[i,index,j] ,self.vocab_size)
                    frame_loss = frame_loss/len(self.encoder.verb2_role_dict[self.encoder.verb_list[gt_verbs[i]]])
                    #print('frame loss', frame_loss, 'verb loss', verb_loss)
                    loss += frame_loss
        else:
            #verb from pre-trained
            loss = 0
            for i in range(batch_size):
                for index in range(gt_labels.size()[1]):
                    frame_loss = 0
                    #verb_loss = utils.cross_entropy_loss(verb_pred[i], gt_verbs[i])
                    #frame_loss = criterion(role_label_pred[i], gt_labels[i,index])
                    for j in range(0, self.max_role_count):
                        frame_loss += utils.cross_entropy_loss(role_label_pred[i][j], gt_labels[i,index,j] ,self.vocab_size)
                    frame_loss = frame_loss/len(self.encoder.verb2_role_dict[self.encoder.verb_list[gt_verbs[i]]])
                    #print('frame loss', frame_loss, 'verb loss', verb_loss)
                    loss += frame_loss


        final_loss = loss/batch_size
        #print('loss :', final_loss)
        return final_loss


def build_baseline0(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)


def build_baseline0_newatt(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)
