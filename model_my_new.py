import torch
import torch.nn as nn
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet
import torch.nn.functional as F
import torchvision as tv
import utils
import numpy as np

class vgg16_modified(nn.Module):
    def __init__(self):
        super(vgg16_modified, self).__init__()
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

        self.verb_transform = nn.Linear(embed_hidden, mlp_hidden)
        self.v_att = Attention(mlp_hidden, mlp_hidden, mlp_hidden)


    def forward(self, img, q_emb):
        #batch_size = q.size(0)

        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1) # [batch, v_dim]

        return v_emb

class ControlUnit(nn.Module):
    def __init__(self, dim, max_step):
        super().__init__()

        self.position_aware = nn.ModuleList()
        for i in range(max_step):
            self.position_aware.append(nn.Linear(dim * 2, dim))

        self.control_question = nn.Linear(dim * 2, dim)
        self.attn = nn.Linear(dim, 1)

        self.dim = dim

    def forward(self, step, context, question, control):
        position_aware = self.position_aware[step](question)

        control_question = torch.cat([control, position_aware], 1)
        control_question = self.control_question(control_question)
        control_question = control_question.unsqueeze(1)

        context_prod = control_question * context
        attn_weight = self.attn(context_prod)

        attn = F.softmax(attn_weight, 1)

        next_control = (attn * context).sum(1)

        return next_control

class WriteUnit(nn.Module):
    def __init__(self, dim=512):
        super().__init__()

        self.concat = nn.Linear(dim * 2, dim)

    def forward(self, prev_mem, retrieved):
        concat = self.concat(torch.cat([retrieved, prev_mem], 1))
        next_mem = concat

        return next_mem

class RoleQHandler(nn.Module):
    def __init__(self,
                 dim=512,
                 max_step=4):
        super(RoleQHandler, self).__init__()

        self.control = ControlUnit(dim, max_step)
        self.vqa_model = TopDown()
        self.write = WriteUnit(dim)

        self.mem_0 = nn.Parameter(torch.zeros(1, dim))
        self.control_0 = nn.Parameter(torch.zeros(1, dim))

        self.dim = dim
        self.max_step = max_step


    def init_forward(self, img, q):
        q_emb , v_emb = self.vqa_model(img, q)
        return q_emb , v_emb

    def forward(self, img, q_words, q):
        b_size = img.size(0)

        control = self.control_0.expand(b_size, self.dim)
        memory = self.mem_0.expand(b_size, self.dim)

        #controls = [control]
        #memories = [memory]

        for i in range(self.max_step):
            control = self.control(i, q_words, q, control)

            #controls.append(control)

            curr_ans = self.vqa_model(img, control)
            memory = self.write(memory, curr_ans)
            #memories.append(memory)

        return memory


class ImSituationHandler(nn.Module):
    def __init__(self,
                 encoder,
                 role_lookup,
                 label_lookup,
                 qword_embeddings,
                 vocab_size,
                 gpu_mode,
                 mlp_hidden=512,
                 emd_hidden=300):
        super(ImSituationHandler, self).__init__()


        self.encoder = encoder
        self.role_lookup = role_lookup
        self.label_lookup = label_lookup
        self.qword_embeddings = qword_embeddings
        self.vocab_size = vocab_size
        self.gpu_mode = gpu_mode
        self.mlp_hidden = mlp_hidden
        self.emd_hidden = emd_hidden
        self.q_emb = nn.LSTM(self.emd_hidden, self.mlp_hidden,
                             batch_first=True, bidirectional=True)
        self.lstm_word_proj = nn.Linear(self.mlp_hidden * 2, self.mlp_hidden)
        self.role_handler = RoleQHandler()
        self.q_net = FCNet([self.mlp_hidden*2, self.mlp_hidden])
        self.v_net = FCNet([self.mlp_hidden, self.mlp_hidden])
        self.classifier = SimpleClassifier(
            self.mlp_hidden, 2 * self.mlp_hidden, self.vocab_size, 0.5)


    def forward(self, img, verb, labels):

        batch_size = img.size(0)
        role_qs, _ = self.encoder.get_role_questions_batch(verb)
        roles = self.encoder.get_role_ids_batch(verb)

        if self.gpu_mode >= 0:
            role_qs = role_qs.to(torch.device('cuda'))
            roles = roles.to(torch.device('cuda'))
        role_qs = role_qs.view(-1, role_qs.size(-1))

        role_qs = self.get_context_q(verb.size(0), roles, role_qs, labels)

        #w_emb = self.qword_embeddings(role_qs)
        w_emb = role_qs
        lstm_out, (h, _) = self.q_emb(w_emb)
        q_emb = h.permute(1, 0, 2).contiguous().view(batch_size, -1)

        lstm_out = self.lstm_word_proj(lstm_out)

        v_emb = self.role_handler(img, lstm_out, q_emb)

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)

        return logits

    def get_context_q(self, batch_size, roles, role_qs, labels):

        #get context
        mask = self.get_mask(batch_size, 6, 300)
        if self.gpu_mode >= 0:
            mask = mask.to(torch.device('cuda'))

        #just get first label for this out of 3
        labels = labels[:,0,:]

        #print('label size :', labels.size(), mask.size())
        role_prev = self.role_lookup(roles)
        role_prev = role_prev.expand(mask.size(1), role_prev.size(0), role_prev.size(1), role_prev.size(2))
        role_prev = role_prev.transpose(0,1)

        label_prev = self.label_lookup(labels)
        label_prev = label_prev.expand(mask.size(1), label_prev.size(0), label_prev.size(1), label_prev.size(2))
        label_prev = label_prev.transpose(0,1)

        role_prev_masked = mask * role_prev
        label_prev_masked = mask * label_prev

        roles_labels = torch.cat((role_prev_masked, label_prev_masked),2)
        roles_labels = roles_labels.view(roles_labels.size(0)*roles_labels.size(1), roles_labels.size(2), -1)
        #print('role ctx size :', roles_labels.size(), role_qs.size())
        updated_role_qs = torch.cat((roles_labels, self.qword_embeddings(role_qs)),1)
        #print('updated q :', updated_role_qs.size())

        return updated_role_qs

    def get_mask(self, batch_size, max_role_count, dim):
        id_mtx = torch.ones([max_role_count, max_role_count])
        np_tensor = id_mtx.numpy()
        np.fill_diagonal(np_tensor, 0)
        id_mtx = torch.from_numpy(np_tensor)
        id_mtx = (id_mtx.unsqueeze(-1)).unsqueeze(0)
        id_mtx = id_mtx.expand(batch_size, id_mtx.size(1), id_mtx.size(2), dim)

        #print('idx size :', id_mtx.size(), id_mtx[0][0], id_mtx[1][1])
        #check whether its correct

        return id_mtx


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
        self.role_lookup = nn.Embedding(self.n_roles + 1, embed_hidden, padding_idx=self.n_roles)
        self.ans_lookup = nn.Embedding(self.vocab_size + 1, embed_hidden, padding_idx=self.vocab_size)
        self.w_emb = nn.Embedding(self.n_role_q_vocab + 1, embed_hidden, padding_idx=self.n_role_q_vocab)

        self.vsrl_model = ImSituationHandler(self.encoder, self.role_lookup, self.ans_lookup,
                                             self.w_emb, self.vocab_size,
                                         self.gpu_mode)

        self.conv_hidden = self.conv.base_size()
        self.mlp_hidden = mlp_hidden
        self.embed_hidden = embed_hidden

    def train_preprocess(self):
        return self.train_transform

    def dev_preprocess(self):
        return self.dev_transform

    def forward(self, img, verb, labels):

        img_features = self.conv(img)
        batch_size, n_channel, conv_h, conv_w = img_features.size()
        img = img_features.view(batch_size, n_channel, -1)
        img = img.permute(0, 2, 1)

        img = img.expand(self.max_role_count,img.size(0), img.size(1), img.size(2))
        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size* self.max_role_count, -1, self.mlp_hidden)

        role_pred = self.vsrl_model(img, verb, labels)
        role_pred = role_pred.contiguous().view(batch_size, -1, self.vocab_size)

        #print('ans sizes :', verb_pred.size(), role_pred.size())

        return role_pred

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


