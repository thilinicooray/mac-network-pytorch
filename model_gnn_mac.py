#original code from https://github.com/JamesChuanggg/ggnn.pytorch/blob/master/model.py

import torch
from torch.autograd import Variable
from torch import nn
from torch.nn.init import kaiming_uniform_, xavier_uniform_, normal
import torch.nn.functional as F
import torchvision as tv
import utils
import math
import copy

def linear(in_dim, out_dim, bias=True):
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()

    return lin

class ControlUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()

        #self.control_question = linear(dim * 2, dim)

        self.dim = dim

    def forward(self, question, control):
        #most simple : just pass the input

        next_control = question
        return next_control


class ReadUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.mem = linear(dim*3, dim)
        self.concat = linear(dim * 2, dim)
        self.attn = linear(dim, 1)

    def forward(self, memory, know, control):
        mem = self.mem(memory).unsqueeze(-1)
        concat = self.concat(torch.cat([mem * know, know], 2) \
                             .permute(0, 1, 3, 2))
        attn = concat * control.unsqueeze(2)
        attn = self.attn(attn).squeeze(3)
        attn = F.softmax(attn, 2).unsqueeze(2)
        read = (attn * know).sum(3)

        return read


class WriteUnit(nn.Module):
    def __init__(self, dim, self_attention=False, memory_gate=False):
        super().__init__()

        if self_attention:
            self.attn = linear(dim, 1)
            self.mem = linear(dim, dim)

        if memory_gate:
            self.control = linear(dim, 1)

        self.self_attention = self_attention
        self.memory_gate = memory_gate

    def forward(self, retrieved):

        next_mem = retrieved

        '''if self.self_attention:
            controls_cat = torch.stack(controls[:-1], 2)
            attn = controls[-1].unsqueeze(2) * controls_cat
            attn = self.attn(attn.permute(0, 2, 1))
            attn = F.softmax(attn, 1).permute(0, 2, 1)

            memories_cat = torch.stack(memories, 2)
            attn_mem = (attn * memories_cat).sum(2)
            next_mem = self.mem(attn_mem) + retrieved

        if self.memory_gate:
            control = self.control(controls[-1])
            gate = F.sigmoid(control)
            next_mem = gate * prev_mem + (1 - gate) * next_mem'''

        return next_mem

class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class GMACUnit(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """
    def __init__(self, state_dim, n_node, n_edge_types, self_attention=False, memory_gate=False):
        super(GMACUnit, self).__init__()

        self.n_node = n_node
        self.n_edge_types = n_edge_types

        self.control = ControlUnit(state_dim)
        self.read = ReadUnit(state_dim)
        self.write = WriteUnit(state_dim, self_attention, memory_gate)

        self.tansform = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.ELU()
        )

    def forward(self, question, knowledge, in_states, out_states, memory,control, A, control_mask, memory_mask):
        A_in = A[:, :, :self.n_node*self.n_edge_types]
        A_out = A[:, :, :self.n_node*self.n_edge_types]
        #both are same as we don't know the exact structure

        control = self.control(question, control)
        if self.training:
            control = control * control_mask
        a_in = torch.bmm(A_in, in_states)
        a_out = torch.bmm(A_out, out_states)
        a = torch.cat((a_in, a_out, memory), 2)

        read = self.read(a, knowledge, control)

        joined_input = torch.cat((a_in, a_out, read * memory), 2)
        transformed = self.tansform(joined_input)

        memory = self.write(transformed)
        if self.training:
            memory = memory * memory_mask

        return memory, control


class GMACNetwork(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, state_dim, n_steps, n_node, self_attention=False, memory_gate=False,
                 classes=2000, dropout=0.15):
        super(GMACNetwork, self).__init__()

        self.state_dim = state_dim
        self.n_edge_types = 1
        self.n_node = n_node
        self.n_steps = n_steps
        self.dropout = dropout

        for i in range(self.n_edge_types):
            # incoming and outgoing edge embedding
            in_fc = linear(self.state_dim, self.state_dim)
            out_fc = linear(self.state_dim, self.state_dim)
            self.add_module("in_{}".format(i), in_fc)
            self.add_module("out_{}".format(i), out_fc)

        self.in_fcs = AttrProxy(self, "in_")
        self.out_fcs = AttrProxy(self, "out_")

        self.mem_0 = nn.Parameter(torch.zeros(1,self.n_node, self.state_dim))
        self.control_0 = nn.Parameter(torch.zeros(1,self.n_node, self.state_dim))

        # Propogation Model
        self.propogator = GMACUnit(self.state_dim, self.n_node, self.n_edge_types, self_attention=False, memory_gate=False)

        # Output Model
        self.classifier = nn.Sequential(linear(self.state_dim * 2, self.state_dim),
                                        nn.ELU(),
                                        linear(self.state_dim, classes))

    def get_mask(self, x, dropout):
        mask = torch.empty_like(x).bernoulli_(1 - dropout)
        mask = mask / (1 - dropout)

        return mask

    def forward(self, question, knowledge, A):

        b_size = question.size(0)

        control = self.control_0.expand(b_size, self.n_node, self.state_dim)
        memory = self.mem_0.expand(b_size, self.n_node, self.state_dim)

        if self.training:
            control_mask = self.get_mask(control, self.dropout)
            memory_mask = self.get_mask(memory, self.dropout)
            control = control * control_mask
            memory = memory * memory_mask
        else:
            control_mask = None
            memory_mask = None

        for i_step in range(self.n_steps):
            in_states = []
            out_states = []
            for i in range(self.n_edge_types):
                in_states.append(self.in_fcs[i](memory))
                out_states.append(self.out_fcs[i](memory))
            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            in_states = in_states.view(-1, self.n_node*self.n_edge_types, self.state_dim)
            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.view(-1, self.n_node*self.n_edge_types, self.state_dim)

            memory, control = self.propogator(question, knowledge, in_states, out_states, memory,control,
                                              A, control_mask, memory_mask)

        out = torch.cat([memory, question], 2)

        out = self.classifier(out)

        return out

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


class E2ENetwork(nn.Module):
    def __init__(
            self,
            encoder,
            gpu_mode,
            embed_hidden=300,
            mlp_hidden=512,
            gmac_enabled=True
    ):
        super(E2ENetwork, self).__init__()

        self.normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.gmac_enabled = gmac_enabled

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

        self.verb = nn.Sequential(
            linear(mlp_hidden*8, mlp_hidden*2),
            nn.BatchNorm1d(mlp_hidden*2),
            nn.ReLU(),
            nn.Dropout(0.5),
            linear(mlp_hidden*2, self.n_verbs),
        )
        #todo: init embedding
        self.role_lookup = nn.Embedding(self.n_roles+1, embed_hidden, padding_idx=self.n_roles)
        self.verb_lookup = nn.Embedding(self.n_verbs, embed_hidden)

        self.q_trasform = linear(embed_hidden, mlp_hidden)

        self.role_labeller = GMACNetwork(mlp_hidden, 4, self.max_role_count, self_attention=False, memory_gate=False,
                                        classes=self.vocab_size)

        self.conv_hidden = self.conv.base_size()
        self.mlp_hidden = mlp_hidden
        self.embed_hidden = embed_hidden


    def train_preprocess(self):
        return self.train_transform

    def dev_preprocess(self):
        return self.dev_transform

    def forward(self, image, verbs, roles, mask=None):

        img_features, conv = self.conv(image)
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        #verb pred
        verb_pred = self.verb(conv)

        verb_embd = self.verb_lookup(verbs)
        role_embd = self.role_lookup(roles)

        role_embed_reshaped = role_embd.transpose(0,1)
        verb_embed_expand = verb_embd.expand(self.max_role_count, verb_embd.size(0), verb_embd.size(1))
        role_verb_embd = verb_embed_expand * role_embed_reshaped
        role_verb_embd = role_verb_embd.transpose(0,1)
        role_verb_embd = self.q_trasform(role_verb_embd)
        img_features = img_features.repeat(1,self.max_role_count, 1, 1)
        img_features = img_features.view(batch_size, self.max_role_count, self.mlp_hidden, -1)


        mask = self.encoder.get_adj_matrix(verbs)
        if self.gpu_mode >= 0:
            mask = mask.to(torch.device('cuda'))

        role_label_pred = self.role_labeller(role_verb_embd, img_features, mask)

        #role_label_pred = role_label_pred.contiguous().view(batch_size, -1, self.vocab_size)

        return verb_pred, role_label_pred


    def forward_eval5(self, image, topk = 5, mask=None):

        img_features_org, conv = self.conv(image)
        batch_size, n_channel, conv_h, conv_w = img_features_org.size()
        beam_role_idx = None
        top1role_label_pred = None

        #verb pred
        verb_pred = self.verb(conv)

        sorted_idx = torch.sort(verb_pred, 1, True)[1]
        #print('sorted ', sorted_idx.size())
        verbs = sorted_idx[:,:topk]
        #print('size verbs :', verbs.size())
        #print('top1 verbs', verbs)

        #print('verbs :', verbs.size(), verbs)
        for k in range(0,topk):
            img_features = img_features_org
            #print('k :', k)
            topk_verb = verbs[:,k]
            #print('ver size :', topk_verb.size())
            roles = self.encoder.get_role_ids_batch(topk_verb)

            roles = roles.type(torch.LongTensor)
            topk_verb = topk_verb.type(torch.LongTensor)

            if self.gpu_mode >= 0:
                roles = roles.to(torch.device('cuda'))
                topk_verb = topk_verb.to(torch.device('cuda'))

            verb_embd = self.verb_lookup(topk_verb)
            role_embd = self.role_lookup(roles)

            role_embed_reshaped = role_embd.transpose(0,1)
            verb_embed_expand = verb_embd.expand(self.max_role_count, verb_embd.size(0), verb_embd.size(1))
            role_verb_embd = verb_embed_expand * role_embed_reshaped
            role_verb_embd = role_verb_embd.transpose(0,1)
            role_verb_embd = self.q_trasform(role_verb_embd)
            img_features = img_features.repeat(1,self.max_role_count, 1, 1)
            img_features = img_features.view(batch_size, self.max_role_count, self.mlp_hidden, -1)


            mask = self.encoder.get_adj_matrix(topk_verb)
            if self.gpu_mode >= 0:
                mask = mask.to(torch.device('cuda'))

            role_label_pred = self.role_labeller(role_verb_embd, img_features, mask)
            #print('role_label_pred' , role_label_pred.size())

            #role_label_pred = role_label_pred.contiguous().view(batch_size, -1, self.vocab_size)
            #print('role_label_pred view' , role_label_pred.size())

            if k == 0:
                top1role_label_pred = role_label_pred
                idx = torch.max(role_label_pred,-1)[1]
                #print(idx[1])
                beam_role_idx = idx
            else:
                idx = torch.max(role_label_pred,-1)[1]
                beam_role_idx = torch.cat((beam_role_idx.clone(), idx), 1)
            if self.gpu_mode >= 0:
                torch.cuda.empty_cache()

        #print('role idx size :', beam_role_idx.size(), top1role_label_pred.size())

        return verb_pred, top1role_label_pred, beam_role_idx


    def calculate_loss(self, verb_pred, gt_verbs, role_label_pred, gt_labels,args):

        batch_size = verb_pred.size()[0]
        if args.train_all:
            loss = 0
            for i in range(batch_size):
                for index in range(gt_labels.size()[1]):
                    frame_loss = 0
                    verb_loss = utils.cross_entropy_loss(verb_pred[i], gt_verbs[i])
                    #frame_loss = criterion(role_label_pred[i], gt_labels[i,index])
                    for j in range(0, self.max_role_count):
                        frame_loss += utils.cross_entropy_loss(role_label_pred[i][j], gt_labels[i,index,j] ,self.vocab_size)
                    frame_loss = verb_loss + frame_loss/len(self.encoder.verb2_role_dict[self.encoder.verb_list[gt_verbs[i]]])
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

    def calculate_eval_loss(self, verb_pred, gt_verbs, role_label_pred, gt_labels,args):

        batch_size = verb_pred.size()[0]

        sorted_idx = torch.sort(verb_pred, 1, True)[1]
        pred_verbs = sorted_idx[:,0]
        #print('eval pred verbs :', pred_verbs)
        if args.train_all:
            loss = 0
            for i in range(batch_size):
                for index in range(gt_labels.size()[1]):
                    frame_loss = 0
                    verb_loss = utils.cross_entropy_loss(verb_pred[i], gt_verbs[i])
                    gt_role_list = self.encoder.get_role_ids(gt_verbs[i])
                    pred_role_list = self.encoder.get_role_ids(pred_verbs[i])

                    #print ('role list diff :', gt_role_list, pred_role_list)

                    for j in range(0, self.max_role_count):
                        if pred_role_list[j] == len(self.encoder.role_list):
                            continue
                        if pred_role_list[j] in gt_role_list:
                            #print('eval loss :', gt_role_list, pred_role_list[j])
                            g_idx = (gt_role_list == pred_role_list[j]).nonzero()
                            #print('found idx' , g_idx)
                            frame_loss += utils.cross_entropy_loss(role_label_pred[i][j], gt_labels[i,index,g_idx] ,self.vocab_size)

                    frame_loss = verb_loss + frame_loss/len(self.encoder.verb2_role_dict[self.encoder.verb_list[gt_verbs[i]]])
                    #print('frame loss', frame_loss)
                    loss += frame_loss
        else:
            loss = 0
            for i in range(batch_size):
                for index in range(gt_labels.size()[1]):
                    frame_loss = 0
                    verb_loss = utils.cross_entropy_loss(verb_pred[i], gt_verbs[i])
                    gt_role_list = self.encoder.get_role_ids(gt_verbs[i])
                    pred_role_list = self.encoder.get_role_ids(pred_verbs[i])

                    #print ('role list diff :', gt_role_list, pred_role_list)

                    for j in range(0, self.max_role_count):
                        if pred_role_list[j] == len(self.encoder.role_list):
                            continue
                        if pred_role_list[j] in gt_role_list:
                            #print('eval loss :', gt_role_list, pred_role_list[j])
                            g_idx = (gt_role_list == pred_role_list[j]).nonzero()
                            #print('found idx' , g_idx)
                            frame_loss += utils.cross_entropy_loss(role_label_pred[i][j], gt_labels[i,index,g_idx] ,self.vocab_size)

                    frame_loss = frame_loss/len(self.encoder.verb2_role_dict[self.encoder.verb_list[gt_verbs[i]]])
                    #print('frame loss', frame_loss)
                    loss += frame_loss


        final_loss = loss/batch_size
        #print('loss :', final_loss)
        return final_loss

    def calculate_eval_loss_new(self, verb_pred, gt_verbs, role_label_pred, gt_labels,args):

        batch_size = verb_pred.size()[0]

        sorted_idx = torch.sort(verb_pred, 1, True)[1]
        pred_verbs = sorted_idx[:,0]
        #print('eval pred verbs :', pred_verbs)
        if args.train_all:
            loss = 0
            for i in range(batch_size):
                for index in range(gt_labels.size()[1]):
                    frame_loss = 0
                    verb_loss = utils.cross_entropy_loss(verb_pred[i], gt_verbs[i])
                    gt_role_list = self.encoder.get_role_ids(gt_verbs[i])
                    pred_role_list = self.encoder.get_role_ids(pred_verbs[i])
                    matching_role_count = 0

                    #print ('role list diff :', gt_role_list, pred_role_list)

                    for j in range(0, self.max_role_count):
                        if pred_role_list[j] == len(self.encoder.role_list):
                            continue
                        if pred_role_list[j] in gt_role_list:
                            #print('eval loss :', gt_role_list, pred_role_list[j])
                            g_idx = (gt_role_list == pred_role_list[j]).nonzero()
                            #print('found idx' , g_idx)
                            frame_loss += utils.cross_entropy_loss(role_label_pred[i][j], gt_labels[i,index,g_idx] ,self.vocab_size)
                            matching_role_count +=1
                    if matching_role_count > 0:
                        frame_loss = verb_loss + frame_loss/matching_role_count
                        #print('frame loss', frame_loss)
                        loss += frame_loss
        else:
            loss = 0
            for i in range(batch_size):
                for index in range(gt_labels.size()[1]):
                    frame_loss = 0
                    verb_loss = utils.cross_entropy_loss(verb_pred[i], gt_verbs[i])
                    gt_role_list = self.encoder.get_role_ids(gt_verbs[i])
                    pred_role_list = self.encoder.get_role_ids(pred_verbs[i])
                    matching_role_count = 0
                    #print ('role list diff :', gt_role_list, pred_role_list)

                    for j in range(0, self.max_role_count):
                        if pred_role_list[j] == len(self.encoder.role_list):
                            continue
                        if pred_role_list[j] in gt_role_list:
                            #print('eval loss :', gt_role_list, pred_role_list[j])
                            g_idx = (gt_role_list == pred_role_list[j]).nonzero()
                            #print('found idx' , g_idx)
                            frame_loss += utils.cross_entropy_loss(role_label_pred[i][j], gt_labels[i,index,g_idx] ,self.vocab_size)
                            matching_role_count +=1

                    if matching_role_count > 0:
                        frame_loss = frame_loss/matching_role_count
                        #print('frame loss', frame_loss)
                        loss += frame_loss


        final_loss = loss/batch_size
        #print('loss :', final_loss)
        return final_loss
