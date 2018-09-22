import torch
from torch.autograd import Variable
from torch import nn
from torch.nn.init import kaiming_uniform_, xavier_uniform_, normal
import torch.nn.functional as F
import torchvision as tv
import utils

def linear(in_dim, out_dim, bias=True):
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()

    return lin

class ControlUnit(nn.Module):
    def __init__(self, dim, max_step):
        super().__init__()

        self.control_question = linear(dim * 2, dim)

        self.dim = dim

    def forward(self, step, question, control):
        #most simple : just pass the input

        next_control = question
        return next_control


class ReadUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.mem = linear(dim, dim)
        self.concat = linear(dim * 2, dim)
        self.attn = linear(dim, 1)

    def forward(self, memory, know, control):
        mem = self.mem(memory[-1]).unsqueeze(2)
        #print('read concat :', mem.size(), know.size())
        concat = self.concat(torch.cat([mem * know, know], 1) \
                             .permute(0, 2, 1))
        attn = concat * control[-1].unsqueeze(1)
        attn = self.attn(attn).squeeze(2)
        attn = F.softmax(attn, 1).unsqueeze(1)

        read = (attn * know).sum(2)

        return read


class WriteUnit(nn.Module):
    def __init__(self, dim, self_attention=False, memory_gate=False):
        super().__init__()

        self.concat = linear(dim * 2, dim)

        if self_attention:
            self.attn = linear(dim, 1)
            self.mem = linear(dim, dim)

        if memory_gate:
            self.control = linear(dim, 1)

        self.self_attention = self_attention
        self.memory_gate = memory_gate

    def forward(self, memories, retrieved, controls):
        prev_mem = memories[-1]
        concat = self.concat(torch.cat([retrieved, prev_mem], 1))
        next_mem = concat

        if self.self_attention:
            controls_cat = torch.stack(controls[:-1], 2)
            attn = controls[-1].unsqueeze(2) * controls_cat
            attn = self.attn(attn.permute(0, 2, 1))
            attn = F.softmax(attn, 1).permute(0, 2, 1)

            memories_cat = torch.stack(memories, 2)
            attn_mem = (attn * memories_cat).sum(2)
            next_mem = self.mem(attn_mem) + concat

        if self.memory_gate:
            control = self.control(controls[-1])
            gate = F.sigmoid(control)
            next_mem = gate * prev_mem + (1 - gate) * next_mem

        return next_mem


class MACUnit(nn.Module):
    def __init__(self, dim, max_step=12,
                 self_attention=False, memory_gate=False,
                 dropout=0.15):
        super().__init__()

        self.control = ControlUnit(dim, max_step)
        self.read = ReadUnit(dim)
        self.write = WriteUnit(dim, self_attention, memory_gate)

        self.mem_0 = nn.Parameter(torch.zeros(1, dim))
        self.control_0 = nn.Parameter(torch.zeros(1, dim))

        self.dim = dim
        self.max_step = max_step
        self.dropout = dropout

    def get_mask(self, x, dropout):
        mask = torch.empty_like(x).bernoulli_(1 - dropout)
        mask = mask / (1 - dropout)

        return mask

    def forward(self, question, knowledge):
        b_size = question.size(0)

        control = self.control_0.expand(b_size, self.dim)
        memory = self.mem_0.expand(b_size, self.dim)

        if self.training:
            control_mask = self.get_mask(control, self.dropout)
            memory_mask = self.get_mask(memory, self.dropout)
            control = control * control_mask
            memory = memory * memory_mask

        controls = [control]
        memories = [memory]

        for i in range(self.max_step):
            control = self.control(i, question, control)
            if self.training:
                control = control * control_mask
            controls.append(control)

            read = self.read(memories, knowledge, controls)
            memory = self.write(memories, read, controls)
            if self.training:
                memory = memory * memory_mask
            memories.append(memory)

        return memory


class MACNetwork(nn.Module):
    def __init__(self, dim,
                 max_step=12, self_attention=False, memory_gate=False,
                 classes=28, dropout=0.15):
        super().__init__()

        self.q_trasform = linear(300, dim)
        self.mac = MACUnit(dim, max_step,
                           self_attention, memory_gate, dropout)


        self.classifier = nn.Sequential(linear(dim * 2, dim),
                                        nn.ELU(),
                                        linear(dim, classes))

        self.max_step = max_step
        self.dim = dim

        self.reset()

    def reset(self):
        kaiming_uniform_(self.classifier[0].weight)

    def forward(self, image, question, dropout=0.15):
        b_size = question.size(0)
        transformed_q = self.q_trasform(question)
        img = image.view(b_size, self.dim, -1)
        memory = self.mac(transformed_q, img)

        out = torch.cat([memory, transformed_q], 1)
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
            mlp_hidden=512
    ):
        super(E2ENetwork, self).__init__()

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

        self.role_labeller = MACNetwork(mlp_hidden, max_step=4, self_attention=False, memory_gate=False,
                                        classes=self.vocab_size)

        self.conv_hidden = self.conv.base_size()
        self.mlp_hidden = mlp_hidden
        self.embed_hidden = embed_hidden


    def train_preprocess(self):
        return self.train_transform

    def dev_preprocess(self):
        return self.dev_transform

    def forward(self, image, verbs, roles):

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
        role_verb_embd = role_verb_embd.contiguous().view(-1, self.embed_hidden)
        img_features = img_features.repeat(1,self.max_role_count, 1, 1)
        img_features = img_features.view(-1, n_channel, conv_h, conv_w)

        role_label_pred = self.role_labeller(img_features, role_verb_embd)

        role_label_pred = role_label_pred.contiguous().view(batch_size, -1, self.vocab_size)

        return verb_pred, role_label_pred

    def forward_eval(self, image):

        img_features, conv = self.conv(image)
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        #verb pred
        verb_pred = self.verb(conv)

        sorted_idx = torch.sort(verb_pred, 1, True)[1]
        #print('sorted ', sorted_idx.size())
        verbs = sorted_idx[:,0]
        #todo:change for 5
        #print('top1 verbs', verbs)

        #print('verbs :', verbs.size(), verbs)

        roles = self.encoder.get_role_ids_batch(verbs)

        roles = roles.type(torch.LongTensor)
        verbs = verbs.type(torch.LongTensor)

        if self.gpu_mode >= 0:
            roles = roles.to(torch.device('cuda'))
            verbs = verbs.to(torch.device('cuda'))

        verb_embd = self.verb_lookup(verbs)
        role_embd = self.role_lookup(roles)

        role_embed_reshaped = role_embd.transpose(0,1)
        verb_embed_expand = verb_embd.expand(self.max_role_count, verb_embd.size(0), verb_embd.size(1))
        role_verb_embd = verb_embed_expand * role_embed_reshaped
        role_verb_embd = role_verb_embd.transpose(0,1)
        role_verb_embd = role_verb_embd.contiguous().view(-1, self.embed_hidden)
        img_features = img_features.repeat(1,self.max_role_count, 1, 1)
        img_features = img_features.view(-1, n_channel, conv_h, conv_w)

        role_label_pred = self.role_labeller(img_features, role_verb_embd)

        role_label_pred = role_label_pred.contiguous().view(batch_size, -1, self.vocab_size)

        return verb_pred, role_label_pred

    def forward_eval5(self, image, topk = 5):

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
            role_verb_embd = role_verb_embd.contiguous().view(-1, self.embed_hidden)
            img_features = img_features.repeat(1,self.max_role_count, 1, 1)
            img_features = img_features.view(-1, n_channel, conv_h, conv_w)

            role_label_pred = self.role_labeller(img_features, role_verb_embd)

            role_label_pred = role_label_pred.contiguous().view(batch_size, -1, self.vocab_size)

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
