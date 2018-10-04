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


class Propogator(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """
    def __init__(self, state_dim, n_node, n_edge_types):
        super(Propogator, self).__init__()

        self.n_node = n_node
        self.n_edge_types = n_edge_types

        self.reset_gate = nn.Sequential(
            linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            linear(state_dim*3, state_dim),
            nn.Tanh()
        )

    def forward(self, state_in, state_out, state_cur, A):
        A_in = A[:, :, :self.n_node*self.n_edge_types]
        A_out = A[:, :, :self.n_node*self.n_edge_types]

        a_in = torch.bmm(A_in, state_in)
        a_out = torch.bmm(A_out, state_out)
        a = torch.cat((a_in, a_out, state_cur), 2)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, a_out, r * state_cur), 2)
        h_hat = self.tansform(joined_input)

        output = (1 - z) * state_cur + z * h_hat

        return output


class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, state_dim, question_dim, n_edge_types,n_node,  n_steps, n_classes):
        super(GGNN, self).__init__()

        self.state_dim = state_dim
        self.annotation_dim = question_dim
        self.n_edge_types = n_edge_types
        self.n_node = n_node
        self.n_steps = n_steps

        for i in range(self.n_edge_types):
            # incoming and outgoing edge embedding
            in_fc = linear(self.state_dim, self.state_dim)
            out_fc = linear(self.state_dim, self.state_dim)
            self.add_module("in_{}".format(i), in_fc)
            self.add_module("out_{}".format(i), out_fc)

        self.in_fcs = AttrProxy(self, "in_")
        self.out_fcs = AttrProxy(self, "out_")

        # Propogation Model
        self.propogator = Propogator(self.state_dim, self.n_node, self.n_edge_types)

        # Output Model
        self.out = nn.Sequential(
            linear(self.state_dim + self.annotation_dim, self.state_dim),
            nn.Tanh(),
            linear(self.state_dim, n_classes)
        )


    def forward(self, prop_state, question, A):
        for i_step in range(self.n_steps):
            in_states = []
            out_states = []
            for i in range(self.n_edge_types):
                in_states.append(self.in_fcs[i](prop_state))
                out_states.append(self.out_fcs[i](prop_state))
            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            in_states = in_states.view(-1, self.n_node*self.n_edge_types, self.state_dim)
            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.view(-1, self.n_node*self.n_edge_types, self.state_dim)

            prop_state = self.propogator(in_states, out_states, prop_state, A)

        join_state = torch.cat((prop_state, question), 2)

        output = self.out(join_state)
        #output = output.sum(2)

        return output


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
        self.mlp_hidden = mlp_hidden
        self.embed_hidden = embed_hidden
        self.role_lookup = nn.Embedding(self.n_roles+1, embed_hidden, padding_idx=self.n_roles)
        self.verb_lookup = nn.Embedding(self.n_verbs, embed_hidden)

        self.img_transform = nn.Sequential(
            linear(mlp_hidden*8, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.embd_transform = nn.Sequential(
            linear(embed_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.role_labeller = GGNN(state_dim = self.mlp_hidden, question_dim=self.embed_hidden, n_edge_types=1,n_node=self.max_role_count,
                                  n_steps=4, n_classes=self.vocab_size)

        self.conv_hidden = self.conv.base_size()



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
        role_verb_embd_org = role_verb_embd.transpose(0,1)
        role_verb_embd = self.embd_transform(role_verb_embd_org)
        img_features_role = conv.repeat(1,self.max_role_count, 1, 1)
        img_features_role = img_features_role.view(batch_size, self.max_role_count, -1)
        img = self.img_transform(img_features_role)
        img_q_fused = img * role_verb_embd


        mask = self.encoder.get_adj_matrix(verbs)
        if self.gpu_mode >= 0:
            mask = mask.to(torch.device('cuda'))

        role_label_pred = self.role_labeller(img_q_fused, role_verb_embd_org, mask)

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
            role_verb_embd_org = role_verb_embd.transpose(0,1)
            role_verb_embd = self.embd_transform(role_verb_embd_org)
            img_features_role = conv.repeat(1,self.max_role_count, 1, 1)
            img_features_role = img_features_role.view(batch_size, self.max_role_count, -1)
            img = self.img_transform(img_features_role)
            img_q_fused = img * role_verb_embd


            mask = self.encoder.get_adj_matrix(topk_verb)
            if self.gpu_mode >= 0:
                mask = mask.to(torch.device('cuda'))

            role_label_pred = self.role_labeller(img_q_fused, role_verb_embd_org, mask)

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
