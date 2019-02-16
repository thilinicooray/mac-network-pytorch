import torch
from torch import nn
import torch.nn.functional as F
import utils
import torchvision as tv
from classifier import SimpleClassifier
from fc import FCNet
from attention import Attention
from torch.nn.init import kaiming_uniform_, xavier_uniform_, normal

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
        self.attn = linear(dim, 1)

        self.dim = dim

    def forward(self, step, context, question, control):

        control_question = torch.cat([control, question], 1)
        control_question = self.control_question(control_question)
        control_question = control_question.unsqueeze(1)

        context_prod = control_question * context
        attn_weight = self.attn(context_prod)

        attn = F.softmax(attn_weight, 1)

        next_control = (attn * context).sum(1)

        return next_control


class ReadUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.mem = linear(dim, dim)
        self.concat = linear(dim * 2, dim)
        self.attn = linear(dim, 1)

    def forward(self, memory, know, control):
        mem = self.mem(memory[-1]).unsqueeze(2)
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

    def forward(self, context, question, knowledge):
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
            control = self.control(i, context, question, control)
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
    def __init__(self, n_vocab, dim, embed_hidden=300,
                 max_step=12, self_attention=False, memory_gate=False,
                 classes=28, dropout=0.15):
        super().__init__()

        #print('embed vocab size :', n_vocab)
        self.embed = nn.Embedding(n_vocab +1, embed_hidden)
        self.lstm = nn.LSTM(embed_hidden, dim,
                            batch_first=True, bidirectional=True)
        self.lstm_proj = linear(dim * 2, dim)
        self.full_q_lstm_proj = linear(dim * 2, dim)
        self.mac = MACUnit(dim, max_step,
                           self_attention, memory_gate, dropout)

        self.q_net = FCNet([dim, dim])
        self.v_net = FCNet([dim, dim])
        self.classifier = SimpleClassifier(
                                dim, 2 * dim, classes, 0.5)

        self.max_step = max_step
        self.dim = dim

        self.reset()

    def reset(self):
        self.embed.weight.data.uniform_(0, 1)

        '''kaiming_uniform_(self.conv[0].weight)
        self.conv[0].bias.data.zero_()
        kaiming_uniform_(self.conv[2].weight)
        self.conv[2].bias.data.zero_()'''

        #kaiming_uniform_(self.classifier[0].weight)

    def forward(self, image, question, dropout=0.15):
        b_size = question.size(0)
        #print('image size :', image.size(), b_size)
        #img = image.view(b_size, self.dim, -1)
        img = image
        #print('question :', question)
        embed = self.embed(question)

        lstm_out, (h, _) = self.lstm(embed)

        lstm_out = self.lstm_proj(lstm_out)
        h = h.permute(1, 0, 2).contiguous().view(b_size, -1)
        h = self.full_q_lstm_proj(h)
        memory = self.mac(lstm_out, h, img)

        #out = torch.cat([memory, h], 1)
        q_repr = self.q_net(h)
        v_repr = self.v_net(memory)
        joint_repr = q_repr * v_repr
        out = self.classifier(joint_repr)

        return out


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

        self.verb = MACNetwork(self.q_word_count, mlp_hidden, max_step=8, self_attention=False, memory_gate=False,
                                          classes=self.n_verbs)


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
        #img = img.permute(0, 2, 1)

        img = img.expand(verb_q_count,img.size(0), img.size(1), img.size(2))
        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size* verb_q_count, self.mlp_hidden, -1)

        verb_pred = self.verb(img, verb_qs)


        verb_pred = verb_pred.contiguous().view(batch_size, -1, self.n_verbs)

        return verb_pred

    def forward_eval(self, image, img_id):

        conv_agent = self.conv_agent(image)
        batch_size = image.size(0)

        #verb pred
        agent_logit = self.agent(conv_agent)
        sorted_idx = torch.sort(agent_logit, 1, True)[1]

        current_agent = sorted_idx[:,0]

        q_word_idx = self.encoder.get_qword_idx_for_agentq_top1(current_agent)
        verbq = self.encoder.common_q_idx
        places = self.encoder.get_places_batch(img_id)
        if self.gpu_mode >= 0:
            q_word_idx = q_word_idx.to(torch.device('cuda'))
            verbq = verbq.to(torch.device('cuda'))
            places = places.to(torch.device('cuda'))


        verbq = verbq.unsqueeze(0)
        verbq = verbq.expand(batch_size, verbq.size(1))
        verbq = verbq.view(batch_size, -1)

        q_word_idx = q_word_idx.view(batch_size, -1)

        verb_q = torch.cat([verbq[:,:3], q_word_idx, verbq[:,4:-1], places.unsqueeze(1)], 1)

        img_feat = self.conv_verb(image)
        batch_size, n_channel, conv_h, conv_w = img_feat.size()
        img = img_feat.view(batch_size, n_channel, -1)
        #img = img.permute(0, 2, 1)

        verb_pred = self.verb(img, verb_q)

        return verb_pred



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



