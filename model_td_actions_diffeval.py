import torch
import torch.nn as nn
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet
import torch.nn.functional as F
import torchvision as tv
import utils


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
        self.n_q_vocab = len(self.encoder.question_words)

        self.conv = vgg16_modified()

        self.w_emb = nn.Embedding(self.n_q_vocab + 1, embed_hidden, padding_idx=self.n_q_vocab)
        self.q_emb = nn.LSTM(embed_hidden, mlp_hidden,
                             batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(mlp_hidden * 2, mlp_hidden)
        self.v_att = Attention(mlp_hidden, mlp_hidden, mlp_hidden)
        self.q_net = FCNet([mlp_hidden, mlp_hidden])
        self.v_net = FCNet([mlp_hidden, mlp_hidden])
        self.classifier = SimpleClassifier(
            mlp_hidden, 2 * mlp_hidden, self.n_verbs, 0.5)

        self.conv_hidden = self.conv.base_size()
        self.mlp_hidden = mlp_hidden
        self.embed_hidden = embed_hidden

    def train_preprocess(self):
        return self.train_transform

    def dev_preprocess(self):
        return self.dev_transform

    def forward(self, img, verb_q):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        #print('role size ', role_q.size())

        img_features, conv = self.conv(img)
        batch_size, n_channel, conv_h, conv_w = img_features.size()
        verb_q = verb_q.view(batch_size, -1)
        img = img_features.view(batch_size, n_channel, -1)
        img = img.permute(0, 2, 1)
        w_emb = self.w_emb(verb_q)

        #do this so lstm ignored pads. i can't order by seq length as all roles of image should be together.
        #hence removing, so padded time steps are also included in the prediction
        '''embed = nn.utils.rnn.pack_padded_sequence(embed, question_length,
                                                  batch_first=True)'''
        lstm_out, (h, _) = self.q_emb(w_emb)
        #used to pad back what was removed by packing
        '''lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out,
                                                       batch_first=True)'''
        #q_emb = self.lstm_proj(lstm_out)
        q_emb = h.permute(1, 0, 2).contiguous().view(batch_size, -1)
        q_emb = self.lstm_proj(q_emb)
        #q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)

        #role_label_pred = logits.contiguous().view(batch_size, -1, self.vocab_size)
        return logits

    def forward_eval(self, img_id, img):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        #print('role size ', role_q.size())

        img_features, conv = self.conv(img)
        batch_size, n_channel, conv_h, conv_w = img_features.size()
        verb_qs = self.encoder.get_verb_questions_batch(img_id)
        if self.gpu_mode >= 0:
            verb_qs = verb_qs.to(torch.device('cuda'))

        verb_q_count = verb_qs.size(1)
        verb_qs = verb_qs.view(-1, verb_qs.size(-1))

        img = img_features.view(batch_size, n_channel, -1)
        img = img.permute(0, 2, 1)

        img = img.expand(verb_q_count,img.size(0), img.size(1), img.size(2))
        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size* verb_q_count, -1, self.mlp_hidden)

        w_emb = self.w_emb(verb_qs)

        #do this so lstm ignored pads. i can't order by seq length as all roles of image should be together.
        #hence removing, so padded time steps are also included in the prediction
        '''embed = nn.utils.rnn.pack_padded_sequence(embed, question_length,
                                                  batch_first=True)'''
        lstm_out, (h, _) = self.q_emb(w_emb)
        #used to pad back what was removed by packing
        '''lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out,
                                                       batch_first=True)'''
        #q_emb = self.lstm_proj(lstm_out)
        q_emb = h.permute(1, 0, 2).contiguous().view(batch_size * 5, -1)
        q_emb = self.lstm_proj(q_emb)
        #q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)

        verb_pred = logits.contiguous().view(batch_size, -1, self.n_verbs)
        return verb_pred

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