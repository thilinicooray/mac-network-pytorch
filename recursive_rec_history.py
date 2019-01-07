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
                 vocab_size,
                 n_verbs,
                 embed_hidden=300,
                 mlp_hidden=512):
        super(TopDown, self).__init__()

        self.vocab_size = vocab_size
        self.verb_size = n_verbs
        self.q_emb = nn.LSTM(embed_hidden, mlp_hidden,
                             batch_first=True, bidirectional=True)
        self.q_prep = FCNet([mlp_hidden, mlp_hidden])
        self.lstm_proj = nn.Linear(mlp_hidden * 2, mlp_hidden)
        self.verb_transform = nn.Linear(embed_hidden, mlp_hidden)
        self.v_att = Attention(mlp_hidden, mlp_hidden, mlp_hidden)
        self.q_net = FCNet([mlp_hidden, mlp_hidden])
        self.v_net = FCNet([mlp_hidden, mlp_hidden])
        self.role_classifier = SimpleClassifier(
            mlp_hidden, 2 * mlp_hidden, self.vocab_size, 0.5)

        self.verb_classifier = SimpleClassifier(
            mlp_hidden, 2 * mlp_hidden, self.verb_size, 0.5)

    def forward(self, img, q, isVerb=False):
        batch_size = img.size(0)
        w_emb = q
        lstm_out, (h, _) = self.q_emb(w_emb)
        q_emb = h.permute(1, 0, 2).contiguous().view(batch_size, -1)
        q_emb = self.lstm_proj(q_emb)

        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        if isVerb:
            logits = self.verb_classifier(joint_repr)
        else:
            logits = self.role_classifier(joint_repr)

        return logits

class RecurrentModel(nn.Module):
    def __init__(self, encoder, role_lookup, ans_lookup, embeddings, vocab_size, n_verbs, gpu_mode
                 ):
        super(RecurrentModel, self).__init__()

        self.encoder = encoder
        self.role_lookup = role_lookup
        self.ans_lookup = ans_lookup
        self.embeddings = embeddings
        self.vocab_size = vocab_size
        self.n_verbs = n_verbs
        self.gpu_mode = gpu_mode
        self.qa_model = TopDown(self.vocab_size, self.n_verbs)

    def forward(self, img, verbq, verb, ans):
        #print('verb q size :', verbq.size())
        # comes original nx7 questions (verb, r1,r2 ....)
        max_steps = 6
        context = None
        ans_dist = None
        verb_ans = self.qa_model(img, verbq, isVerb=True)

        #todo : labels have 3, we only need 1

        if self.training:
            role_qs = self.encoder.get_role_questions_batch(verb)
            roles = self.encoder.get_role_ids_batch(verb)

            if self.gpu_mode >= 0:
                role_qs = role_qs.to(torch.device('cuda'))
                roles = roles.to(torch.device('cuda'))

            for i in range(max_steps):
                if context is None :
                    ans_dist = self.qa_model(img, self.embeddings(role_qs[:,i]))
                    context = torch.cat((self.role_lookup(roles[:,i]).unsqueeze(1),
                                         self.ans_lookup(ans[:,0,i]).unsqueeze(1)), 1)
                else :
                    #print('sizes :', context.size(), self.embeddings(role_qs[:,i]).size())
                    q_expand = torch.cat((context, self.embeddings(role_qs[:,i])),1)
                    ans_dist = torch.cat((ans_dist, self.qa_model(img, q_expand)), 1)
                    context = torch.cat((context, self.role_lookup(roles[:,i]).unsqueeze(1),
                                         self.ans_lookup(ans[:,0,i]).unsqueeze(1)), 1)

                    #print('context size :', context.size(), ans_dist.size())

        else:
            '''
            get all from verb's arg max-, roles and role q,
            ans are also arg max
            '''
            sorted_idx = torch.sort(verb_ans, 1, True)[1]
            verb_ids = sorted_idx[:,0]
            role_qs = self.encoder.get_role_questions_batch(verb_ids)
            roles = self.encoder.get_role_ids_batch(verb_ids)

            if self.gpu_mode >= 0:
                role_qs = role_qs.to(torch.device('cuda'))
                roles = roles.to(torch.device('cuda'))

            for i in range(max_steps):
                if context is None :
                    ans_dist = self.qa_model(img, self.embeddings(role_qs[:,i]))
                    sorted_ans = torch.sort(ans_dist, 1, True)[1]
                    ans = sorted_ans[:,0]
                    context = torch.cat((self.role_lookup(roles[:,i]).unsqueeze(1), self.ans_lookup(ans).unsqueeze(1)), 1)
                else :
                    q_expand = torch.cat((context, self.embeddings(role_qs[:,i])),1)
                    curr_ans = self.qa_model(img, q_expand)
                    #print('size ans :', curr_ans.size())
                    sorted_ans = torch.sort(curr_ans, 1, True)[1]
                    ans = sorted_ans[:,0]
                    #print('ans :', ans)
                    ans_dist = torch.cat((ans_dist, curr_ans), 1)
                    context = torch.cat((context, self.role_lookup(roles[:,i]).unsqueeze(1), self.ans_lookup(ans).unsqueeze(1)), 1)

        return verb_ans, context, ans_dist

    #todo : top n

    '''def forward_eval5(self, img, verbq, max_steps, topn=5):

        context = None
        ans_dist = None
        beam_role_idx = None
        top1role_label_pred = None
        verb_ans = self.qa_model(img, verbq, isVerb=True)

        sorted_idx = torch.sort(verb_ans, 1, True)[1]
        verb_ids = sorted_idx[:,:topn]

        for k in range(topn):
            role_qs = self.encoder.get_role_qs(verb_ids)
            roles = self.encoder.get_roles(verb_ids)

            for i in range(max_steps):
                if context is None :
                    ans_dist = self.qa_model(img, role_qs[:,i])
                    sorted_ans = torch.sort(ans_dist, 1, True)[1]
                    ans = sorted_ans[:,0]
                    context = torch.cat((self.embeddings.role_lookup(roles[:,i]), self.embeddings.ans_lookup(ans)), 1)
                else :
                    q_expand = torch.cat((context, role_qs[:,i]),1)
                    ans_dist = torch.cat((ans_dist, self.qa_model(img, q_expand)), 1)
                    sorted_ans = torch.sort(ans_dist, 1, True)[1]
                    ans = sorted_ans[:,0]
                    context = torch.cat((context, self.embeddings.role_lookup(roles[:,i]), self.embeddings.ans_lookup(ans)), 1)

            if k == 0:
                top1role_label_pred = ans_dist
                idx = torch.max(ans_dist,-1)[1]
                #print(idx[1])
                beam_role_idx = idx
            else:
                idx = torch.max(ans_dist,-1)[1]
                beam_role_idx = torch.cat((beam_role_idx.clone(), idx), 1)


        return verb_ids, beam_role_idx'''

class RecursiveModel(nn.Module):
    def __init__(self, encoder, role_lookup, ans_lookup, embeddings, vocab_size, n_verbs, gpu_mode,
                 n_iter,
                 mlp_hidden=512
                 ):
        super(RecursiveModel, self).__init__()

        self.encoder = encoder
        self.role_lookup = role_lookup
        self.ans_lookup = ans_lookup
        self.embeddings = embeddings
        self.n_iter = n_iter
        self.vocab_size = vocab_size
        self.n_verbs = n_verbs
        self.gpu_mode = gpu_mode
        self.recurrent_model = RecurrentModel(self.encoder, self.role_lookup, self.ans_lookup, self.embeddings,
                                              self.vocab_size,
                                              self.n_verbs,
                                              self.gpu_mode)
        #todo: this is wrong. adjust somewhere else before classifier
        self.verb_concat = nn.Linear(self.n_verbs, self.n_verbs)
        self.noun_concat = nn.Linear(self.vocab_size, self.vocab_size)

    def forward(self, img, verbq, verb, ans):
        verb_preds = []
        noun_preds = []
        if self.training:
            verb_ans, context, ans_dist = self.recurrent_model(img, self.embeddings(verbq), verb, ans)
            ans_dist = ans_dist.contiguous().view(ans_dist.size(0), -1, self.vocab_size)
            verb_preds.append(verb_ans)
            noun_preds.append(ans_dist)

            for j in range(self.n_iter - 1):
                verbq_updated = torch.cat((context, self.embeddings(verbq)),1)
                verb_ans, context, ans_dist = self.recurrent_model(img, verbq_updated, verb, ans)
                ans_dist = ans_dist.contiguous().view(ans_dist.size(0), -1, self.vocab_size)

                next_verb = self.verb_concat(verb_ans * verb_preds[-1])
                next_noun = self.noun_concat(ans_dist * noun_preds[-1])

                verb_preds.append(next_verb)
                noun_preds.append(next_noun)


        else :
            #todo : top 5
            verb_ans, context, ans_dist = self.recurrent_model(img, self.embeddings(verbq), verb, ans)
            ans_dist = ans_dist.contiguous().view(ans_dist.size(0), -1, self.vocab_size)
            verb_preds.append(verb_ans)
            noun_preds.append(ans_dist)

            for j in range(self.n_iter - 1):
                verbq_updated = torch.cat((context, self.embeddings(verbq)),1)
                verb_ans, context, ans_dist = self.recurrent_model(img, verbq_updated, verb, ans)
                ans_dist = ans_dist.contiguous().view(ans_dist.size(0), -1, self.vocab_size)
                next_verb = self.verb_concat(verb_ans * verb_preds[-1])
                next_noun = self.noun_concat(ans_dist * noun_preds[-1])

                verb_preds.append(next_verb)
                noun_preds.append(next_noun)

            '''verbq_updated = torch.cat((context, verbq),1)
            verb_ans, ans_dist = self.recurrent_model.forward_eval5(img, verbq_updated, role_qs, roles, ans)'''

        return verb_preds[-1], noun_preds[-1]

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
        self.role_lookup = nn.Embedding(self.n_roles +1, embed_hidden, padding_idx=self.n_roles)
        self.ans_lookup = nn.Embedding(self.vocab_size + 1, embed_hidden, padding_idx=self.vocab_size)
        self.w_emb = nn.Embedding(self.n_role_q_vocab + 1, embed_hidden, padding_idx=self.n_role_q_vocab)

        self.vsrl_model = RecursiveModel(self.encoder, self.role_lookup, self.ans_lookup, self.w_emb, self.vocab_size,
                                         self.n_verbs,
                                         self.gpu_mode,
                                         n_iter=2)


        self.conv_hidden = self.conv.base_size()
        self.mlp_hidden = mlp_hidden
        self.embed_hidden = embed_hidden

    def train_preprocess(self):
        return self.train_transform

    def dev_preprocess(self):
        return self.dev_transform

    def forward(self, img, verbq, verb, ans):

        img_features = self.conv(img)
        batch_size, n_channel, conv_h, conv_w = img_features.size()
        img = img_features.view(batch_size, n_channel, -1)
        img = img.permute(0, 2, 1)

        verb_pred, role_pred = self.vsrl_model(img, verbq, verb, ans)
        role_pred = role_pred.contiguous().view(batch_size, -1, self.vocab_size)

        #print('ans sizes :', verb_pred.size(), role_pred.size())

        return verb_pred, role_pred

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


