'''
original https://raw.githubusercontent.com/hengyuan-hu/bottom-up-attention-vqa/master/attention.py
'''

import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from fc import FCNet


class NonLinearLayer(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(NonLinearLayer, self).__init__()
        self.for_y = nn.Linear(input_dim, out_dim)
        self.for_g = nn.Linear(input_dim, out_dim)

    def forward(self, x):
        y = torch.tanh(self.for_y(x))
        g = torch.sigmoid(self.for_g(x))
        return y * g


class Attention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid):
        super(Attention, self).__init__()
        self.nonlinear = FCNet([v_dim + q_dim, num_hid])
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.nonlinear(vq)
        logits = self.linear(joint_repr)
        return logits

class Attentionjust(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid):
        super(Attention, self).__init__()
        self.nonlinear = nn.Sequential(
            nn.Linear(v_dim + q_dim, num_hid),
            nn.ReLU(inplace=True)
        )
        self.linear = nn.Linear(num_hid, 1)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.nonlinear(vq)
        logits = self.linear(joint_repr)
        return logits

class PropAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid):
        super(PropAttention, self).__init__()
        self.nonlinear = NonLinearLayer(v_dim + q_dim, num_hid)
        self.linear = nn.Linear(num_hid, 1)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.nonlinear(vq)
        logits = self.linear(joint_repr)
        return logits

class BigAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid):
        super(BigAttention, self).__init__()
        self.nonlinear = FCNet([v_dim + q_dim, num_hid])
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        num_objs = v.size(2)
        q = q.unsqueeze(2).repeat(1, 1,num_objs, 1)
        vq = torch.cat((v, q), 3)
        joint_repr = self.nonlinear(vq)
        logits = self.linear(joint_repr)
        return logits

class RoleWeightAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid):
        super(RoleWeightAttention, self).__init__()
        self.nonlinear = FCNet([v_dim + q_dim, num_hid])
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        batch_size, num_roles, num_roles, featsize = v.size()
        v = v.view(-1, num_roles, featsize)
        q = q.unsqueeze(1).repeat(1, num_roles, 1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.nonlinear(vq)
        logits = self.linear(joint_repr)
        logits = logits.view(batch_size, num_roles,num_roles, 1)
        return logits


class NewAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(NewAttention, self).__init__()

        self.v_proj = FCNet([v_dim, num_hid])
        self.q_proj = FCNet([q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v) # [batch, k, qdim]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits

class NewAttentionjust(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(NewAttentionjust, self).__init__()

        self.v_proj = nn.Sequential(
            nn.Linear(v_dim , num_hid),
            nn.ReLU(inplace=True)
        )
        self.q_proj = nn.Sequential(
            nn.Linear(q_dim, num_hid),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(num_hid, 1)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v) # [batch, k, qdim]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits

class NewAttentionmultihead(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(NewAttentionmultihead, self).__init__()

        self.v_proj = nn.Sequential(
            nn.Linear(v_dim , num_hid),
        )
        self.q_proj = nn.Sequential(
            nn.Linear(q_dim, num_hid),
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(num_hid, 1)
        self.h = 1
        self.d_k = num_hid // self.h

    def forward(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v) # [batch, k, qdim]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)

        v_proj= v_proj.view(batch, -1, self.h, self.d_k).transpose(1, 2)
        q_proj = q_proj.view(batch, -1, self.h, self.d_k).transpose(1, 2)

        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        p_attn = nn.functional.softmax(logits, dim = -1)

        x = p_attn * v_proj

        x = x.transpose(1, 2).contiguous() \
            .view(batch, -1, self.h * self.d_k)

        return x