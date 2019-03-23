'''
original https://raw.githubusercontent.com/hengyuan-hu/bottom-up-attention-vqa/master/fc.py
'''

from __future__ import print_function
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from torch.nn.init import kaiming_uniform_, xavier_uniform_, normal


class FCNet1(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def linear(self, in_dim, out_dim, bias=True):
        lin = nn.Linear(in_dim, out_dim, bias=bias)
        xavier_uniform_(lin.weight)
        if bias:
            lin.bias.data.zero_()

        return lin

    def __init__(self, dims):
        super(FCNet1, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            layers.append(weight_norm(self.linear(in_dim, out_dim), dim=None))
            layers.append(nn.ReLU())
        layers.append(weight_norm(self.linear(dims[-2], dims[-1]), dim=None))
        layers.append(nn.ReLU())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def linear(self, in_dim, out_dim, bias=True):
        lin = nn.Linear(in_dim, out_dim, bias=bias)
        xavier_uniform_(lin.weight)
        if bias:
            lin.bias.data.zero_()

        return lin

    def __init__(self, dims):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            layers.append(self.linear(in_dim, out_dim))
            layers.append(nn.Tanh())
        layers.append(self.linear(dims[-2], dims[-1]))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


if __name__ == '__main__':
    fc1 = FCNet([10, 20, 10])
    print(fc1)

    print('============')
    fc2 = FCNet([10, 20])
    print(fc2)