import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing

from utils.utils import initialize_weights
import numpy as np



# class GAP_self_Attention(nn.Module):
#     def __init__(self,L = 512, D = 256,dropout = False,num_classes=1):
#         super(GAP_self_Attention, self).__init__()
#         self.module = [
#             nn.Linear(L, D),
#             nn.Sigmoid()]
#         if dropout:
#             self.module.append(nn.Dropout(0.25))
#
#         self.module.append(nn.AdaptiveAvgPool1d(num_classes))
#
#         self.module = nn.Sequential(*self.module)
#
#     def forward(self, x):
#         gap = self.module(x)
#         return gap
#
#
# class GMP_self_Attention(nn.Module):
#     def __init__(self, L = 512, D = 256, dropout = False,num_classes=1):
#         super(GMP_self_Attention, self).__init__()
#         self.module = [
#             nn.Linear(L, D),
#             nn.Sigmoid()]
#         if dropout:
#             self.module.append(nn.Dropout(0.25))
#
#         self.module.append(nn.AdaptiveMaxPool1d(num_classes))
#
#         self.module = nn.Sequential(*self.module)
#
#     def forward(self, x):
#         gmp = self.module(x)
#         return gmp

#
# class GMP_self_Attention(nn.Module):
#     def __init__(self, L = 1024, D = 512, dropout = False,num_classes=1):
#         super(GMP_self_Attention, self).__init__()
#         self.attention_a = [
#             nn.Linear(L, D),
#             nn.Tanh()]
#
#         self.attention_b = [nn.Linear(L, D),
#                             nn.Sigmoid()]
#         if dropout:
#             self.attention_a.append(nn.Dropout(0.25))
#             self.attention_b.append(nn.Dropout(0.25))
#
#         self.attention_a = nn.Sequential(*self.attention_a)
#         self.attention_b = nn.Sequential(*self.attention_b)
#
#         self.attention_c = nn.AdaptiveMaxPool1d(1)
#
#     def forward(self, x):
#         a = self.attention_a(x)
#         b = self.attention_b(x)
#         A = a.mul(b)
#         A = torch.unsqueeze(A, dim=0)
#         gmp = self.attention_c(A)  # N x n_classes
#         gmp = torch.squeeze(gmp, dim=0)
#         return gmp

class GAP_self_Attention(nn.Module):
    def __init__(self,L = 1024, D = 512,dropout = False,num_classes=1):
        super(GAP_self_Attention, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(num_classes)


    def forward(self, x):
        gap = self.gap(x)
        return gap


class GMP_self_Attention(nn.Module):
    def __init__(self, L = 1024, D = 512, dropout = False,num_classes=1):
        super(GMP_self_Attention, self).__init__()
        self.gmp = nn.AdaptiveMaxPool1d(num_classes)

    def forward(self, x):
        gmp = self.gmp(x)
        return gmp


class ADJ(nn.Module):
    def __init__(self,num_classes,size_arg = "small"):
        super(ADJ, self).__init__()
        self.num_classes = num_classes
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        self.fc = nn.Sequential(nn.Linear(size[0], size[1]), nn.Sigmoid())
        self.gap = GAP_self_Attention(L=size[1],D=size[0],dropout=True,num_classes=1)
        self.gmp = GMP_self_Attention(L=size[1],D=size[0],dropout=True,num_classes=1)

    def forward(self,x):
        # x = kwargs['data'].float()
        device = x.device
        # print(x.shape)
        # print(x)
        x = self.fc(x)
        gap = self.gap(x)
        gmp = self.gmp(x)
        adj = torch.mm(gap,gmp.transpose(0,1))

        return adj


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (节点数N, 输入节点的特征维度in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)  # (N, N, 2 * out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2) + adj)
        zero_vec = -9e15 * torch.ones_like(e)
        # mask注意力系数
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        # 注意力系数加权求和
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # 节点数N
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)  # (N*N, out_features)
        Wh_repeated_alternating = Wh.repeat(N, 1)  # (N*N, out_features)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nheads, nclass, dropout, alpha, nhid):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.fc1 = nn.Sequential(nn.Linear(1024, nfeat), nn.ReLU())
        # 多个图注意力层
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        # 输出层
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = self.fc1(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        x, m_indices = torch.sort(x, 0,descending=True)  # sort class scores along the instance dimension, m_indices in shape N x C
        top1_x = x[0]
        top1_x = torch.unsqueeze(top1_x,dim=0)
        logits = F.log_softmax(top1_x, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        return logits,Y_prob,Y_hat,x,adj


def GAT_model(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = GAT(nfeat=512,
                nheads=8,
                nclass=num_classes,
                dropout=0.6,
                alpha=0.4,
                nhid=4)
    return model
