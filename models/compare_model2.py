"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import os
from functools import partial
from collections import OrderedDict

import pylab as p
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from nystrom_attention import NystromAttention

def to_csv(cur,epoch,slide_id,gap,gmp,adj,str):
    # 导入CSV安装包
    import csv

    # 1. 创建文件对象
    adj_path = '/remote-home/sunxinhuan/PycharmProject/CLAM-master/ADJ_1/{}_dataset/{}/{}'.format(str,cur,epoch)
    if not os.path.exists(adj_path):
        os.makedirs(adj_path)
    f = open(os.path.join(adj_path,'{}.csv'.format(slide_id)), 'w', encoding='utf-8')


    # 2. 基于文件对象构建 csv写入对象
    csv_writer = csv.writer(f)
    gap = gap.cpu().numpy()
    gmp = gmp.cpu().numpy()
    adj = adj.cpu().numpy()
    # 4. 写入csv文件内容
    csv_writer.writerow('gap')
    csv_writer.writerow(gap)
    csv_writer.writerow('gmp')
    csv_writer.writerow(gmp)
    csv_writer.writerow('adj')
    for i in range(adj.shape[0]):
        csv_writer.writerow(adj[i])
    # 5. 关闭文件
    f.close()


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=512, D=256, dropout=False, n_classes=1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        x = A.mul(x)
        return x
############################################adj############################################
class GAP_self_Attention(nn.Module):
    def __init__(self,L = 1024, D = 512,dropout = False,num_classes=1):
        super(GAP_self_Attention, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(num_classes)
        # self.module = [
        #     nn.Linear(L, D),
        #     nn.Tanh()]
        # if dropout:
        #     self.module.append(nn.Dropout(0.25))
        #
        # self.module.append(nn.AdaptiveAvgPool1d(num_classes))
        #
        # self.module = nn.Sequential(*self.module)

    def forward(self, x):
        gap = self.gap(x)
        # gap = F.sigmoid(gap)
        # print('zuixiaozhi:',torch.min(gap))
        # print('zuidazhi:',torch.max(gap))
        return gap


class GMP_self_Attention(nn.Module):
    def __init__(self, L = 1024, D = 512, dropout = False,num_classes=1):
        super(GMP_self_Attention, self).__init__()
        self.gmp = nn.AdaptiveMaxPool1d(num_classes)

        # self.module = [
        #     nn.Linear(L, D),
        #     nn.Sigmoid()]
        # if dropout:
        #     self.module.append(nn.Dropout(0.25))
        #
        # self.module.append(nn.AdaptiveMaxPool1d(num_classes))
        #
        # self.module = nn.Sequential(*self.module)

    def forward(self, x):
        gmp = self.gmp(x)
        # gmp = F.sigmoid(gmp)
        # print('gmpzuixiaozhi:', torch.min(gmp))
        # print('gmpzuidazhi:', torch.max(gmp))
        return gmp


class ADJ(nn.Module):
    def __init__(self,num_classes,size_arg = "small"):
        super(ADJ, self).__init__()
        self.num_classes = num_classes
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        self.fc = nn.Sequential(nn.Linear(size[0], size[1]), nn.ReLU())
        self.gap = GAP_self_Attention(L=size[0],D=size[1],dropout=True,num_classes=1)
        self.gmp = GMP_self_Attention(L=size[0],D=size[1],dropout=True,num_classes=1)

    # def forward(self,slide_id,x,str,cur=0,epoch=-1):
    def forward(self, x):
        # x = kwargs['data'].float()
        device = x.device
        x = self.fc(x)
        if x.shape[1] >= 6000:
            x = x[:,:6000,:]
        gap = torch.squeeze(self.gap(x),dim=0)
        gmp = torch.squeeze(self.gmp(x),dim=0)
        adj = torch.mm(gap,gmp.transpose(0,1))
        # to_csv(cur,epoch,slide_id,gap,gmp,adj,str)
        return adj


############################################Droppath############################################
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# class PatchEmbed(nn.Module):
#     """
#     2D Image to Patch Embedding
#     """
#     def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
#         super().__init__()
#         img_size = (img_size, img_size)
#         patch_size = (patch_size, patch_size)
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
#         self.num_patches = self.grid_size[0] * self.grid_size[1]
#
#         self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         assert H == self.img_size[0] and W == self.img_size[1], \
#             f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
#
#         # flatten: [B, C, H, W] -> [B, C, HW]
#         # transpose: [B, C, HW] -> [B, HW, C]
#         x = self.proj(x).flatten(2).transpose(1, 2)
#         x = self.norm(x)
#         return x

############################################ Attention ############################################
class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)#[3,1,1,n,512]
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)[1,1,n,512]

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GML(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=2.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(GML, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, drop_ratio, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = nn.Dropout(drop_ratio)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha #激活函数有关
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.fc = nn.Linear(out_features, in_features)

    def forward(self, h, adj):
        # print(h.shape)
        Wh = torch.mm(h, self.W)  # h.shape: (节点数N, 输入节点的特征维度in_features), Wh.shape: (N, out_features)
        # print(self.W.shape)
        # print(self.a.shape)
        a_input = self._prepare_attentional_mechanism_input(Wh)  # (N, N, 2 * out_features)
        # print(a_input.shape)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # print('e1:',e1.shape)
        # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2) + adj)
        # print('e:',e.shape)
        zero_vec = -9e15 * torch.ones_like(e)
        # mask注意力系数
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        attention = self.dropout(attention)
        # 注意力系数加权求和
        h_prime = torch.matmul(attention, Wh)
        h_prime = self.fc(h_prime)

        if self.concat:
            return F.elu(h_prime) #elu激活函数
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # 节点数N
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)  # (N*N, out_features)将wh中每个元素重复N次
        Wh_repeated_alternating = Wh.repeat(N, 1)  # (N*N, out_features)将wh整个张量重复N次
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class LML(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=2.,
                 drop_ratio=0.,
                 alpha=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(LML, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GraphAttentionLayer(dim, dim//8, drop_ratio=drop_ratio, alpha=alpha, concat=False)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self,lmlkeys):
        x = lmlkeys[0]
        adj = lmlkeys[1]

        x = x + self.drop_path(self.attn(self.norm1(x), adj))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        lmlkeys[0] = x
        return lmlkeys


class Dynamic_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., type='nofuse'):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if type == 'fuse':
            self.q, self.k, self.v = nn.Linear(dim, dim, bias=qkv_bias), nn.Linear(dim, dim, bias=qkv_bias), nn.Linear(
                dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.first_batch = True

    def forward(self, x, ft=False):
        if ft == True:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x, attn

        B, N, C = x.shape

        # for the multi-modal fusion
        if N > 1:
            if self.first_batch:
                self.q.weight.data, self.q.bias.data = self.qkv.weight.data[:self.dim, :], self.qkv.bias.data[:self.dim]
                self.k.weight.data, self.k.bias.data = self.qkv.weight.data[self.dim:self.dim * 2,
                                                       :], self.qkv.bias.data[self.dim:self.dim * 2]
                self.v.weight.data, self.v.bias.data = self.qkv.weight.data[self.dim * 2:, :], self.qkv.bias.data[
                                                                                               self.dim * 2:]
                self.first_batch = False
            qkv = torch.cat([self.q(x), self.k(x), self.v(x)], dim=-1).reshape(B, N, 3, self.num_heads,
                                                                               C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                            4)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)q/k/v:[1,8,6000,128]
        attn = (q @ k.transpose(-2, -1)) * self.scale #[1,8,6000,6000]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        if N > 1:
            vis_attn = attn.detach().mean(0).mean(0) #[6000,6000]
        elif N <= 1:
            vis_attn = None

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, vis_attn


class HDAF(nn.Module):
    def __init__(self,
                 dim,
                 norm_layer=nn.LayerNorm):
        super(HDAF, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Dynamic_Attention(dim=dim,type='fuse',qkv_bias=True)

    def forward(self, x):

        xf,_ = self.attn(self.norm1(torch.unsqueeze(x, dim=0)))
        x = x + torch.squeeze(xf, dim=0)

        return x

# class GAT(nn.Module):
#     def __init__(self, nfeat, nheads, nclass, dropout, alpha, nhid):
#         """Dense version of GAT."""
#         super(GAT, self).__init__()
#         self.dropout = dropout
#         # 多个图注意力层
#         self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)
#         # 输出层
#         self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
#
#     def forward(self, x, adj):
#         # print(x.shape)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
#         # print(x.shape)
#         # x = F.dropout(x, self.dropout, training=self.training)
#         # x = F.elu(x)
#         # print(x.shape)
#         x = torch.transpose(x,1,0)
#         return x

class MILFusion(nn.Module):
    def __init__(self, embed_dim=512, num_classes=1000,subtyping=True,depth1 = 2, depth2 = 2,
                 num_heads=8, mlp_ratio=2.0, qkv_bias=True, qk_scale=None, representation_size=None,
                 distilled=False, drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(MILFusion, self).__init__()
        self._fc1 = nn.Sequential(nn.Linear(1024, embed_dim), nn.ReLU())
        self.num_classes = num_classes
        self.subtyping = subtyping
        # self.k_sample = k_sample
        # self.instance_loss_fn = instance_loss_fn
        # norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        # act_layer = act_layer or nn.GELU

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth1)]  # stochastic depth decay rule
        self.gml = nn.Sequential(*[
            GML(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i])
            for i in range(depth1)
        ])
        self.lml = nn.Sequential(*[
                    LML(dim=embed_dim, mlp_ratio=mlp_ratio, drop_ratio=drop_ratio, drop_path_ratio=dpr[i])
                    for i in range(depth2)
                ])
        # self.add_module('attention', self.lml)
        self.hdaf = HDAF(dim=embed_dim)

        self.gate = Attn_Net_Gated()

        self.norm = nn.LayerNorm(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())#这里改为nn.ReLU()
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        # self.fcA = nn.Linear(512,32)
        self.classifiers = nn.Linear(512,self.num_classes)

    def forward_features(self, x ,adj):

        xg = self.gml(x) #h[1,N,512]
        # print('transformer shape:',x.shape)
        lmlkeys = {0: torch.squeeze(x, dim=0), 1: adj}
        xl = self.lml(lmlkeys)
        xl = torch.unsqueeze(xl[0], dim=0)
        # print('GAT shape:',A.shape)
        # x1 = torch.squeeze(x1, dim=0)
        # x2 = torch.squeeze(x2, dim=0)
        # x = x1 + x2
        return xg,xl


    def relocate(self):
        device=torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)
        self.lml = self.lml.to(device)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    # instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    # instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, **kwargs):
        x, ct = kwargs['data'].float(), kwargs['CT_data'].float()
        ct = torch.squeeze(ct, dim=0)#[n,1024]
        ct = self._fc1(ct)
        if x.shape[1] >= 6000:
            x = x[:,:6000,:]
        adj = kwargs['adj'].float()
        # x = torch.unsqueeze(x,dim=0)
        x = self._fc1(x)  # [1,n,512]

        xg, xl = self.forward_features(x, adj) # [1,N,512]
        xg = torch.mean(xg, dim=1)
        xl = torch.mean(xl, dim=1)

        # fusion
        # xg_ct = self.hdaf(torch.cat([xg,ct],dim=0))
        # xl_ct = self.hdaf(torch.cat([xl,ct],dim=0))
        # x_ct = self.hdaf(torch.cat([xl_ct,xg_ct],dim=0))
        # x_ct = torch.unsqueeze(torch.mean(x_ct, dim=0), dim=0)
        # xl_ct = torch.unsqueeze(torch.mean(xl_ct, dim=0), dim=0)
        # xg_ct = torch.unsqueeze(torch.mean(xg_ct, dim=0), dim=0)

        # single fusion
        # x_ct = self.hdaf(torch.cat([xl,xg,ct],dim=0))
        # x_ct = torch.unsqueeze(torch.mean(x_ct, dim=0), dim=0)

        # gate-atten
        x_ct = self.gate(torch.cat([xl,xg,ct],dim=0))
        x_ct = torch.unsqueeze(torch.mean(x_ct, dim=0), dim=0)

        # Kr fusion

        # 去除融合层
      #  x_ct = ct+xg+xl

        ## wsi
        # x_ct = self.hdaf(torch.cat([xg,xl],dim=0))
        # x_ct = torch.unsqueeze(torch.mean(x_ct, dim=0), dim=0)

        # CT
        # x_ct = ct

        logits0 = self.classifiers(x_ct)
        logits1 = logits0
        logits2 = logits0
        # logits1 = self.classifiers(xl_ct)
        # logits2 = self.classifiers(xg_ct)
        logits = [logits0, logits1, logits2]
        # print(logits.shape)
        Y_hat = torch.topk(logits0, 1, dim=1)[1]
        # print(Y_hat.shape)
        Y_prob = F.softmax(logits0, dim=1)
        # print(Y_prob.shape)

        return logits, Y_prob, Y_hat



def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True,instance_loss_fn=nn.CrossEntropyLoss()):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    # model = VisionTransformer(img_size=224,
    #                           patch_size=16,
    #                           embed_dim=512,
    #                           depth=1,
    #                           num_heads=2,
    #                           representation_size=768 if has_logits else None,
    #                           num_classes=num_classes,
    #                           subtyping=True,
    #                           k_sample=15,
    #                           instance_loss_fn = instance_loss_fn,
    #                           nfeat=512,
    #                           nheads=1,
    #                           dropout=0.6,
    #                           alpha=0.4,
    #                           nhid=3
    #                           )
    model = MILFusion(embed_dim=512,
                      depth1=2,
                      depth2=2,
                      num_heads=1,
                      representation_size=768 if has_logits else None,
                      num_classes=num_classes,
                      subtyping=False,
                      )

    return model

if __name__ == "__main__":
    data = torch.randn((3, 512)).cuda()


    # data = torch.randn((1, 6000, 1024)).cuda()
    # ct_data = torch.randn((1, 1, 1024)).cuda()
    # adj = torch.randn((6000, 6000)).cuda()


    # model = MILFusion(embed_dim=512,
    #                   depth1=2,
    #                   depth2=2,
    #                   num_heads=1,
    #                   representation_size=768,
    #                   num_classes=2,
    #                   subtyping=False,
    #                   ).cuda()

    model = Attn_Net_Gated().cuda()
    results_dict = model(x=data)
    print(results_dict)


