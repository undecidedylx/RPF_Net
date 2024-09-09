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


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=1):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # shared MLP
        # self.mlp = nn.Sequential(
        #     # Conv2d比Linear方便操作
        #     # nn.Linear(channel, channel // reduction, bias=False)
        #     nn.Conv2d(channel, channel // reduction, 1, bias=False),
        #     # inplace=True直接替换，节省内存
        #     nn.ReLU(inplace=True),
        #     # nn.Linear(channel // reduction, channel,bias=False)
        #     nn.Conv2d(channel // reduction, channel, 1, bias=False)
        # )

        self.gat = GraphAttentionLayer(1024, 1024, dropout=0.25, alpha=0.4, concat=True)
        # self.atn = Attention()

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel, bias=False)
        # padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.max_pool(x).squeeze(0)
        avg_out = self.avg_pool(x).squeeze(0)
        adj = torch.mm(max_out,avg_out.transpose(0,1))
        x_gat = x.squeeze(0)
        x_gat = self.gat(x_gat,adj)
        x_gat = x_gat.unsqueeze(0)

        # max_out = self.mlp(self.max_pool(x).unsqueeze(3))
        # avg_out = self.mlp(self.avg_pool(x).unsqueeze(3))
        # channel_out = self.sigmoid(max_out + avg_out).squeeze(3)
        # x = channel_out * x

        # max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.atn(avg_out))
        # spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1).unsqueeze(3))).squeeze(3)
        x = spatial_out * x

        x = x + x_gat
        return x

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


class Attn_Net(nn.Module):

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes

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
        x = x.unsqueeze(0)
        gap = self.gap(x)
        gap = gap.squeeze(0)
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
        x = x.unsqueeze(0)
        gmp = self.gmp(x)
        gmp = gmp.squeeze(0)
        # gmp = F.sigmoid(gmp)
        # print('gmpzuixiaozhi:', torch.min(gmp))
        # print('gmpzuidazhi:', torch.max(gmp))
        return gmp

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

class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class Attention(nn.Module):
    def __init__(self,
                 dim=1024,   # 输入token的dim
                 num_heads=2,
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
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

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
class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # self.cbam = CBAM(dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
        # self.mlp = CBAM(dim)
        # self.mlp = ConvolutionalGLU(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

## GAT
class ADJ(nn.Module):
    def __init__(self,num_classes,size_arg = "small"):
        super(ADJ, self).__init__()
        self.num_classes = num_classes
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        # self.fc = nn.Sequential(nn.Linear(size[0], size[1]), nn.ReLU())
        self.gap = GAP_self_Attention(L=size[0],D=size[1],dropout=True,num_classes=1)
        self.gmp = GMP_self_Attention(L=size[0],D=size[1],dropout=True,num_classes=1)

    # def forward(self,slide_id,x,str,cur=0,epoch=-1):
    def forward(self, x):
        # x = kwargs['data'].float()
        device = x.device
        # x = self.fc(x)
        x = x.squeeze(0)
        gap = self.gap(x)
        gmp = self.gmp(x)
        # gmp = self.gap(x.transpose(0,1))
        # gmp_1 = gmp.transpose(0,1)
        adj = torch.mm(gap,gmp.transpose(0,1))#gap一般在0.0。gmp一般在0.
        # to_csv(cur,epoch,slide_id,gap,gmp,adj,str)
        return adj
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.norm = nn.LayerNorm(out_features)

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)


    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (节点数N, 输入节点的特征维度in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)  # (N, N, 2 * out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # max = e.max()
        # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2) + adj)
        zero_vec = -9e15 * torch.ones_like(e)
        # mask注意力系数
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        # 注意力系数加权求和
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.sigmoid(h_prime)
        else:
            return h_prime #N*out_feature

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
        # 多个图注意力层
        self.attentions = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
        self.CML = nn.Sequential(nn.LayerNorm(nhid), nn.Linear(nhid, nhid), nn.LeakyReLU(), nn.Linear(nhid, nhid))
        # self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)
        # 输出层
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        # print(x.shape)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        n = self.attentions(x, adj)
        # n = x + n
        # c = self.CML(n)
        # x = n + c
        # print(x.shape)
        # x = F.elu(x)
        # print(x.shape)
        return n

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,subtyping=True,
                 k_sample=8,instance_loss_fn=nn.CrossEntropyLoss(),
                 embed_dim=512, depth_Block=4, depth_GAT=4, num_heads=8, mlp_ratio=2.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None,nfeat=None, nheads=None, dropout=None, alpha=None, nhid=None):
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
        super(VisionTransformer, self).__init__()
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.LeakyReLU())
        self.CML = nn.Sequential(nn.LayerNorm(512),nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 16))
        self.num_classes = num_classes
        self.subtyping = subtyping
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.attn = Attn_Net(L=1024,D=512)
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth_Block)]

        # self.cbam = CBAMLayer(1024)

        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth_Block)])

        self.GAT = GAT(nfeat=nfeat, nheads=nheads, nclass=self.num_classes, dropout=dropout, alpha=alpha, nhid=nhid)

        # self.norm =norm_layer(embed_dim)
        self.norm1 = nn.LayerNorm(1024)

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
        # self._fc2 = nn.Linear(embed_dim, self.num_classes)if num_classes > 0 else nn.Identity()
        self.fcA = nn.Linear(1024,512)
        self.fcB = nn.Linear(512, 256)
        self.ADJ = ADJ(num_classes=self.num_classes)
        self.classifiers = nn.Linear(256,self.num_classes)

        # instance_classifiers = [nn.Linear(512, 2) for i in range(self.num_classes)]
        # self.instance_classifiers = nn.ModuleList(instance_classifiers)

    def forward_features(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # A1 = CBAMLayer(x.shape[1]).to(device)(x)
        A1 = self.blocks(x).to(device)
        # A1 = A1.squeeze(0)
        # adj = self.ADJ(x)
        # A = self.GAT(torch.squeeze(x,dim=0),adj) #图网络
        # A = self.GAT(torch.squeeze(A,dim=0),adj) #图网络
        # print('GAT shape:',A.shape)
        # x2 = self.cbam(x)
        # x1 = torch.squeeze(x1, dim=0)
        # x2 = torch.squeeze(x2, dim=0)
        # x = x1 + x2
        # return A, A1
        return A1

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifiers = self.classifiers.to(device)
        # self.instance_classifiers = self.instance_classifiers.to(device)
        self.GAT = self.GAT.to(device)
        self.ADJ = self.ADJ.to(device)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    def forward(self, **kwargs):
        wsi, ct = kwargs['data'].float(), kwargs['CT_data'].float()#[1,n,1024]
        wsi, ct = wsi.squeeze(0), ct.squeeze(0)
        # print(x.shape)
        ##开始处理病理图像
        # if wsi.shape[0] > 2048:
        #     wsi = wsi[:2048, :]
        ## 设计每次提取一部分切片进行图网络，有些切片数量过多，并且结果不好
        # start = 0
        # ls = []
        # for i in range(8):
        #     step = int(wsi.shape[0]/8)
        #     x = wsi[start:start+step,:]
        #     start += step
        #     x1 = torch.unsqueeze(x, dim=0)
        #     # print(x1.shape)
        #     x1 = self._fc1(x1)  # [1,n,512]，全连接层降维，特征值中出现好多零,因为relu
        #     a = self.forward_features(x1)
        #     ls.append(a)
        # A = torch.cat(ls,dim=0)

        ## 根据全连接层筛选切片，没效果
        # a, x = self.attn(x)
        # a = torch.transpose(a, 1, 0)
        # a = F.softmax(a, dim=1)
        # if x.shape[0] > 2000:
        #     top_p_ids = torch.topk(a, 2000)[1][-1]
        #     top_p = torch.index_select(x, dim=0, index=top_p_ids)
            # top_n_ids = torch.topk(-a, 2500, dim=1)[1][-1]
            # top_n = torch.index_select(x, dim=0, index=top_n_ids)
            # x = torch.cat([top_p, top_n], dim=0)
            # top_p_ids = torch.range(0,2000)
            # x = torch.index_select(x,dim=0,index=top_p_ids)
            # x = top_p

        ## 全连接
        # wsi = torch.mean(wsi, dim=0)
        # wsi = torch.unsqueeze(wsi, dim=0)
        # AA = wsi * ct
        # A_raw = AA
        # results_dict_instance = {}
        # AA = self.fcA(AA)
        # AA = self.fcB(AA)
        # logits = self.classifiers(AA)
        # Y_hat = torch.topk(logits, 1, dim=1)[1]
        # Y_prob = F.softmax(logits, dim=1)

        ## vit
        wsi = torch.mean(wsi, dim=0)
        wsi = torch.unsqueeze(wsi, dim=0)
        wsi = self._fc1(wsi) #维度对齐
        wsi = wsi.unsqueeze(0)
        wsi_t = self.forward_features(wsi)
        A1 = wsi_t.squeeze(0)
        AA = A1
        A_raw = AA
        results_dict_instance = {}
        AA = self.fcA(AA)
        logits = self.classifiers(AA)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)


        # x = torch.unsqueeze(x,dim=0)
        # x = self._fc1(x)  # [1,n,512]，全连接层降维，特征值中出现好多零,因为relu
        # A, x = self.forward_features(x) # [1,N,512]，A是图网络输出，此时x是transformer输出
        # x = torch.squeeze(x, dim=0)
        # wsi = torch.mean(wsi, dim=0)
        # wsi = torch.unsqueeze(wsi, dim=0)
        # wsi = self._fc1(wsi) #维度对齐
        # wsi = wsi.unsqueeze(0)
        # wsi_t = self.forward_features(wsi)
        # wsi_t = self.forward_features(wsi)
        # wsi_t = self.forward_features(wsi_t)
        # wsi_t = self.forward_features(wsi_t)
        # wsi_t = self.forward_features(wsi_t)
        # A1 = wsi_t.squeeze(0)
        # A = F.softmax(wsi_g, dim=1)
        # A1 = F.softmax(wsi_t, dim=1)

        # A1 = torch.mean(wsi, dim=0)
        # A = torch.mean(A, dim=0)
        # A1 = torch.unsqueeze(A1, dim=0)  # 16*1
        # A = torch.unsqueeze(A, dim=0)  # 1*16
        # A1 = self.CML(A1)
        # A = self.CML(A)
        # A1 = self.fcA(A1)
        # A = self.fcA(A)
        # A = self.fcB(wsi)
        # A = torch.transpose(A,1,0)

        # y = self.norm1(ct)
        # y = self._fc1(y)  # 1*512
        # y = self.fcA(y)  # 1*16
        # y = F.softmax(y, dim=1)

        # Ay = torch.cat((A, torch.transpose(y,1,0)),dim=0)
        # A1y = torch.cat((y,A1),dim=1)#32*1

        # AA = torch.mm(A,A1)
        # AA = wsi
        # AA = torch.mean(wsi, dim=0)

        # 单独病理一个模态
        # AA = torch.mm(A,A1)
        # print(AA.shape)

        # 单独影像一个模态
        # AA = y

        # A_raw = AA
        # if attention_only:
        #     return A_raw

        # results_dict_instance = {}

        # AA = AA.reshape(1,-1) #将张量改变为一个行向量
        # print(AA.shape)
        # AA = self.fcA(AA)
        # # AA = self.fcB(AA)
        # logits = self.classifiers(AA)
        # # print(logits.shape)
        # Y_hat = torch.topk(logits, 1, dim=1)[1]
        # # print(Y_hat.shape)
        # Y_prob = F.softmax(logits, dim=1)
        # print(Y_prob.shape)

        return logits, Y_prob, Y_hat, A_raw, results_dict_instance



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

def vit_base_patch16_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=512,
                              depth=2,
                              num_heads=8,
                              representation_size=None,
                              num_classes=num_classes)
    return model

def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True,instance_loss_fn=nn.CrossEntropyLoss()):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=512,
                              depth_Block=2,
                              depth_GAT=2,
                              num_heads=4,
                              representation_size = 768 if has_logits else None,
                              num_classes=num_classes,
                              subtyping=False,
                              k_sample=2,
                              instance_loss_fn=instance_loss_fn,
                              nfeat=512,
                              nheads=1,
                              dropout=0.25,  # 这里待会改为0.25
                              # drop_ratio=0.25,
                              # attn_drop_ratio=0.25,
                              # drop_path_ratio=0.1,
                              alpha=0.4,
                              nhid=512
                              )
    return model


def vit_base_patch32_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1hCv0U8pQomwAtHBYc4hmZg  密码: s5hl
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model

def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model

def vit_large_patch16_224(num_classes: int = 1000):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1cxBgZJJ6qUWPSBNcE4TdRQ  密码: qqt8
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=None,
                              num_classes=num_classes)
    return model

def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model

def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model

def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model