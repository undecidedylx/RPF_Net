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
        x = x.squeeze(0)
        gap = self.gap(x)
        gmp = self.gmp(x)
        # gmp_1 = gmp.transpose(0,1)
        adj = torch.mm(gap,gmp.transpose(0,1))
        # to_csv(cur,epoch,slide_id,gap,gmp,adj,str)
        return adj
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
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # self.cbam = CBAM(dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
        # self.mlp = CBAM(dim)

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
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
        # print(self.W.shape)
        # print(self.a.shape)
        a_input = self._prepare_attentional_mechanism_input(Wh)  # (N, N, 2 * out_features)
        # print(a_input.shape)
        e= self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # print('e1:',e1.shape)
        # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2) + adj)
        # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # print('e:',e.shape)
        zero_vec = -9e15 * torch.ones_like(e)
        # mask注意力系数
        # attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(e, dim=1)
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
        # 多个图注意力层
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        # 输出层
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        # print(x.shape)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # print(x.shape)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(x)
        # print(x.shape)
        x = torch.transpose(x,1,0)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,subtyping=True,
                 k_sample=8,instance_loss_fn=nn.CrossEntropyLoss(),
                 embed_dim=512, depth=4, num_heads=8, mlp_ratio=4.0, qkv_bias=True,
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
        self._fc1 = nn.Sequential(nn.Linear(1024, embed_dim), nn.ReLU())
        self.num_classes = num_classes
        self.subtyping = subtyping
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])

        # self.norm =norm_layer(embed_dim)
        self.norm1 = nn.LayerNorm(512)


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
        self.fcA = nn.Linear(512,16)
        self.GAT = GAT(nfeat=nfeat,nheads=nheads,nclass=self.num_classes,dropout=dropout,alpha=alpha, nhid=nhid)
        # self.fcA = nn.AdaptiveAvgPool1d(1)
        # self.fcA = nn.AdaptiveMaxPool1d(1)
        self.classifiers = nn.Linear(256,self.num_classes)

        # bag_classifiers = [nn.Linear(512, 1) for i in
        #                    range(self.num_classes)]  # use an indepdent linear layer to predict each class
        # self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(512, 2) for i in range(self.num_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)


    def forward_features(self, x1,adj):

        #---->class_token
        # B = x.shape[0]
        # cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        # x = torch.cat((cls_tokens, x), dim=1)  # h[B,N+1,512]

        # x = self.cbam(x)
        x = self.blocks(x1) #h[1,N+1,512]
        # print('transformer shape:',x.shape)
        A = self.GAT(torch.squeeze(x1,dim=0),adj)
        # print('GAT shape:',A.shape)
        # x2 = self.cbam(x)
        # x1 = torch.squeeze(x1, dim=0)
        # x2 = torch.squeeze(x2, dim=0)
        # x = x1 + x2
        return A,x


    def relocate(self):
        device=torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)
        self.GAT = self.GAT.to(device)

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
        # x = kwargs['data'].float()  #[n,1024]
        x, y = kwargs['data'].float(),kwargs['CT_data'].float()  #[1,n,1024]
        x, y = x.squeeze(0), y.squeeze(0)
        # print(x.shape)
        if x.shape[0] >= 10000:
            x = x[:10000,:]
        # print(x.shape)
        adj = kwargs['adj'].float() #adj是图网络的对象
        attention_only = kwargs['attention_only']
        x = torch.unsqueeze(x,dim=0)
        device = x.device
        # print(x.shape)
        x1 = self._fc1(x)  # [1,n,512]
        # print(x1.shape)
        A,x = self.forward_features(x1,adj) # [1,N+1,512]，A是图网络输出，此时x是transformer输出
        # print(A.shape,x.shape)

        # # ---->cls_token
        # A = self.norm(x)[:, 0]  # 取出类标签 经过多层感知机的分类，得到病理图像级别的诊断标签
        # print(A.shape)
        #
        x = torch.squeeze(x, dim=0)
        x1 = torch.squeeze(x1,dim=0)


        #-->cls_token
        x = self.norm1(x)
        A1 = self.fcA(x)
        # print(A1.shape)
        # print(A.shape,A1.shape)
        # AA = A + A1
        # print(A.shape)

        # print('HSA:', A1.shape)
        # print('HGNA:', A.shape)
        A1 = torch.mean(A1,dim=0)
        A = torch.mean(A, dim=1)
        # print('HSA:', A1.shape)
        #
        # print('HGNA:', A.shape)

        ## CT向量处理
        # y = self._fc1(y) #1*512
        # y = self.fcA(y) #1*32

        A = torch.unsqueeze(A, dim=1) #32*1
        A1 = torch.unsqueeze(A1, dim=0) #1*32
        # print('HGNA:', A.shape)
        # print('HSA:', A1.shape)

        A = F.softmax(A, dim=1)
        A1 = F.softmax(A1, dim=1)

        AA = torch.mm(A, A1)

        A_raw = AA
        if attention_only:
            return A_raw

        results_dict_instance = {}

        AA = AA.reshape(1,-1) #将张量改变为一个行向量
        # print(AA.shape)

        logits = self.classifiers(AA)
        # print(logits.shape)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        # print(Y_hat.shape)
        Y_prob = F.softmax(logits, dim=1)
        # print(Y_prob.shape)

        # results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        # return results_dict,results_dict_instance
        return logits, Y_prob, Y_hat, A_raw, results_dict_instance
        # print(AA)
        # A_raw = AA
        # if attention_only:
        #     return A_raw
        # # A = F.softmax(A, dim=1)  # softmax over N
        # # A1 = F.softmax(A1, dim=1)
        # # AA = F.softmax(AA, dim=1)
        # # print(AA)
        #
        # if kwargs['instance_eval']:
        #     total_inst_loss = 0.0
        #     all_preds = []
        #     all_targets = []
        #     inst_labels = F.one_hot(kwargs['label'], num_classes=self.num_classes).squeeze()  # binarize label
        #     for i in range(len(self.instance_classifiers)):
        #         inst_label = inst_labels[i].item()
        #         classifier = self.instance_classifiers[i]
        #         if inst_label == 1:  # in-the-class:
        #             instance_loss, preds, targets = self.inst_eval(A1[i], x, classifier)
        #             all_preds.extend(preds.cpu().numpy())
        #             all_targets.extend(targets.cpu().numpy())
        #         else:  # out-of-the-class
        #             if self.subtyping:
        #                 instance_loss, preds, targets = self.inst_eval_out(A1[i], x, classifier)
        #                 all_preds.extend(preds.cpu().numpy())
        #                 all_targets.extend(targets.cpu().numpy())
        #             else:
        #                 continue
        #         total_inst_loss += instance_loss
        #
        #     if self.subtyping:
        #         total_inst_loss /= len(self.instance_classifiers)
        #     results_dict_instance = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
        #                              'inst_preds': np.array(all_preds)}
        # else:
        #     results_dict_instance = {}
        #
        # M = torch.mm(AA, x)
        # # print(M.shape)
        # # logits = self.classifiers(M)
        # logits = torch.empty(1, self.num_classes).float().to(device)
        # for c in range(self.num_classes):
        #     logits[0, c] = self.classifiers[c](M[c])
        # # print(logits.shape)
        # Y_hat = torch.topk(logits, 1, dim=1)[1]
        # # print(Y_hat.shape)
        # Y_prob = F.softmax(logits, dim=1)
        # # print(Y_prob.shape)
        #
        # # results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        # # return results_dict,results_dict_instance
        # return logits, Y_prob, Y_hat, A_raw, results_dict_instance

        # x = self.norm1(x)[:, 0]  # 取出类标签 经过多层感知机的分类，得到病理图像级别的诊断标签
        # print(x.shape)
        # # ---->predict
        # logits = self._fc2(x)  # [B, n_classes]
        # Y_hat = torch.argmax(logits, dim=1)
        # Y_prob = F.softmax(logits, dim=1)
        # results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        # return results_dict


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
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=512,
                              depth=2,
                              num_heads=8,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes,
                              subtyping=True,
                              k_sample=2,
                              instance_loss_fn=instance_loss_fn,
                              nfeat=512,
                              nheads=4,
                              # dropout=0.1,  #这里待会改为0.25
                              dropout=0.6,  # 这里待会改为0.25
                              # drop_path_ratio=0.1,
                              alpha=0.4,
                              nhid=4
                              )
    # model = VisionTransformer_mb(img_size=224,
    #                           patch_size=16,
    #                           embed_dim=512,
    #                           depth=2,
    #                           num_heads=8,
    #                           representation_size=768 if has_logits else None,
    #                           num_classes=num_classes,
    #                           subtyping=True,
    #                           k_sample=8,
    #                           instance_loss_fn = instance_loss_fn)

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