import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention
from torch import Tensor
import math


class Attention(nn.Module):
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

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # self.shared_MLP = nn.Sequential(
        #     # nn.Linear(in_planes, in_planes // ratio),
        #     nn.Conv2d(in_planes, in_planes,1, bias=False),
        #     nn.LeakyReLU(),
        #     # nn.Linear(in_planes // ratio, in_planes)
        # )
        self.avg_out = nn.AdaptiveAvgPool1d(1)
        self.max_out = nn.AdaptiveMaxPool1d(1)
        # self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        # self.relu1 = nn.ReLU()
        # self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        cls_token, feat_token = x[:, 0], x[:, 1:]
        # MLP_out =self.shared_MLP(feat_token)# self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        MLP_out = feat_token
        avg_out = self.avg_out(MLP_out)
        max_out = self.max_out(MLP_out)

        out = avg_out + max_out
        feat_token = self.sigmoid(out) * feat_token
        x = torch.cat((cls_token.unsqueeze(1), feat_token), dim=1)
        return x


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout = 0.5
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG1(nn.Module):
    def __init__(self, dim=512):
        super(PPEG1, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        # cls_token, feat_token = x[:, 0], x[:, 1:]
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        # cnn_feat = feat_token.view(B, N-1, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        # x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x
class PPEG2(nn.Module):
    def __init__(self, dim=512):
        super(PPEG2, self).__init__()
        self.proj = nn.Conv3d(dim, dim, kernel_size=(7, 7, 7),stride=(1, 1, 1), padding=(3, 3, 3), groups=dim)
        self.proj1 = nn.Conv3d(dim, dim, kernel_size=(5, 5, 5),stride=(1, 1, 1), padding=(2, 2, 2),  groups=dim)
        self.proj2 = nn.Conv3d(dim, dim, kernel_size=(3, 3, 3),stride=(1, 1, 1), padding=(1, 1, 1), groups=dim)

    def forward(self, x, H, W):
        _, B, N, C = x.shape
        # cls_token, feat_token = x[:,:, 0], x[:,:, 1:]
        feat_token = x
        # cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        cnn_feat = feat_token.view(B, C, H, W).unsqueeze(0)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(3).transpose(2, 3)
        # x = torch.cat((cls_token.unsqueeze(1), x), dim=2)
        return x


class TransMIL(nn.Module):
    def __init__(self, n_classes):
        super(TransMIL, self).__init__()
        self.pos_layer1 = PPEG1(dim=256)
        self.pos_layer2 = PPEG2(dim=1)
        # self.cal = ChannelAttention(in_planes=512)
        self.attn = Attention(dim=512,type='fuse',qkv_bias=True)
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        # self.layer2 = TransLayer(dim=512)
        self.norm1 = nn.LayerNorm(1024)
        # self.norm2 = nn.LayerNorm(512)
        # self._fc2 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0))
        self._fc3 = nn.Sequential(nn.Linear(512, self.n_classes), nn.Dropout(0))


    def forward(self, **kwargs):
        h, y = kwargs['data'].float(), kwargs['CT_data'].float()  # [B, n, 1024]
        h = self.norm1(h)
        y = self.norm1(y)
        h = self._fc1(h)  # [B, n, 512]
        y = self._fc1(y)  # [B, n, 512]
        # cls_tokens = torch.mean(h, dim=1).unsqueeze(1)

        # # ---->pad
        # H = h.shape[1]
        # _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        # add_length = _H * _W - H
        # h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]
        #
        # # ---->cls_token
        # # B = h.shape[0]
        # # cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        # # h = torch.cat((cls_tokens, h), dim=1)
        # # h = self.norm(h)
        #
        # # ---->Translayer x1
        # # h = self.layer1(h)  # [B, N, 512]
        #
        # # h_c = self.cal(h)
        #
        # # ---->PPEG
        # h_2 = self.pos_layer1(h, _H, _W)  # [B, N, 512]
        # h_3 = self.pos_layer2(h.unsqueeze(0), _H, _W).squeeze(0)  # [B, N, 512]
        #
        # # h = h_2 + h_3
        # h = torch.cat([h_2,h_3],dim=2)
        #
        # # ---->Translayer x2
        # h = self.layer1(h)  # [B, N, 512]
        #
        # # ---->cls_token
        # # h = self.norm2(h)
        # h = torch.mean(h,dim=1)

        A_raw = h

        # mix = torch.cat([h.unsqueeze(0),y],dim=1)
        # mix,_ = self.attn(mix)
        # mix = mix.flatten(1)
        # mix = torch.mean(mix,dim=1)

        # ---->predict
        # mix = self._fc2(h)  # [B, n_classes]
        logits = self._fc3(y).squeeze(0)  # [B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}

        return logits, Y_prob, Y_hat, A_raw, results_dict


if __name__ == "__main__":
    data = torch.randn((1, 60000, 1024)).cuda()
    ct_data = torch.randn((1, 1, 1024)).cuda()
    model = TransMIL(n_classes=2).cuda()
    # model = Attention(dim=1024,type='fuse',qkv_bias=True).cuda()
    print(model.eval())
    results_dict = model(data = data, CT_data = ct_data)
    print(results_dict)