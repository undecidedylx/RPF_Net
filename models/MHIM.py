import torch
import numpy as np
from math import ceil
from torch import nn, einsum
from einops import repeat, reduce, rearrange
# from modules.datten import *
import torch.nn.functional as F
# from modules.satten import *

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

def exists(val):
    return val is not None

def moore_penrose_iter_pinv(x, iters = 6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim = -1)
    row = abs_x.sum(dim = -2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device = device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z
class NystromAttention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_landmarks=256,
            pinv_iterations=6,
            residual=True,
            residual_conv_kernel=33,
            eps=1e-8,
            dropout=0.
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding=(padding, 0), groups=heads, bias=False)

    def forward(self, x, mask=None, return_attn=False):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        # pad so that sequence can be evenly divided into m landmarks

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value=0)

            if exists(mask):
                mask = F.pad(mask, (padding, 0), value=False)

        # derive query, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # set masked positions to 0 in queries, keys, values

        if exists(mask):
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q * self.scale

        # generate landmarks by sum reduction, and then calculate mean using the mask

        l = ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l=l)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l=l)

        # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean

        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, '... (n l) -> ... n', 'sum', l=l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        # masked mean (if mask exists)

        q_landmarks /= divisor
        k_landmarks /= divisor

        # similarities

        einops_eq = '... i d, ... j d -> ... i j'
        attn1 = einsum(einops_eq, q, k_landmarks)
        attn2 = einsum(einops_eq, q_landmarks, k_landmarks)
        attn3 = einsum(einops_eq, q_landmarks, k)

        # masking

        # if exists(mask):
        #     mask_value = -torch.finfo(q.dtype).max
        #     sim1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
        #     sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
        #     sim3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

        # eq (15) in the paper and aggregate values

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (attn1, attn2, attn3))
        attn2 = moore_penrose_iter_pinv(attn2, iters)
        out = (attn1 @ attn2) @ (attn3 @ v)

        # add depth-wise conv residual of values
        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        out = self.to_out(out)
        out = out[:, -n:]
        if return_attn:
            attn1 = attn1[:, :, -n].unsqueeze(-2) @ attn2
            attn1 = (attn1 @ attn3)

            return out, attn1[:, :, 0, -n + 1:]

        return out
class PPEG(nn.Module):
    def __init__(self, dim=512, k=7, conv_1d=False, bias=True):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim,
                                                                                                           (k, 1), 1,
                                                                                                           (k // 2, 0),
                                                                                                           groups=dim,
                                                                                                           bias=bias)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim,
                                                                                                            (5, 1), 1,
                                                                                                            (5 // 2, 0),
                                                                                                            groups=dim,
                                                                                                            bias=bias)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim,
                                                                                                            (3, 1), 1,
                                                                                                            (3 // 2, 0),
                                                                                                            groups=dim,
                                                                                                            bias=bias)

    def forward(self, x):
        B, N, C = x.shape

        # padding
        H, W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))

        add_length = H * W - N
        # if add_length >0:
        x = torch.cat([x, x[:, :add_length, :]], dim=1)

        if H < 7:
            H, W = 7, 7
            zero_pad = H * W - (N + add_length)
            x = torch.cat([x, torch.zeros((B, zero_pad, C), device=x.device)], dim=1)
            add_length += zero_pad

        # H, W = int(N**0.5),int(N**0.5)
        # cls_token, feat_token = x[:, 0], x[:, 1:]
        # feat_token = x
        cnn_feat = x.transpose(1, 2).view(B, C, H, W)

        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        # print(add_length)
        if add_length > 0:
            x = x[:, :-add_length]
        # x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x
class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, head=8):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=head,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
        )

    def forward(self, x, need_attn=False):
        if need_attn:
            z, attn = self.attn(self.norm(x), return_attn=need_attn)
            x = x + z
            return x, attn
        else:
            x = x + self.attn(self.norm(x))
            return x


class SAttention(nn.Module):

    def __init__(self, mlp_dim=512, pos_pos=0, pos='ppeg', peg_k=7, head=8):
        super(SAttention, self).__init__()
        self.norm = nn.LayerNorm(mlp_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, mlp_dim))

        self.layer1 = TransLayer(dim=mlp_dim, head=head)
        self.layer2 = TransLayer(dim=mlp_dim, head=head)

        if pos == 'ppeg':
            self.pos_embedding = PPEG(dim=mlp_dim, k=peg_k)
        # elif pos == 'sincos':
        #     self.pos_embedding = SINCOS(embed_dim=mlp_dim)
        # elif pos == 'peg':
        #     self.pos_embedding = PEG(512, k=peg_k)
        else:
            self.pos_embedding = nn.Identity()

        self.pos_pos = pos_pos

    # Modified by MAE@Meta
    def masking(self, x, ids_shuffle=None, len_keep=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        assert ids_shuffle is not None

        _, ids_restore = ids_shuffle.sort()

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x, mask_ids=None, len_keep=None, return_attn=False, mask_enable=False):
        batch, num_patches, C = x.shape

        attn = []

        if self.pos_pos == -2:
            x = self.pos_embedding(x)

        # masking
        if mask_enable and mask_ids is not None:
            x, _, _ = self.masking(x, mask_ids, len_keep)

        # cls_token
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=batch)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_pos == -1:
            x = self.pos_embedding(x)

        # translayer1
        if return_attn:
            x, _attn = self.layer1(x, True)
            attn.append(_attn.clone())
        else:
            x = self.layer1(x)

        # add pos embedding
        if self.pos_pos == 0:
            x[:, 1:, :] = self.pos_embedding(x[:, 1:, :])

        # translayer2
        if return_attn:
            x, _attn = self.layer2(x, True)
            attn.append(_attn.clone())
        else:
            x = self.layer2(x)

        # ---->cls_token
        x = self.norm(x)

        logits = x[:, 0, :]

        if return_attn:
            _a = attn
            return logits, _a
        else:
            return logits

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class SoftTargetCrossEntropy_v2(nn.Module):

    def __init__(self, temp_t=1., temp_s=1.):
        super(SoftTargetCrossEntropy_v2, self).__init__()
        self.temp_t = temp_t
        self.temp_s = temp_s

    def forward(self, x: torch.Tensor, target: torch.Tensor, mean: bool = True) -> torch.Tensor:
        loss = torch.sum(-F.softmax(target / self.temp_t, dim=-1) * F.log_softmax(x / self.temp_s, dim=-1), dim=-1)
        if mean:
            return loss.mean()
        else:
            return loss


class MHIM(nn.Module):
    def __init__(self, mlp_dim=512, mask_ratio=0, n_classes=2, temp_t=1., temp_s=1., dropout=0.25, act='relu',
                 select_mask=True, select_inv=False, msa_fusion='vote', mask_ratio_h=0., mrh_sche=None,
                 mask_ratio_hr=0., mask_ratio_l=0., da_act='gelu', baseline='selfattn', head=8, attn_layer=0):
        super(MHIM, self).__init__()

        self.mask_ratio = mask_ratio
        self.mask_ratio_h = mask_ratio_h
        self.mask_ratio_hr = mask_ratio_hr
        self.mask_ratio_l = mask_ratio_l
        self.select_mask = select_mask
        self.select_inv = select_inv
        self.msa_fusion = msa_fusion
        self.mrh_sche = mrh_sche
        self.attn_layer = attn_layer

        self.patch_to_emb = [nn.Linear(1024, 512)]

        if act.lower() == 'relu':
            self.patch_to_emb += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self.patch_to_emb += [nn.GELU()]

        self.dp = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.patch_to_emb = nn.Sequential(*self.patch_to_emb)

        if baseline == 'selfattn':
            self.online_encoder = SAttention(mlp_dim=mlp_dim, head=head)
        # elif baseline == 'attn':
        #     self.online_encoder = DAttention(mlp_dim, da_act)
        # elif baseline == 'dsmil':
        #     self.online_encoder = DSMIL(mlp_dim=mlp_dim, mask_ratio=mask_ratio)

        self.predictor = nn.Linear(mlp_dim, n_classes)

        self.temp_t = temp_t
        self.temp_s = temp_s

        self.cl_loss = SoftTargetCrossEntropy_v2(self.temp_t, self.temp_s)

        self.predictor_cl = nn.Identity()
        self.target_predictor = nn.Identity()

        self.apply(initialize_weights)
        self.hdaf = HDAF(dim=mlp_dim)

    def select_mask_fn(self, ps, attn, largest, mask_ratio, mask_ids_other=None, len_keep_other=None,
                       cls_attn_topk_idx_other=None, random_ratio=1., select_inv=False):
        ps_tmp = ps
        mask_ratio_ori = mask_ratio
        mask_ratio = mask_ratio / random_ratio
        if mask_ratio > 1:
            random_ratio = mask_ratio_ori
            mask_ratio = 1.

        # print(attn.size())
        if mask_ids_other is not None:
            if cls_attn_topk_idx_other is None:
                cls_attn_topk_idx_other = mask_ids_other[:, len_keep_other:].squeeze()
                ps_tmp = ps - cls_attn_topk_idx_other.size(0)
        if len(attn.size()) > 2:
            if self.msa_fusion == 'mean':
                _, cls_attn_topk_idx = torch.topk(attn, int(np.ceil((ps_tmp * mask_ratio)) // attn.size(1)),
                                                  largest=largest)
                cls_attn_topk_idx = torch.unique(cls_attn_topk_idx.flatten(-3, -1))
            elif self.msa_fusion == 'vote':
                vote = attn.clone()
                vote[:] = 0

                _, idx = torch.topk(attn, k=int(np.ceil((ps_tmp * mask_ratio))), sorted=False, largest=largest)
                mask = vote.clone()
                mask = mask.scatter_(2, idx, 1) == 1
                vote[mask] = 1
                vote = vote.sum(dim=1)
                _, cls_attn_topk_idx = torch.topk(vote, k=int(np.ceil((ps_tmp * mask_ratio))), sorted=False)
                # print(cls_attn_topk_idx.size())
                cls_attn_topk_idx = cls_attn_topk_idx[0]
        else:
            k = int(np.ceil((ps_tmp * mask_ratio)))
            _, cls_attn_topk_idx = torch.topk(attn, k, largest=largest)
            cls_attn_topk_idx = cls_attn_topk_idx.squeeze(0)

        # randomly
        if random_ratio < 1.:
            random_idx = torch.randperm(cls_attn_topk_idx.size(0), device=cls_attn_topk_idx.device)

            cls_attn_topk_idx = torch.gather(cls_attn_topk_idx, dim=0, index=random_idx[:int(
                np.ceil((cls_attn_topk_idx.size(0) * random_ratio)))])

        # concat other masking idx
        if mask_ids_other is not None:
            cls_attn_topk_idx = torch.cat([cls_attn_topk_idx, cls_attn_topk_idx_other]).unique()

        # if cls_attn_topk_idx is not None:
        len_keep = ps - cls_attn_topk_idx.size(0)
        a = set(cls_attn_topk_idx.tolist())
        b = set(list(range(ps)))
        mask_ids = torch.tensor(list(b.difference(a)), device=attn.device)
        if select_inv:
            mask_ids = torch.cat([cls_attn_topk_idx, mask_ids]).unsqueeze(0)
            len_keep = ps - len_keep
        else:
            mask_ids = torch.cat([mask_ids, cls_attn_topk_idx]).unsqueeze(0)

        return len_keep, mask_ids

    def get_mask(self, ps, i, attn, mrh=None):
        if attn is not None and isinstance(attn, (list, tuple)):
            if self.attn_layer == -1:
                attn = attn[1]
            else:
                attn = attn[self.attn_layer]
        else:
            attn = attn

        # random mask
        if attn is not None and self.mask_ratio > 0.:
            len_keep, mask_ids = self.select_mask_fn(ps, attn, False, self.mask_ratio, select_inv=self.select_inv,
                                                     random_ratio=0.001)
        else:
            len_keep, mask_ids = ps, None

        # low attention mask
        if attn is not None and self.mask_ratio_l > 0.:
            if mask_ids is None:
                len_keep, mask_ids = self.select_mask_fn(ps, attn, False, self.mask_ratio_l, select_inv=self.select_inv)
            else:
                cls_attn_topk_idx_other = mask_ids[:, :len_keep].squeeze() if self.select_inv else mask_ids[:,
                                                                                                   len_keep:].squeeze()
                len_keep, mask_ids = self.select_mask_fn(ps, attn, False, self.mask_ratio_l, select_inv=self.select_inv,
                                                         mask_ids_other=mask_ids, len_keep_other=ps,
                                                         cls_attn_topk_idx_other=cls_attn_topk_idx_other)

        # high attention mask
        mask_ratio_h = self.mask_ratio_h
        if self.mrh_sche is not None:
            mask_ratio_h = self.mrh_sche[i]
        if mrh is not None:
            mask_ratio_h = mrh
        if mask_ratio_h > 0.:
            # mask high conf patch
            if mask_ids is None:
                len_keep, mask_ids = self.select_mask_fn(ps, attn, largest=True, mask_ratio=mask_ratio_h,
                                                         len_keep_other=ps, random_ratio=self.mask_ratio_hr,
                                                         select_inv=self.select_inv)
            else:
                cls_attn_topk_idx_other = mask_ids[:, :len_keep].squeeze() if self.select_inv else mask_ids[:,
                                                                                                   len_keep:].squeeze()

                len_keep, mask_ids = self.select_mask_fn(ps, attn, largest=True, mask_ratio=mask_ratio_h,
                                                         mask_ids_other=mask_ids, len_keep_other=ps,
                                                         cls_attn_topk_idx_other=cls_attn_topk_idx_other,
                                                         random_ratio=self.mask_ratio_hr, select_inv=self.select_inv)

        return len_keep, mask_ids

    @torch.no_grad()
    def forward_teacher(self, x, return_attn=False):

        x = self.patch_to_emb(x)
        x = self.dp(x)

        if return_attn:
            x, attn = self.online_encoder(x, return_attn=True)
        else:
            x = self.online_encoder(x)
            attn = None

        return x, attn

    @torch.no_grad()
    def forward_test(self, x, return_attn=False, no_norm=False):
        x = self.patch_to_emb(x)
        x = self.dp(x)

        if return_attn:
            x, a = self.online_encoder(x, return_attn=True, no_norm=no_norm)
        else:
            x = self.online_encoder(x)
        x = self.predictor(x)

        if return_attn:
            return x, a
        else:
            return x

    def pure(self, x, return_attn=False):
        x = self.patch_to_emb(x)
        x = self.dp(x)
        ps = x.size(1)

        if return_attn:
            x, attn = self.online_encoder(x, return_attn=True)
        else:
            x = self.online_encoder(x)

        x = self.predictor(x)

        if self.training:
            if return_attn:
                return x, 0, ps, ps, attn
            else:
                return x, 0, ps, ps
        else:
            if return_attn:
                return x, attn
            else:
                return x

    def forward_loss(self, student_cls_feat, teacher_cls_feat):
        if teacher_cls_feat is not None:
            cls_loss = self.cl_loss(student_cls_feat, teacher_cls_feat.detach())
        else:
            cls_loss = 0.

        return cls_loss

    def forward(self, data, CT_data, attn=None, teacher_cls_feat=None, i=None):
        ct = torch.squeeze(CT_data, dim=0)  # [n,1024]
        ct = self.patch_to_emb(ct)
        if data.shape[1] >= 6000:
            x = data[:, :6000, :]
        else:
            x = data

        x = self.patch_to_emb(x)
        x = self.dp(x)

        ps = x.size(1)

        # get mask
        if self.select_mask:
            len_keep, mask_ids = self.get_mask(ps, i, attn)
        else:
            len_keep, mask_ids = ps, None

        # forward online network
        student_cls_feat = self.online_encoder(x, len_keep=len_keep, mask_ids=mask_ids, mask_enable=True)

        # prediction
        # student_logit = self.predictor(student_cls_feat)

        h = self.hdaf(torch.cat([student_cls_feat,ct], dim=0))
        h = torch.unsqueeze(torch.mean(h, dim=0), dim=0)
        logits = self.predictor(h)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        # print(Y_hat.shape)
        Y_prob = F.softmax(logits, dim=1)

        # cl loss
        # cls_loss = self.forward_loss(student_cls_feat=student_cls_feat, teacher_cls_feat=teacher_cls_feat)

        return logits, Y_prob, Y_hat

if __name__ == "__main__":
    torch.cuda.set_device(3)
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    data = torch.randn((1, 100, 1024)).cuda()
    model = MHIM().cuda()
    student_logit, cls_loss, ps, len_keep = model(data)
    print(student_logit)