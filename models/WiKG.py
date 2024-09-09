import torch

torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention

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
class WiKG(nn.Module):
    def __init__(self, dim_in=384, dim_hidden=512, topk=6, n_classes=2, agg_type='bi-interaction', dropout=0.3,
                 pool='attn'):
        super().__init__()

        self._fc1 = nn.Sequential(nn.Linear(dim_in, dim_hidden), nn.LeakyReLU())

        self.W_head = nn.Linear(dim_hidden, dim_hidden)
        self.W_tail = nn.Linear(dim_hidden, dim_hidden)

        self.scale = dim_hidden ** -0.5
        self.topk = topk
        self.agg_type = agg_type

        self.gate_U = nn.Linear(dim_hidden, dim_hidden // 2)
        self.gate_V = nn.Linear(dim_hidden, dim_hidden // 2)
        self.gate_W = nn.Linear(dim_hidden // 2, dim_hidden)

        if self.agg_type == 'gcn':
            self.linear = nn.Linear(dim_hidden, dim_hidden)
        elif self.agg_type == 'sage':
            self.linear = nn.Linear(dim_hidden * 2, dim_hidden)
        elif self.agg_type == 'bi-interaction':
            self.linear1 = nn.Linear(dim_hidden, dim_hidden)
            self.linear2 = nn.Linear(dim_hidden, dim_hidden)
        else:
            raise NotImplementedError

        self.activation = nn.LeakyReLU()
        self.message_dropout = nn.Dropout(dropout)
        self.hdaf = HDAF(dim=dim_hidden)

        self.norm = nn.LayerNorm(dim_hidden)
        self.fc = nn.Linear(dim_hidden, n_classes)

        if pool == "mean":
            self.readout = global_mean_pool
        elif pool == "max":
            self.readout = global_max_pool
        elif pool == "attn":
            att_net = nn.Sequential(nn.Linear(dim_hidden, dim_hidden // 2), nn.LeakyReLU(),
                                    nn.Linear(dim_hidden // 2, 1))
            self.readout = GlobalAttention(att_net)

    def forward(self, data, CT_data):
        ct = torch.squeeze(CT_data, dim=0)  # [n,1024]
        ct = self._fc1(ct)
        # if data.shape[1] >= 6000:
        #     x = data[:, :6000, :]
        # else:
        #     x = data
        x = data

        x = self._fc1(x)  # [B,N,C]

        # B, N, C = x.shape
        x = (x + x.mean(dim=1, keepdim=True)) * 0.5 #简单的归一化

        e_h = self.W_head(x)
        e_t = self.W_tail(x)

        # construct neighbour
        attn_logit = (e_h * self.scale) @ e_t.transpose(-2, -1)  #@是矩阵点乘
        topk_weight, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)

        # add an extra dimension to the index tensor, making it available for advanced indexing, aligned with the dimensions of e_t
        topk_index = topk_index.to(torch.long)

        # expand topk_index dimensions to match e_t
        topk_index_expanded = topk_index.expand(e_t.size(0), -1, -1)  # shape: [1, 10000, 4]

        # create a RANGE tensor to help indexing
        batch_indices = torch.arange(e_t.size(0)).view(-1, 1, 1).to(topk_index.device)  # shape: [1, 1, 1]

        Nb_h = e_t[batch_indices, topk_index_expanded, :]  # shape: [1, 10000, 4, 512]

        # use SoftMax to obtain probability
        topk_prob = F.softmax(topk_weight, dim=2)
        eh_r = torch.mul(topk_prob.unsqueeze(-1), Nb_h) + torch.matmul((1 - topk_prob).unsqueeze(-1),
                                                                       e_h.unsqueeze(2))  # 1 pixel wise   2 matmul

        # gated knowledge attention
        e_h_expand = e_h.unsqueeze(2).expand(-1, -1, self.topk, -1)
        gate = torch.tanh(e_h_expand + eh_r)
        ka_weight = torch.einsum('ijkl,ijkm->ijk', Nb_h, gate) #对最后一维进行内积运算

        ka_prob = F.softmax(ka_weight, dim=2).unsqueeze(dim=2)
        e_Nh = torch.matmul(ka_prob, Nb_h).squeeze(dim=2)

        if self.agg_type == 'gcn':
            embedding = e_h + e_Nh
            embedding = self.activation(self.linear(embedding))
        elif self.agg_type == 'sage':
            embedding = torch.cat([e_h, e_Nh], dim=2)
            embedding = self.activation(self.linear(embedding))
        elif self.agg_type == 'bi-interaction':
            sum_embedding = self.activation(self.linear1(e_h + e_Nh))
            bi_embedding = self.activation(self.linear2(e_h * e_Nh))
            embedding = sum_embedding + bi_embedding

        h = self.message_dropout(embedding)

        h = self.readout(h.squeeze(0), batch=None)
        h = self.norm(h)

        # h = self.hdaf(torch.cat([h,ct], dim=0))
        # h = torch.unsqueeze(torch.mean(h, dim=0), dim=0)
        logits = self.fc(h)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        # print(Y_hat.shape)
        Y_prob = F.softmax(logits, dim=1)

        return logits, Y_prob, Y_hat


if __name__ == "__main__":
    data = torch.randn((1, 100, 1024)).cuda()
    model = WiKG(dim_in=1024, dim_hidden=512, topk=6, n_classes=2, agg_type='bi-interaction', dropout=0.3,
                 pool='attn').cuda()
    output = model(data)
    print(output.shape)