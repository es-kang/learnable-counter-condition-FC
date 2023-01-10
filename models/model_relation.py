# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg
from timm.models.layers import trunc_normal_, DropPath
from torch.autograd import Variable
import torch.nn.functional as F
import copy
import numpy as np
import math

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_sinusoid_encoding_table(n_position, d_model):

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table)


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.3):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(F.gelu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (max_seq_len ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (max_seq_len ** ((2 * (i + 1)) / d_model)))

        # pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, seq_len):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)

        # add constant to embedding
        # seq_len = x.size(1)
        for i, s in enumerate(seq_len):
            x[i, :s, :] = x[i, :s, :] + Variable(self.pe[:s], requires_grad=False).cuda()
        return x

class Attention_talking_head(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=True, attn_drop=0.3, proj_drop=0.3):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads

        # self.scale = qk_scale or head_dim ** -0.5
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale_Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=True, qk_scale=True, proj_drop=0.3, attn_drop=0.3,
                 act_layer=nn.GELU, norm_layer=Norm, Attention_block=Attention_talking_head, Mlp_block=Mlp, init_values=1e-3):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=proj_drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class NonLinearEmbedding(nn.Module):
    def __init__(self, input, dropout=0.3):
        super().__init__()

        self.emb = nn.Sequential(nn.Linear(input, int(input/2)),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(int(input/2), input),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 )
    def forward(self, x):

        return self.emb(x)


class FCTransformer(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.intra_net = NonLinearEmbedding(args.input_size)
        self.encoding_block_sa = get_clones(LayerScale_Block(dim=args.input_size, num_heads=args.num_heads_sa,
                                                             mlp_hidden_dim=args.d_ff), args.num_stack_sa)

        self.embed_pos = PositionalEncoder(args.input_size, int(args.input_size))
        self.cls_token = nn.Parameter(torch.randn(1, 1, args.input_size), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            nn.init.xavier_normal_(weight)

    def set_require_grad(self, isgrad):
        for param in self.parameters():
            param.requires_grad = isgrad

    def forward(self, x):
        x = self.intra_net(x)
        cls_token = self.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.embed_pos(x, np.repeat(self.args.input_size, x.shape[0]))
        for i, blk in enumerate(self.encoding_block_sa):
            x = blk(x)
        return x



class FCTransformer_woIntra(nn.Module):

    def __init__(self, args):
        super().__init__()
        "ViT"
        self.args = args
        self.encoding_block_sa = get_clones(LayerScale_Block(dim=args.input_size, num_heads=args.num_heads_sa,
                                                             mlp_hidden_dim=args.d_ff), args.num_stack_sa)

        self.embed_pos = PositionalEncoder(args.input_size, int(args.input_size))
        # self.norm = Norm(args.input_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, args.input_size), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            nn.init.xavier_normal_(weight)
            # nn.init.kaiming_normal(weight)

            # stv = 1. / math.sqrt(weight.size(1))
            # nn.init.uniform_(weight, -stv, stv)

    def set_require_grad(self, isgrad):
        for param in self.parameters():
            param.requires_grad = isgrad

    def forward(self, x):
        cls_token = self.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.embed_pos(x, np.repeat(self.args.input_size, x.shape[0]))
        for i, blk in enumerate(self.encoding_block_sa):
            x = blk(x)
        return x