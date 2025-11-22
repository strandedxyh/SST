# layers/Long_encoder.py —— 完整可替换版
from typing import Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
from layers.LWT_layers import *
from layers.RevIN import RevIN
from mamba_ssm import Mamba

class Long_encoder(nn.Module):
    def __init__(self, c_in:int, long_context_window:int, target_window:int, m_patch_len:int, m_stride:int,
                 m_layers:int=3, d_model=128, d_ff:int=256, norm:str='BatchNorm', dropout:float=0., act:str="gelu",
                 pre_norm:bool=False, d_state:int=16, d_conv:int=4, fc_dropout:float=0., head_dropout:float=0., padding_patch=None,
                 pretrain_head:bool=False, head_type='flatten', individual=False, revin=True, affine=True, subtract_last=False,
                 verbose:bool=False, **kwargs):
        super().__init__()
        self.m_patch_len = m_patch_len
        self.m_stride    = m_stride
        self.padding_patch = padding_patch
        m_patch_num = int((long_context_window - m_patch_len)/m_stride + 1)
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, m_stride))
            m_patch_num += 1

        # CI/GSP 开关（默认开启 CI；group_count=0 不分组）
        self.ci_proj = kwargs.get('ci_proj', True)
        self.group_count = kwargs.get('group_count', 0)

        self.backbone = Mamba_Encoder(c_in, m_patch_num, m_patch_len, m_layers, d_model, d_ff, dropout, act, d_state, d_conv,
                                      ci_proj=self.ci_proj, group_count=self.group_count)

    def forward(self, z):  # [B, C, L]
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.m_patch_len, step=self.m_stride)   # [B, C, m_patch_num, m_patch_len]
        z = z.permute(0, 1, 3, 2)                                               # [B, C, m_patch_len, m_patch_num]
        z = self.backbone(z)                                                    # [B, C, d_model, m_patch_num]
        return z


class Mamba_Encoder(nn.Module):
    def __init__(self, c_in, m_patch_num, m_patch_len, m_layers, d_model, d_ff, dropout, act, d_state, d_conv,
                 ci_proj=True, group_count=0):
        super().__init__()
        self.c_in = c_in
        self.m_layers = m_layers
        self.m_patch_len = m_patch_len
        self.m_patch_num = m_patch_num
        self.ci_proj = ci_proj
        self.group_count = group_count

        # Input encoding（CI/GSP）
        if self.group_count and self.group_count > 0:
            self.groups = self.group_count
            self.group_linears = nn.ModuleList([nn.Linear(m_patch_len, d_model) for _ in range(self.groups)])
        elif self.ci_proj:
            self.W_P_list = nn.ModuleList([nn.Linear(m_patch_len, d_model) for _ in range(c_in)])
        else:
            self.W_P = nn.Linear(m_patch_len, d_model)

        # Mamba 层堆叠
        self.mamba_layers = nn.ModuleList([
            Mamba_Encoder_Layer(d_model, d_ff, dropout, act, d_state, d_conv, m_patch_len)
            for _ in range(m_layers)
        ])

    def _proj_one(self, x_ci: Tensor, idx: int) -> Tensor:
        # x_ci: [B, m_patch_num, m_patch_len] -> [B, m_patch_num, d_model]
        if self.group_count and self.group_count > 0:
            group_size = (self.c_in + self.group_count - 1) // self.group_count
            g = min(idx // group_size, self.group_count - 1)
            return self.group_linears[g](x_ci)
        elif self.ci_proj:
            return self.W_P_list[idx](x_ci)
        else:
            return self.W_P(x_ci)

    def forward(self, x):  # [B, C, m_patch_len, m_patch_num]
        B, C, P, N = x.shape
        assert C == self.c_in and P == self.m_patch_len and N == self.m_patch_num
        x = x.permute(0, 1, 3, 2)   # [B, C, m_patch_num, m_patch_len]

        proj_list = []
        for i in range(C):
            xi = x[:, i, :, :]               # [B, m_patch_num, m_patch_len]
            ui = self._proj_one(xi, i)       # [B, m_patch_num, d_model]
            proj_list.append(ui)
        u = torch.stack(proj_list, dim=1)    # [B, C, m_patch_num, d_model]

        z = u.reshape(B*C, N, -1)            # [B*C, m_patch_num, d_model]
        for i in range(self.m_layers):
            z = self.mamba_layers[i](z)      # [B*C, m_patch_num, d_model]

        z = z.view(B, C, N, -1).permute(0, 1, 3, 2)  # [B, C, d_model, m_patch_num]
        return z


class Mamba_Encoder_Layer(nn.Module):
    def __init__(self, d_model, d_ff, dropout, act, d_state, d_conv, m_patch_len):
        super().__init__()
        self.mamba = Mamba(d_model, d_state=d_state, d_conv=d_conv)
        self.lin1  = nn.Linear(d_model, d_ff)
        self.lin2  = nn.Linear(d_ff, d_model)
        self.ln    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = F.relu if act == "relu" else F.gelu

    def forward(self, x):  # [B*, N, d_model]
        x = self.mamba(x)
        x = self.lin2(self.act(self.lin1(x)))
        return x
