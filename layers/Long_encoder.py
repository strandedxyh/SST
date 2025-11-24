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
    """
    Long-range Mamba encoder with flexible patch projection sharing.

    - group_count <= 1      : shared projection (all channels share one Linear)
    - 1 < group_count < C   : group-shared projection (group_count groups)
    - group_count >= C      : per-channel projection

    ci_proj 参数保留为“兼容旧代码”的占位，但不再影响逻辑。
    """
    def __init__(self,
                 c_in: int,
                 m_patch_num: int,
                 m_patch_len: int,
                 m_layers: int,
                 d_model: int,
                 d_ff: int,
                 dropout: float,
                 act: str,
                 d_state: int,
                 d_conv: int,
                 group_count: int = 0,
                 ci_proj: bool = False,
                 **kwargs):
        super().__init__()
        self.c_in = c_in
        self.m_layers = m_layers
        self.m_patch_len = m_patch_len
        self.m_patch_num = m_patch_num

        # ★ 关键：完全由 group_count 控制，不再用 ci_proj 改写
        self.group_count = int(group_count)

        # ====== 根据 group_count 决定模式 & 创建线性层 ======
        if self.group_count <= 1:
            self.mode = 'shared'
            self.W_P = nn.Linear(m_patch_len, d_model)
        elif self.group_count >= self.c_in:
            self.mode = 'per_channel'
            self.W_P_list = nn.ModuleList(
                [nn.Linear(m_patch_len, d_model) for _ in range(self.c_in)]
            )
        else:
            self.mode = 'group'
            self.groups = self.group_count
            self.group_size = math.ceil(self.c_in / self.groups)
            self.group_linears = nn.ModuleList(
                [nn.Linear(m_patch_len, d_model) for _ in range(self.groups)]
            )

        print(f"[DEBUG][LongEncoder] c_in={self.c_in}, group_count={self.group_count}, mode={self.mode}")

        # ====== 原来的 Mamba 层堆叠，基本保持原逻辑 ======
        self.mamba_layers = nn.ModuleList([
            Mamba_Encoder_Layer(
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
                act=act,
                d_state=d_state,
                d_conv=d_conv,
                m_patch_len=m_patch_len
            )
            for _ in range(m_layers)
        ])
    def _proj_one(self, x_ci: torch.Tensor, idx: int) -> torch.Tensor:
        """
        x_ci: [B, m_patch_num, m_patch_len]  当前第 idx 个变量的所有 patch
        返回: [B, m_patch_num, d_model]
        """
        if self.mode == 'shared':
            return self.W_P(x_ci)
        elif self.mode == 'per_channel':
            return self.W_P_list[idx](x_ci)
        else:
            g = min(idx // self.group_size, self.groups - 1)
            return self.group_linears[g](x_ci)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, m_patch_len, m_patch_num]
        返回: [B, C, d_model, m_patch_num]
        """
        B, C, P, N = x.shape
        assert C == self.c_in and P == self.m_patch_len and N == self.m_patch_num

        # [B, C, m_patch_num, m_patch_len]
        x = x.permute(0, 1, 3, 2)

        # 对每个变量应用对应的线性层
        proj_list = []
        for i in range(C):
            xi = x[:, i, :, :]          # [B, N, P]
            ui = self._proj_one(xi, i)  # [B, N, d_model]
            proj_list.append(ui)
        # [B, C, N, d_model]
        u = torch.stack(proj_list, dim=1)

        # 展平成 [B*C, N, d_model] 送入 Mamba 层
        z = u.reshape(B * C, N, -1)
        for layer in self.mamba_layers:
            z = layer(z)                # [B*C, N, d_model]

        # 还原回 [B, C, d_model, N]
        z = z.reshape(B, C, N, -1).permute(0, 1, 3, 2)
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
