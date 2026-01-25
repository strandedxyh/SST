# layers/Long_encoder.py  —— 长程 Mamba 编码器 + 通道分组投影（CI / GSP）
from typing import Optional
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from mamba_ssm import Mamba  # 你原来工程里就有的依赖


class Long_encoder(nn.Module):
    """
    长程编码器：先做时间维 patch，再用 Mamba 堆叠做建模。
    这里只负责：
      - patch 划分
      - 调用 Mamba_Encoder（里面做通道分组投影 + Mamba）
    """
    def __init__(
        self,
        c_in: int,
        long_context_window: int,
        target_window: int,
        m_patch_len: int,
        m_stride: int,
        m_layers: int = 3,
        d_model: int = 128,
        d_ff: int = 256,
        norm: str = 'BatchNorm',
        dropout: float = 0.,
        act: str = "gelu",
        pre_norm: bool = False,
        d_state: int = 16,
        d_conv: int = 4,
        fc_dropout: float = 0.,
        head_dropout: float = 0.,
        padding_patch: Optional[str] = None,
        pretrain_head: bool = False,
        head_type: str = 'flatten',
        individual: bool = False,
        revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.c_in = c_in
        self.m_patch_len = m_patch_len
        self.m_stride = m_stride
        self.padding_patch = padding_patch

        # 计算 patch 数量
        m_patch_num = int((long_context_window - m_patch_len) / m_stride + 1)
        if padding_patch == 'end':
            # 在时间结尾做 padding，保证可以多取一个 patch
            self.padding_patch_layer = nn.ReplicationPad1d((0, m_stride))
            m_patch_num += 1

        # 读取通道分组超参（来自 configs.group_count）
        group_count = kwargs.get("group_count", 0)

        # 主体编码器：包含通道分组投影 + 多层 Mamba
        self.backbone = Mamba_Encoder(
            c_in=c_in,
            m_patch_num=m_patch_num,
            m_patch_len=m_patch_len,
            m_layers=m_layers,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            act=act,
            d_state=d_state,
            d_conv=d_conv,
            group_count=group_count,
        )

        # 调试信息：看一下当前到底是 shared / group / per_channel
        print(f"[DEBUG][LongEncoder] c_in={c_in}, group_count={group_count}, mode={self.backbone.mode}")

    def forward(self, z: Tensor) -> Tensor:
        """
        输入：z [B, C, L]
        输出： [B, C, d_model, m_patch_num]
        """
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)  # [B, C, L + stride]

        # unfold 成 patch：[B, C, patch_num, patch_len]
        z = z.unfold(dimension=-1, size=self.m_patch_len, step=self.m_stride)
        # 调整成 [B, C, patch_len, patch_num]
        z = z.permute(0, 1, 3, 2)

        # 交给 Mamba_Encoder 做通道分组投影 + Mamba 堆叠
        z = self.backbone(z)  # [B, C, d_model, m_patch_num]
        return z


class Mamba_Encoder(nn.Module):
    """
    长程 Mamba 编码器：
      - 先在 patch 维度做通道分组投影（共享 / 分组 / 独立）
      - 再把 [B,C,N,d_model] 展平为 [B*C,N,d_model] 送入多层 Mamba
    """
    def __init__(
        self,
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
    ):
        super().__init__()

        self.c_in = c_in
        self.m_patch_num = m_patch_num
        self.m_patch_len = m_patch_len
        self.m_layers = m_layers
        self.d_model = d_model

        # ========= 通道分组投影（CI / GSP）核心逻辑 =========
        # 约定：
        #   group_count <= 1       -> 所有变量共享一套 W_P（shared）
        #   1 < group_count < C    -> 按顺序均分成 group_count 组，每组变量共享一套 W_P（group）
        #   group_count >= C      -> 每个变量一套 W_P（per_channel）
        if group_count is None or group_count <= 1:
            self.mode = "shared"
            self.shared_proj = nn.Linear(m_patch_len, d_model)
        elif group_count >= c_in:
            self.mode = "per_channel"
            self.per_channel_proj = nn.ModuleList(
                nn.Linear(m_patch_len, d_model) for _ in range(c_in)
            )
        else:
            self.mode = "group"
            self.group_count = group_count
            self.group_proj = nn.ModuleList(
                nn.Linear(m_patch_len, d_model) for _ in range(self.group_count)
            )

        # ========= Mamba 层堆叠 =========
        self.layers = nn.ModuleList(
            [
                Mamba_Encoder_Layer(
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    act=act,
                    d_state=d_state,
                    d_conv=d_conv,
                )
                for _ in range(m_layers)
            ]
        )

    def _project(self, x: Tensor) -> Tensor:
        """
        x: [B, C, m_patch_len, m_patch_num]
        返回: [B, C, m_patch_num, d_model]
        """
        B, C, P, N = x.shape
        assert C == self.c_in and P == self.m_patch_len and N == self.m_patch_num

        # 调整成 [B, C, N, P]，方便最后一维做 Linear
        x = x.permute(0, 1, 3, 2)

        if self.mode == "shared":
            # 所有变量共享一套 W_P
            u = self.shared_proj(x)  # [B, C, N, d_model]
        elif self.mode == "per_channel":
            outs = []
            for i in range(C):
                xi = x[:, i, :, :]  # [B, N, P]
                ui = self.per_channel_proj[i](xi)  # [B, N, d_model]
                outs.append(ui)
            u = torch.stack(outs, dim=1)  # [B, C, N, d_model]
        else:  # group 模式
            outs = []
            group_size = math.ceil(C / self.group_count)
            for i in range(C):
                g = min(i // group_size, self.group_count - 1)
                xi = x[:, i, :, :]
                ui = self.group_proj[g](xi)
                outs.append(ui)
            u = torch.stack(outs, dim=1)

        return u  # [B, C, N, d_model]

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [B, C, m_patch_len, m_patch_num]
        输出: [B, C, d_model, m_patch_num]
        """
        B, C, P, N = x.shape
        u = self._project(x)              # [B, C, N, d_model]
        z = u.reshape(B * C, N, self.d_model)  # [B*C, N, d_model]

        for layer in self.layers:
            z = layer(z)                  # [B*C, N, d_model]

        z = z.view(B, C, N, self.d_model).permute(0, 1, 3, 2)
        # -> [B, C, d_model, m_patch_num]
        return z


class Mamba_Encoder_Layer(nn.Module):
    """
    单层 Mamba + 简单 FFN
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float,
        act: str,
        d_state: int,
        d_conv: int,
    ):
        super().__init__()

        self.mamba = Mamba(d_model, d_state=d_state, d_conv=d_conv)
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = F.relu if act == "relu" else F.gelu

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [B*, N, d_model]
        """
        x = self.mamba(x)
        x = self.lin2(self.act(self.lin1(x)))
        return x
