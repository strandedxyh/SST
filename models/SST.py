# models/SST.py  —— 顶层 SST 模型，支持：
#   1）router_type = global / per_var
#   2）长程 / 短程编码器内部的通道分组投影（group_count）

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from typing import Optional

from layers.Short_encoder import Short_encoder
from layers.LWT_layers import series_decomp
from layers.Long_encoder import Long_encoder
from layers.RevIN import RevIN


class Model(nn.Module):
    """
    SST (State Space Transformer)
    """
    def __init__(
        self,
        configs,
        max_seq_len: Optional[int] = 1024,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        norm: str = 'BatchNorm',
        attn_dropout: float = 0.,
        act: str = "gelu",
        key_padding_mask: bool = 'auto',
        padding_var: Optional[int] = None,
        attn_mask: Optional[Tensor] = None,
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = 'zeros',
        learn_pe: bool = True,
        pretrain_head: bool = False,
        head_type: str = 'flatten',
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__()

        # ====== 从 configs 读取超参（基本和原版一致） ======
        long_context_window = configs.seq_len
        m_layers = configs.m_layers
        d_state = configs.d_state
        d_conv = configs.d_conv
        m_patch_len = configs.m_patch_len
        m_stride = configs.m_stride

        c_in = configs.enc_in
        context_window = configs.label_len
        self.label_len = configs.label_len
        target_window = configs.pred_len
        local_ws = configs.local_ws

        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        individual = configs.individual

        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last

        decomposition = configs.decomposition
        kernel_size = configs.kernel_size

        concat = configs.concat

        # ===== 新增开关：路由 + 通道分组 =====
        self.router_type = getattr(configs, "router_type", "global")   # 'global' or 'per_var'
        self.group_count = getattr(configs, "group_count", 0)          # 通道分组数

        print(f"[DEBUG][SST.Model] router_type={self.router_type}, group_count={self.group_count}")

        # ===== patch 数量，给 Head 准备 flatten 维度 =====
        m_patch_num = int((long_context_window - m_patch_len) / m_stride + 1)
        if padding_patch == 'end':
            m_patch_num += 1
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == 'end':
            patch_num += 1

        c_out = configs.c_out
        m_head_nf = d_model * m_patch_num
        t_head_nf = d_model * patch_num
        head_nf = d_model * (m_patch_num + patch_num)

        # ===== RevIN =====
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # ===== 长程编码器：Mamba + 通道分组投影 =====
        self.long_encoder = Long_encoder(
            c_in=c_in,
            long_context_window=long_context_window,
            target_window=target_window,
            m_patch_len=m_patch_len,
            m_stride=m_stride,
            m_layers=m_layers,
            d_model=d_model,
            d_ff=d_ff,
            norm=norm,
            dropout=dropout,
            act=act,
            pre_norm=pre_norm,
            d_state=d_state,
            d_conv=d_conv,
            fc_dropout=fc_dropout,
            head_dropout=head_dropout,
            padding_patch=padding_patch,
            pretrain_head=pretrain_head,
            head_type=head_type,
            individual=individual,
            revin=revin,
            affine=affine,
            subtract_last=subtract_last,
            verbose=verbose,
            group_count=self.group_count,   # ⭐ 关键：把 group_count 传进去
            **kwargs,
        )

        # ===== 短程编码器：Transformer + 通道分组投影 =====
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = Short_encoder(
                c_in=c_in,
                context_window=context_window,
                target_window=target_window,
                local_ws=local_ws,
                patch_len=patch_len,
                stride=stride,
                max_seq_len=max_seq_len,
                n_layers=n_layers,
                d_model=d_model,
                n_heads=n_heads,
                d_k=d_k,
                d_v=d_v,
                d_ff=d_ff,
                norm=norm,
                attn_dropout=attn_dropout,
                dropout=dropout,
                act=act,
                key_padding_mask=key_padding_mask,
                padding_var=padding_var,
                attn_mask=attn_mask,
                res_attention=res_attention,
                pre_norm=pre_norm,
                store_attn=store_attn,
                pe=pe,
                learn_pe=learn_pe,
                fc_dropout=fc_dropout,
                head_dropout=head_dropout,
                padding_patch=padding_patch,
                pretrain_head=pretrain_head,
                head_type=head_type,
                individual=individual,
                revin=revin,
                affine=affine,
                subtract_last=subtract_last,
                verbose=verbose,
                group_count=self.group_count,   # ⭐
                **kwargs,
            )
            self.model_res = Short_encoder(
                c_in=c_in,
                context_window=context_window,
                target_window=target_window,
                local_ws=local_ws,
                patch_len=patch_len,
                stride=stride,
                max_seq_len=max_seq_len,
                n_layers=n_layers,
                d_model=d_model,
                n_heads=n_heads,
                d_k=d_k,
                d_v=d_v,
                d_ff=d_ff,
                norm=norm,
                attn_dropout=attn_dropout,
                dropout=dropout,
                act=act,
                key_padding_mask=key_padding_mask,
                padding_var=padding_var,
                attn_mask=attn_mask,
                res_attention=res_attention,
                pre_norm=pre_norm,
                store_attn=store_attn,
                pe=pe,
                learn_pe=learn_pe,
                fc_dropout=fc_dropout,
                head_dropout=head_dropout,
                padding_patch=padding_patch,
                pretrain_head=pretrain_head,
                head_type=head_type,
                individual=individual,
                revin=revin,
                affine=affine,
                subtract_last=subtract_last,
                verbose=verbose,
                group_count=self.group_count,   # ⭐
                **kwargs,
            )
        else:
            self.model = Short_encoder(
                c_in=c_in,
                context_window=context_window,
                target_window=target_window,
                local_ws=local_ws,
                patch_len=patch_len,
                stride=stride,
                max_seq_len=max_seq_len,
                n_layers=n_layers,
                d_model=d_model,
                n_heads=n_heads,
                d_k=d_k,
                d_v=d_v,
                d_ff=d_ff,
                norm=norm,
                attn_dropout=attn_dropout,
                dropout=dropout,
                act=act,
                key_padding_mask=key_padding_mask,
                padding_var=padding_var,
                attn_mask=attn_mask,
                res_attention=res_attention,
                pre_norm=pre_norm,
                store_attn=store_attn,
                pe=pe,
                learn_pe=learn_pe,
                fc_dropout=fc_dropout,
                head_dropout=head_dropout,
                padding_patch=padding_patch,
                pretrain_head=pretrain_head,
                head_type=head_type,
                individual=individual,
                revin=revin,
                affine=affine,
                subtract_last=subtract_last,
                verbose=verbose,
                group_count=self.group_count,   # ⭐
                **kwargs,
            )

        # ===== Router（global / per_var） =====
        if self.router_type == 'per_var':
            self.router = RouterPV(long_context_window=long_context_window, c_in=c_in, d_hidden=d_model)
        else:
            self.router = Router(
                long_context_window=long_context_window,
                context_window=context_window,
                c_in=c_in,
                d_model=d_model,
            )

        # ===== Fusion Head =====
        self.head = Fusion_Head(
            concat=concat,
            individual=individual,
            c_in=c_in,
            c_out=c_out,
            nf=head_nf,
            m_nf=m_head_nf,
            t_nf=t_head_nf,
            target_window=target_window,
            head_dropout=head_dropout,
        )

    def forward(self, long: Tensor, long_mark: Tensor, short_mark: Tensor, self_mask: Optional[Tensor] = None) -> Tensor:
        """
        long: [B, L, C]
        """
        # 1）归一化
        if self.revin:
            long = self.revin_layer(long, 'norm')

        # 2）Router 计算长/短分配权重
        if self.router_type == 'per_var':
            m_weight, t_weight = self.router(long)  # [B, C], [B, C]
        else:
            m_weight, t_weight = self.router(long)  # [B], [B]

        short = long[:, -self.label_len:, :]  # 最近一段

        # 3）长程：Mamba
        long_feat = long.permute(0, 2, 1)        # [B, C, L]
        long_feat = self.long_encoder(long_feat) # [B, C, d_model, m_patch_num]

        # 4）短程：Transformer
        if self.decomposition:
            res_init, trend_init = self.decomp_module(short)
            res_init = res_init.permute(0, 2, 1)
            trend_init = trend_init.permute(0, 2, 1)
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            short_feat = (res + trend).permute(0, 2, 1)
        else:
            short_feat = self.model(short.permute(0, 2, 1))  # [B, C, d_model, patch_num]

        # 5）长短融合
        out = self.head(long_feat, short_feat, m_weight, t_weight).permute(0, 2, 1)

        # 6）反归一化
        if self.revin:
            out = self.revin_layer(out, 'denorm')

        return out


# ================= Router & Fusion Head =================

class Router(nn.Module):
    """
    原版全局 Router：每个样本一对权重 [B] / [B]
    """
    def __init__(self, long_context_window: int, context_window: int, c_in: int, d_model: int, bias: bool = True):
        super().__init__()
        self.context_window = context_window
        self.W_P = nn.Linear(c_in, d_model, bias=bias)
        self.flatten = nn.Flatten(start_dim=-2)
        self.W_w = nn.Linear(long_context_window * d_model, 2, bias=bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, long: Tensor):
        # long: [B, L, C]
        x = self.W_P(long)             # [B, L, d_model]
        x = self.flatten(x)            # [B, L*d_model]
        prob = self.softmax(self.W_w(x))  # [B, 2]
        m_weight, t_weight = prob[:, 0], prob[:, 1]
        return m_weight, t_weight


class RouterPV(nn.Module):
    """
    按变量的 Router：每个变量自己的长/短权重 [B,C]
    """
    def __init__(self, long_context_window: int, c_in: int, d_hidden: int = 128):
        super().__init__()
        self.c_in = c_in
        self.mlp = nn.Sequential(
            nn.Linear(2, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, 2),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, long: Tensor):
        # long: [B, L, C]
        mean = long.mean(dim=1)           # [B, C]
        std = long.std(dim=1) + 1e-8      # [B, C]
        feats = torch.stack([mean, std], dim=-1)  # [B, C, 2]
        out = self.mlp(feats.view(-1, 2))         # [B*C, 2]
        out = self.softmax(out).view(mean.shape[0], self.c_in, 2)  # [B, C, 2]
        m_weight = out[..., 0]            # [B, C]
        t_weight = out[..., 1]            # [B, C]
        return m_weight, t_weight


class Fusion_Head(nn.Module):
    """
    长短结果融合头：
      - 支持 concat / add 两种融合方式（和原版保持）
      - m_weight / t_weight 既可以是 [B] 也可以是 [B, C]
    """
    def __init__(
        self,
        concat: bool,
        individual: bool,
        c_in: int,
        c_out: int,
        nf: int,
        m_nf: int,
        t_nf: int,
        target_window: int,
        head_dropout: float = 0.,
    ):
        super().__init__()
        self.concat = concat
        self.individual = individual
        self.c_in = c_in
        self.c_out = c_out
        self.target_window = target_window

        if self.concat:
            if self.individual:
                self.linears = nn.ModuleList()
                self.dropouts = nn.ModuleList()
                self.flattens = nn.ModuleList()
                for _ in range(self.c_in):
                    self.flattens.append(nn.Flatten(start_dim=-2))
                    self.linears.append(nn.Linear(nf, target_window))
                    self.dropouts.append(nn.Dropout(head_dropout))
            else:
                self.flatten = nn.Flatten(start_dim=-2)
                self.linear = nn.Linear(nf, target_window)
                self.dropout = nn.Dropout(head_dropout)
        else:
            if self.individual:
                self.linears = nn.ModuleList()
                self.dropouts = nn.ModuleList()
                self.flattens = nn.ModuleList()
                self.long_to_shorts = nn.ModuleList()
                for _ in range(self.c_in):
                    self.flattens.append(nn.Flatten(start_dim=-2))
                    self.long_to_shorts.append(nn.Linear(m_nf, t_nf))
                    self.linears.append(nn.Linear(t_nf, target_window))
                    self.dropouts.append(nn.Dropout(head_dropout))
            else:
                self.flatten = nn.Flatten(start_dim=-2)
                self.long_to_short = nn.Linear(m_nf, t_nf)
                self.linear = nn.Linear(t_nf, target_window)
                self.dropout = nn.Dropout(head_dropout)

    @staticmethod
    def _expand_weight(w: Tensor, target_3d: Tensor) -> Tensor:
        # target_3d: [B, C, *]
        if w.dim() == 1:      # [B]
            return w.view(-1, 1, 1)
        elif w.dim() == 2:    # [B, C]
            return w.unsqueeze(-1)
        else:
            return w

    def forward(self, long: Tensor, short: Tensor, m_weight: Tensor, t_weight: Tensor) -> Tensor:
        if self.concat:
            if self.individual:
                outs = []
                for i in range(self.c_in):
                    l = self.flattens[i](long[:, i, :, :])
                    s = self.flattens[i](short[:, i, :, :])
                    mw = m_weight if m_weight.dim() == 1 else m_weight[:, i]
                    tw = t_weight if t_weight.dim() == 1 else t_weight[:, i]
                    ls = torch.cat((mw.view(-1, 1) * l, tw.view(-1, 1) * s), dim=1)
                    y = self.linears[i](ls)
                    y = self.dropouts[i](y)
                    outs.append(y)
                return torch.stack(outs, dim=1)
            else:
                l = self.flatten(long)   # [B, C, d_model*patch_num]
                s = self.flatten(short)
                mw = self._expand_weight(m_weight, l)
                tw = self._expand_weight(t_weight, s)
                ls = torch.cat((mw * l, tw * s), dim=2)
                y = self.linear(ls)
                y = self.dropout(y)
                return y
        else:
            if self.individual:
                outs = []
                for i in range(self.c_in):
                    l = self.flattens[i](long[:, i, :, :])
                    s = self.flattens[i](short[:, i, :, :])
                    l2s = self.long_to_shorts[i](l)
                    mw = m_weight if m_weight.dim() == 1 else m_weight[:, i]
                    tw = t_weight if t_weight.dim() == 1 else t_weight[:, i]
                    y = mw.view(-1, 1) * l2s + tw.view(-1, 1) * s
                    y = self.linears[i](y)
                    y = self.dropouts[i](y)
                    outs.append(y)
                return torch.stack(outs, dim=1)
            else:
                l = self.flatten(long)
                s = self.flatten(short)
                l2s = self.long_to_short(l)
                mw = self._expand_weight(m_weight, l2s)
                tw = self._expand_weight(t_weight, s)
                y = mw * l2s + tw * s
                y = self.linear(y)
                y = self.dropout(y)
                return y
