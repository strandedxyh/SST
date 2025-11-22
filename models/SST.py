# models/SST.py  —— 完整可替换版
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
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 norm:str='BatchNorm', attn_dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True,
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True,
                 pretrain_head:bool=False, head_type='flatten', verbose:bool=False, **kwargs):
        super().__init__()

        # ----- load configs (与原版保持一致) -----
        long_context_window = configs.seq_len
        m_layers = configs.m_layers
        d_state = configs.d_state
        d_conv  = configs.d_conv
        m_patch_len = configs.m_patch_len
        m_stride    = configs.m_stride

        c_in = configs.enc_in
        context_window = configs.label_len
        self.label_len = configs.label_len
        target_window = configs.pred_len
        local_ws = configs.local_ws

        n_layers = configs.e_layers
        n_heads  = configs.n_heads
        d_model  = configs.d_model
        d_ff     = configs.d_ff
        dropout  = configs.dropout
        fc_dropout   = configs.fc_dropout
        head_dropout = configs.head_dropout
        individual   = configs.individual

        patch_len    = configs.patch_len
        stride       = configs.stride
        padding_patch= configs.padding_patch

        revin         = configs.revin
        affine        = configs.affine
        subtract_last = configs.subtract_last

        decomposition = configs.decomposition
        kernel_size   = configs.kernel_size

        concat = configs.concat

        # ===== 新增可选开关（不传就默认原行为） =====
        self.router_type = getattr(configs, 'router_type', 'global')   # 'global' or 'per_var'
        # 通道独立投影 / 分组共享投影由 encoder 内部读取 configs.*（见 layers 里的实现）

        # ----- patch 数量（与原版一致） -----
        m_patch_num = int((long_context_window - m_patch_len)/m_stride + 1)
        if padding_patch == 'end': m_patch_num += 1
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end': patch_num += 1

        # ----- 头部维度 -----
        c_out = configs.c_out
        m_head_nf = d_model * m_patch_num
        t_head_nf = d_model * patch_num
        head_nf   = d_model * (m_patch_num + patch_num)

        # ----- RevIN -----
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # ----- Long encoder (Mamba) -----
        self.long_encoder = Long_encoder(
            c_in=c_in, long_context_window=long_context_window,
            target_window=target_window, m_patch_len=m_patch_len, m_stride=m_stride,
            m_layers=m_layers, d_model=d_model, d_ff=d_ff, norm=norm, dropout=dropout, act=act,
            pre_norm=pre_norm, d_state=d_state, d_conv=d_conv,
            fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch=padding_patch,
            pretrain_head=pretrain_head, head_type=head_type, individual=individual,
            revin=revin, affine=affine, subtract_last=subtract_last, verbose=verbose, **kwargs
        )

        # ----- Short encoder (Transformer) -----
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = Short_encoder(
                c_in=c_in, context_window=context_window, target_window=target_window,
                local_ws=local_ws, patch_len=patch_len, stride=stride, max_seq_len=max_seq_len,
                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                norm=norm, attn_dropout=attn_dropout, dropout=dropout, act=act,
                key_padding_mask=key_padding_mask, padding_var=padding_var, attn_mask=attn_mask,
                res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout,
                padding_patch=padding_patch, pretrain_head=pretrain_head, head_type=head_type,
                individual=individual, revin=revin, affine=affine, subtract_last=subtract_last,
                verbose=verbose, **kwargs
            )
            self.model_res = Short_encoder(
                c_in=c_in, context_window=context_window, target_window=target_window,
                local_ws=local_ws, patch_len=patch_len, stride=stride, max_seq_len=max_seq_len,
                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                norm=norm, attn_dropout=attn_dropout, dropout=dropout, act=act,
                key_padding_mask=key_padding_mask, padding_var=padding_var, attn_mask=attn_mask,
                res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout,
                padding_patch=padding_patch, pretrain_head=pretrain_head, head_type=head_type,
                individual=individual, revin=revin, affine=affine, subtract_last=subtract_last,
                verbose=verbose, **kwargs
            )
        else:
            self.model = Short_encoder(
                c_in=c_in, context_window=context_window, target_window=target_window,
                local_ws=local_ws, patch_len=patch_len, stride=stride, max_seq_len=max_seq_len,
                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                norm=norm, attn_dropout=attn_dropout, dropout=dropout, act=act,
                key_padding_mask=key_padding_mask, padding_var=padding_var, attn_mask=attn_mask,
                res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout,
                padding_patch=padding_patch, pretrain_head=pretrain_head, head_type=head_type,
                individual=individual, revin=revin, affine=affine, subtract_last=subtract_last,
                verbose=verbose, **kwargs
            )

        # ----- Router（新增按变量版本开关） -----
        if self.router_type == 'per_var':
            self.router = RouterPV(long_context_window=long_context_window, c_in=c_in, d_hidden=d_model)
        else:
            self.router = Router(long_context_window=long_context_window, context_window=context_window,
                                 c_in=c_in, d_model=d_model)

        # ----- Fusion Head（保持类名与调用不变） -----
        self.head = Fusion_Head(concat=concat, individual=individual,
                                c_in=c_in, c_out=c_out, nf=head_nf, m_nf=m_head_nf, t_nf=t_head_nf,
                                target_window=target_window, head_dropout=head_dropout)

    def forward(self, long, long_mark, short_mark, self_mask=None):
        # norm
        if self.revin:
            long = self.revin_layer(long, 'norm')

        # router
        if self.router_type == 'per_var':
            m_weight, t_weight = self.router(long)        # [B,C], [B,C]
        else:
            m_weight, t_weight = self.router(long)        # [B], [B]（与原版一致）  :contentReference[oaicite:8]{index=8}

        short = long[:, -self.label_len:, :]

        # mamba（长）
        long = long.permute(0, 2, 1)              # [B, C, T]
        long = self.long_encoder(long)            # [B, C, d_model, m_patch_num]

        # transformer（短）
        if self.decomposition:
            res_init, trend_init = self.decomp_module(short)
            res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
            res   = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            short = (res + trend).permute(0, 2, 1)
        else:
            short = self.model(short.permute(0, 2, 1))   # [B, C, d_model, patch_num]

        # 融合（权重形状自动兼容）
        long_short = self.head(long, short, m_weight, t_weight).permute(0, 2, 1)

        # denorm
        if self.revin:
            long_short = self.revin_layer(long_short, 'denorm')

        return long_short


class Router(nn.Module):
    """
    长短路由（原版，全局权重）  —— 保持不变
    """
    def __init__(self, long_context_window, context_window, c_in, d_model, bias=True):
        super().__init__()
        self.context_window = context_window
        self.W_P = nn.Linear(c_in, d_model, bias=bias)
        self.flatten = nn.Flatten(start_dim=-2)
        self.W_w = nn.Linear(long_context_window * d_model, 2, bias=bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, long):              # long: [B, T, C]
        x = self.W_P(long)
        x = self.flatten(x)
        prob = self.softmax(self.W_w(x))
        m_weight, t_weight = prob[:, 0], prob[:, 1]
        return m_weight, t_weight


class RouterPV(nn.Module):
    """
    按变量路由（Per-Variable Router）
    输入: long [B, T, C]
    输出: m_weight, t_weight 形状都是 [B, C]
    """
    def __init__(self, long_context_window, c_in, d_hidden=128):
        super().__init__()
        self.c_in = c_in
        # 用每个变量在时间维上的统计量做轻量特征（均值、标准差），再走 MLP
        self.mlp = nn.Sequential(
            nn.Linear(2, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, 2)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, long):              # [B, T, C]
        mean = long.mean(dim=1)           # [B, C]
        std  = long.std(dim=1) + 1e-6     # [B, C]
        feats = torch.stack([mean, std], dim=-1)       # [B, C, 2]
        out = self.mlp(feats.view(-1, 2))              # [B*C, 2]
        out = self.softmax(out).view(mean.shape[0], self.c_in, 2)   # [B, C, 2]
        m_weight = out[..., 0]            # [B, C]
        t_weight = out[..., 1]            # [B, C]
        return m_weight, t_weight


class Fusion_Head(nn.Module):
    """
    长短结果融合头（保持类名与参数不变）
    新增：支持 m_weight/t_weight 为 [B] 或 [B, C] 两种形状
    """
    def __init__(self, concat, individual, c_in, c_out, nf, m_nf, t_nf, target_window, head_dropout=0):
        super().__init__()
        self.concat = concat
        self.individual = individual
        self.c_in = c_in
        self.c_out = c_out
        self.target_window = target_window

        if self.concat:
            if self.individual:
                self.linears  = nn.ModuleList()
                self.dropouts = nn.ModuleList()
                self.flattens = nn.ModuleList()
                for _ in range(self.c_in):
                    self.flattens.append(nn.Flatten(start_dim=-2))
                    self.linears.append(nn.Linear(nf, target_window))
                    self.dropouts.append(nn.Dropout(head_dropout))
            else:
                self.flatten = nn.Flatten(start_dim=-2)
                self.linear  = nn.Linear(nf, target_window)
                self.dropout = nn.Dropout(head_dropout)
        else:
            if self.individual:
                self.linears  = nn.ModuleList()
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
                self.linear  = nn.Linear(t_nf, target_window)
                self.dropout = nn.Dropout(head_dropout)

    @staticmethod
    def _expand_weight(w, target_3d: torch.Tensor):
        # target_3d: [B, C, *]
        if w.dim() == 1:    # [B]
            return w.view(-1, 1, 1)
        elif w.dim() == 2:  # [B, C]
            return w.unsqueeze(-1)
        else:
            return w

    def forward(self, long, short, m_weight, t_weight):
        if self.concat:
            if self.individual:
                outs = []
                for i in range(self.c_in):
                    l = self.flattens[i](long[:, i, :, :])   # [B, d_model*patch_num]
                    s = self.flattens[i](short[:, i, :, :])
                    # [B] or [B,C] -> 取通道 i
                    mw = m_weight if m_weight.dim()==1 else m_weight[:, i]
                    tw = t_weight if t_weight.dim()==1 else t_weight[:, i]
                    ls = torch.cat((mw.view(-1,1)*l, tw.view(-1,1)*s), dim=1)
                    y = self.linears[i](ls)
                    y = self.dropouts[i](y)
                    outs.append(y)
                return torch.stack(outs, dim=1)             # [B, C, T_out]
            else:
                l = self.flatten(long)                       # [B, C, d_model*patch_num]
                s = self.flatten(short)
                mw = self._expand_weight(m_weight, l)        # [B,1,1] or [B,C,1]
                tw = self._expand_weight(t_weight, s)
                ls = torch.cat((mw*l, tw*s), dim=2)
                y  = self.linear(ls)
                y  = self.dropout(y)
                return y
        else:
            if self.individual:
                outs = []
                for i in range(self.c_in):
                    l = self.flattens[i](long[:, i, :, :])   # [B, d_model*patch_num]
                    s = self.flattens[i](short[:, i, :, :])
                    l2s = self.long_to_shorts[i](l)
                    mw = m_weight if m_weight.dim()==1 else m_weight[:, i]
                    tw = t_weight if t_weight.dim()==1 else t_weight[:, i]
                    y = mw.view(-1,1)*l2s + tw.view(-1,1)*s
                    y = self.linears[i](y)
                    y = self.dropouts[i](y)
                    outs.append(y)
                return torch.stack(outs, dim=1)
            else:
                l = self.flatten(long)                       # [B, C, d_model*patch_num]
                s = self.flatten(short)                      # [B, C, d_model*patch_num] or [B, C, t_nf]
                l2s = self.long_to_short(l)                  # [B, C, t_nf]
                mw = self._expand_weight(m_weight, l2s)
                tw = self._expand_weight(t_weight, s)
                y = mw*l2s + tw*s
                y = self.linear(y)
                y = self.dropout(y)
                return y
