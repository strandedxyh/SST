# layers/Short_encoder.py —— 完整可替换版
from typing import Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np

from layers.LWT_layers import *
from layers.RevIN import RevIN

# ------------------------ Short_encoder ------------------------
class Short_encoder(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, local_ws:int, patch_len:int, stride:int,
                 max_seq_len:Optional[int]=1024, n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout:float=0., padding_patch=None,
                 pretrain_head:bool=False, head_type='flatten', individual=False, revin=True, affine=True, subtract_last=False,
                 verbose:bool=False, **kwargs):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        # CI/GSP 开关（未传参则默认开启 CI，仅等价原版也OK）
        self.ci_proj = kwargs.get('ci_proj', True)
        self.group_count = kwargs.get('group_count', 0)

        self.backbone = TSTiEncoder(context_window, c_in, local_ws=local_ws, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                    n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                    attn_dropout=attn_dropout, dropout=dropout, act=act,
                                    key_padding_mask=key_padding_mask, padding_var=padding_var,
                                    attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                    pe=pe, learn_pe=learn_pe, verbose=verbose,
                                    ci_proj=self.ci_proj, group_count=self.group_count)

    def forward(self, z):  # [B, C, L]
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)   # [B, C, patch_num, patch_len]
        z = z.permute(0, 1, 3, 2)                                           # [B, C, patch_len, patch_num]
        z = self.backbone(z)                                                # [B, C, d_model, patch_num]
        return z


# ------------------------ TSTiEncoder ------------------------
class TSTiEncoder(nn.Module):  # i: channel-independent
    def __init__(self, context_window, c_in, local_ws, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, ci_proj=True, group_count=0, **kwargs):
        super().__init__()
        self.patch_num = patch_num
        self.patch_len = patch_len
        self.c_in = c_in
        self.ci_proj = ci_proj
        self.group_count = group_count

        # Input encoding（CI/GSP）
        W_inp_len = patch_len
        if self.group_count and self.group_count > 0:
            self.groups = self.group_count
            self.group_linears = nn.ModuleList([nn.Linear(W_inp_len, d_model) for _ in range(self.groups)])
        elif self.ci_proj:
            self.W_P_list = nn.ModuleList([nn.Linear(W_inp_len, d_model) for _ in range(c_in)])
        else:
            self.W_P = nn.Linear(W_inp_len, d_model)

        q_len = patch_num

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, local_ws=local_ws, norm=norm,
                                  attn_dropout=attn_dropout, dropout=dropout, pre_norm=pre_norm,
                                  activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)

    def _proj_one(self, x_ci: Tensor, idx: int) -> Tensor:
        # x_ci: [B, patch_num, patch_len] -> [B, patch_num, d_model]
        if self.group_count and self.group_count > 0:
            # 均匀分组
            group_size = (self.c_in + self.group_count - 1) // self.group_count
            g = min(idx // group_size, self.group_count - 1)
            return self.group_linears[g](x_ci)
        elif self.ci_proj:
            return self.W_P_list[idx](x_ci)
        else:
            return self.W_P(x_ci)

    def forward(self, x) -> Tensor:                      # x: [B, C, patch_len, patch_num]
        B, C, P, N = x.shape
        assert C == self.c_in and P == self.patch_len and N == self.patch_num
        x = x.permute(0, 1, 3, 2)                        # [B, C, patch_num, patch_len]

        proj_list = []
        for i in range(C):
            xi = x[:, i, :, :]                           # [B, patch_num, patch_len]
            ui = self._proj_one(xi, i)                   # [B, patch_num, d_model]
            proj_list.append(ui)
        u = torch.stack(proj_list, dim=1)                # [B, C, patch_num, d_model]

        u = self.dropout(u + self.W_pos)                 # 位置编码 + dropout
        z = u.reshape(B*C, N, -1)                        # [B*C, patch_num, d_model]
        z = self.encoder(z)                              # [B*C, patch_num, d_model]
        z = z.reshape(B, C, N, -1).permute(0, 1, 3, 2)   # [B, C, d_model, patch_num]
        return z


# ------------------------ TSTEncoder / TSTEncoderLayer（原版保持一致） ------------------------
class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, local_ws=None,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, local_ws=local_ws, norm=norm,
                            attn_dropout=attn_dropout, dropout=dropout,
                            activation=activation, res_attention=res_attention,
                            pre_norm=pre_norm, store_attn=store_attn) for _ in range(n_layers)
        ])
        self.res_attention = res_attention

    def forward(self, src: Tensor, key_padding_mask: Optional[Tensor]=None, attn_mask: Optional[Tensor]=None):
        output = src
        if self.res_attention:
            scores = None
            for mod in self.layers:
                output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers:
                output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output


class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, local_ws=None, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, local_ws=local_ws,
                                             attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src: Tensor, prev: Optional[Tensor]=None, key_padding_mask: Optional[Tensor]=None, attn_mask: Optional[Tensor]=None) -> Tensor:
        if self.pre_norm:
            src = self.norm_attn(src)
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)

        if self.pre_norm:
            src = self.norm_ffn(src)
        src2 = self.ff(src)
        src  = src + self.dropout_ffn(src2)
        if not self.pre_norm:
            src = self.norm_ffn(src)
        return src





class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, local_ws=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, local_ws=local_ws, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)   #output: [bs , q_len , d_model]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, local_ws=None, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa
        self.local_ws = local_ws

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev
        # Add Local Attntion
        local_mask = self.get_local_mask(q.shape[2], self.local_ws).to(q.device)
        attn_scores.masked_fill_(local_mask, -np.inf)

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)  #[bs, 1, 1, q_len]

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


    def get_local_mask(self, attn_size: int, window_size: int, p=1.0):
        # Check if window size is odd
        assert window_size % 2 == 1, "The window size is assumed to be odd (counts self-attention + 2 wings)"
        h_win_size = window_size // 2 + 1

        # Generate grid for the 1D pattern
        coords = torch.arange(attn_size)
        grid = torch.meshgrid(coords, indexing='ij')
        grid_flat = grid[0].flatten().float()

        # Calculate distances using the cdist function
        d = torch.cdist(grid_flat.unsqueeze(1), grid_flat.unsqueeze(1), p=p)

        # Generate the local pattern mask based on the half window size
        mask = d < h_win_size

        return mask
