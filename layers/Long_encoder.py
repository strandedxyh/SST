from typing import Optional
import math
import torch
from torch import nn, Tensor
import numpy as np        # ★★★ 新增这一行 ★★★
from layers.LWT_layers import *  # positional_encoding, _MultiheadAttention, Transpose, get_activation_fn, etc.
from layers.RevIN import RevIN
import torch.nn.functional as F

class Short_encoder(nn.Module):
    """
    Short-term encoder (LWT / Transformer branch).
    We keep the original interface and only extend it with group_count for
    shared / group-shared / per-channel patch projection.
    """
    def __init__(self, c_in: int, context_window: int, target_window: int,
                 local_ws: int, patch_len: int, stride: int,
                 max_seq_len: Optional[int] = 1024,
                 n_layers: int = 3, d_model: int = 128, n_heads: int = 16,
                 d_k: Optional[int] = None, d_v: Optional[int] = None,
                 d_ff: int = 256, norm: str = 'BatchNorm',
                 attn_dropout: float = 0., dropout: float = 0.,
                 act: str = "gelu", key_padding_mask: bool = 'auto',
                 padding_var: Optional[int] = None,
                 attn_mask: Optional[Tensor] = None,
                 res_attention: bool = True, pre_norm: bool = False,
                 store_attn: bool = False,
                 pe: str = 'zeros', learn_pe: bool = True,
                 fc_dropout: float = 0., head_dropout: float = 0.,
                 padding_patch: Optional[str] = None,
                 pretrain_head: bool = False, head_type: str = 'flatten',
                 individual: bool = False, revin: bool = True,
                 affine: bool = True, subtract_last: bool = False,
                 verbose: bool = False, group_count: int = 0, **kwargs):
        super().__init__()

        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        self.group_count = int(group_count)

        self.backbone = TSTiEncoder(
            context_window=context_window,
            c_in=c_in,
            local_ws=local_ws,
            patch_num=patch_num,
            patch_len=patch_len,
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
            verbose=verbose,
            group_count=self.group_count
        )

    def forward(self, z: Tensor) -> Tensor:
        """
        Input:  z [B, C, L]
        Output: [B, C, d_model, patch_num]
        """
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        # [B, C, patch_num, patch_len]
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # [B, C, patch_len, patch_num]
        z = z.permute(0, 1, 3, 2)
        # [B, C, d_model, patch_num]
        z = self.backbone(z)
        return z


class TSTiEncoder(nn.Module):
    """
    Channel-wise encoder with flexible patch projection sharing:
    - group_count <= 1 : shared projection for all channels
    - 1 < group_count < C : group-shared projections
    - group_count >= C : per-channel projection
    """
    def __init__(self, context_window: int, c_in: int,
                 local_ws: int, patch_num: int, patch_len: int,
                 max_seq_len: int = 1024,
                 n_layers: int = 3, d_model: int = 128, n_heads: int = 16,
                 d_k: Optional[int] = None, d_v: Optional[int] = None,
                 d_ff: int = 256, norm: str = 'BatchNorm',
                 attn_dropout: float = 0., dropout: float = 0.,
                 act: str = "gelu", store_attn: bool = False,
                 key_padding_mask: bool = 'auto',
                 padding_var: Optional[int] = None,
                 attn_mask: Optional[Tensor] = None,
                 res_attention: bool = True, pre_norm: bool = False,
                 pe: str = 'zeros', learn_pe: bool = True,
                 verbose: bool = False, group_count: int = 0, **kwargs):
        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len
        self.c_in = c_in
        self.group_count = int(group_count)

        # ===== Input projection over patch_len: shared / group / per-channel =====
        W_inp_len = patch_len
        if self.group_count <= 1:
            self.mode = 'shared'
            self.W_P = nn.Linear(W_inp_len, d_model)
        elif self.group_count >= self.c_in:
            self.mode = 'per_channel'
            self.W_P_list = nn.ModuleList(
                [nn.Linear(W_inp_len, d_model) for _ in range(c_in)]
            )
        else:
            self.mode = 'group'
            self.groups = self.group_count
            self.group_size = math.ceil(self.c_in / self.groups)
            self.group_linears = nn.ModuleList(
                [nn.Linear(W_inp_len, d_model) for _ in range(self.groups)]
            )
        # 打印放这里，确保 self.mode 已经存在
        print(f"[DEBUG][ShortEncoder] c_in={self.c_in}, group_count={self.group_count}, mode={self.mode}")

        q_len = patch_num
        # positional encoding: shape broadcastable to [B, C, q_len, d_model]
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.encoder = TSTEncoder(
            q_len=q_len,
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            local_ws=local_ws,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            activation=act,
            res_attention=res_attention,
            n_layers=n_layers,
            pre_norm=pre_norm,
            store_attn=store_attn
        )

    def _proj_one(self, x_ci: Tensor, idx: int) -> Tensor:
        """
        x_ci: [B, patch_num, patch_len]
        return: [B, patch_num, d_model]
        """
        if self.mode == 'shared':
            return self.W_P(x_ci)
        elif self.mode == 'per_channel':
            return self.W_P_list[idx](x_ci)
        else:
            g = min(idx // self.group_size, self.groups - 1)
            return self.group_linears[g](x_ci)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [B, C, patch_len, patch_num]
        return: [B, C, d_model, patch_num]
        """
        B, C, P, N = x.shape
        assert C == self.c_in and P == self.patch_len and N == self.patch_num
        # [B, C, patch_num, patch_len]
        x = x.permute(0, 1, 3, 2)

        proj_list = []
        for i in range(C):
            xi = x[:, i, :, :]     # [B, patch_num, patch_len]
            ui = self._proj_one(xi, i)   # [B, patch_num, d_model]
            proj_list.append(ui)
        # [B, C, patch_num, d_model]
        u = torch.stack(proj_list, dim=1)

        # add positional encoding (broadcast) and dropout
        u = self.dropout(u + self.W_pos)

        # merge batch & channel for encoder: [B*C, patch_num, d_model]
        z = u.reshape(B * C, N, -1)
        z = self.encoder(z)       # [B*C, patch_num, d_model]
        # reshape back: [B, C, patch_num, d_model] -> [B, C, d_model, patch_num]
        z = z.reshape(B, C, N, -1).permute(0, 1, 3, 2)
        return z


class TSTEncoder(nn.Module):
    """
    Stack of TSTEncoderLayer. We keep the original res_attention interface:
    - if res_attention == True: each layer returns (output, scores)
    - else: each layer returns output only
    """
    def __init__(self, q_len: int, d_model: int, n_heads: int,
                 d_k: Optional[int] = None, d_v: Optional[int] = None,
                 d_ff: Optional[int] = None, local_ws: Optional[int] = None,
                 norm: str = 'BatchNorm', attn_dropout: float = 0.,
                 dropout: float = 0., activation: str = 'gelu',
                 res_attention: bool = False, n_layers: int = 1,
                 pre_norm: bool = False, store_attn: bool = False):
        super().__init__()

        self.layers = nn.ModuleList([
            TSTEncoderLayer(
                q_len=q_len,
                d_model=d_model,
                n_heads=n_heads,
                d_k=d_k,
                d_v=d_v,
                d_ff=d_ff,
                local_ws=local_ws,
                norm=norm,
                attn_dropout=attn_dropout,
                dropout=dropout,
                activation=activation,
                res_attention=res_attention,
                pre_norm=pre_norm,
                store_attn=store_attn
            )
            for _ in range(n_layers)
        ])
        self.res_attention = res_attention

    def forward(self, src: Tensor,
                key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) -> Tensor:
        output = src
        if self.res_attention:
            scores = None
            for mod in self.layers:
                # TSTEncoderLayer will return (output, scores)
                output, scores = mod(
                    output,
                    prev=scores,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask
                )
            return output
        else:
            for mod in self.layers:
                output = mod(
                    output,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask
                )
            return output


class TSTEncoderLayer(nn.Module):
    """
    Single Transformer encoder layer used in TST.
    res_attention:
      - if False: behaves like standard Transformer, returns src
      - if True: also propagates 'scores' for residual attention routing,
                 returns (src, scores)
    """
    def __init__(self, q_len: int, d_model: int, n_heads: int,
                 d_k: Optional[int] = None, d_v: Optional[int] = None,
                 d_ff: int = 256, local_ws: Optional[int] = None,
                 store_attn: bool = False, norm: str = 'BatchNorm',
                 attn_dropout: float = 0., dropout: float = 0.,
                 bias: bool = True, activation: str = "gelu",
                 res_attention: bool = False, pre_norm: bool = False):
        super().__init__()

        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.res_attention = res_attention

        self.self_attn = _MultiheadAttention(
            d_model, n_heads, d_k, d_v,
            local_ws=local_ws,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            res_attention=res_attention
        )

        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(
                Transpose(1, 2),
                nn.BatchNorm1d(d_model),
                Transpose(1, 2)
            )
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=bias)
        )

        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(
                Transpose(1, 2),
                nn.BatchNorm1d(d_model),
                Transpose(1, 2)
            )
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src: Tensor,
                prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None):
        """
        src: [B*, L, d_model]
        if res_attention:
            returns (src, scores)
        else:
            returns src
        """
        if self.pre_norm:
            src = self.norm_attn(src)

        if self.res_attention:
            # _MultiheadAttention is expected to return (src2, attn, scores)
            src2, attn, scores = self.self_attn(
                src, src, src,
                prev,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask
            )
        else:
            # expected to return (src2, attn)
            src2, attn = self.self_attn(
                src, src, src,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask
            )
            scores = None

        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)

        if self.pre_norm:
            src = self.norm_ffn(src)
        src2 = self.ff(src)
        src = src + self.dropout_ffn(src2)
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
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
