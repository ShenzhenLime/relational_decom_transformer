import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from math import sqrt
import os

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

"""
Full Attention
"""
class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=3, scale=None, attention_dropout=0.1, T=1, activation='softmax',
                 output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.activation = activation
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.T = T

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * scale

        if self.activation == 'softmax':
            if self.mask_flag:
                if attn_mask is None:
                    attn_mask = TriangularCausalMask(B, L, device=queries.device)

                scores.masked_fill_(attn_mask.mask, -np.inf)

            A = self.dropout(torch.softmax(scores / self.T, dim=-1))
            V = torch.einsum("bhls,bshd->blhd", A, values)

        elif self.activation == 'linear':
            V = torch.einsum("bhls,bshd->blhd", scores, values)

        elif self.activation == 'linear_norm':
            mins = scores.min(dim=-1)[0].unsqueeze(-1).expand(-1, -1, -1, scores.shape[3])
            scores = scores - mins + 1e-8

            if self.mask_flag:
                if attn_mask is None:
                    attn_mask = TriangularCausalMask(B, L, device=queries.device)
                scores.masked_fill_(attn_mask.mask, 0)

            sums = scores.sum(dim=-1).unsqueeze(-1).expand(-1, -1, -1, scores.shape[3])
            scores /= sums
            V = torch.einsum("bhls,bshd->blhd", scores, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

"""
Fourier Attention
"""
class FourierAttention(nn.Module):
    def __init__(self, T=1, activation='softmax', output_attention=False):
        super(FourierAttention, self).__init__()
        print(' fourier enhanced cross attention used!')
        """
        1D Fourier Cross Attention layer. It does FFT, linear transform, attention mechanism and Inverse FFT.    
        """
        self.activation = activation
        self.output_attention = output_attention
        self.T = T

    def forward(self, q, k, v, mask):
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        _, S, H, E = k.shape
        xq = q.permute(0, 2, 3, 1)  # size = [B, H, E, L]
        xk = k.permute(0, 2, 3, 1)
        xv = v.permute(0, 2, 3, 1)

        xq_ft_ = torch.fft.rfft(xq, dim=-1, norm='ortho')
        xk_ft_ = torch.fft.rfft(xk, dim=-1, norm='ortho')
        xv_ft_ = torch.fft.rfft(xv, dim=-1, norm='ortho')

        xqk_ft = torch.einsum("bhex,bhey->bhxy", xq_ft_, torch.conj(xk_ft_)) / sqrt(E)

        if self.activation == 'softmax':
            xqk_ft = torch.softmax(xqk_ft.abs() / self.T, dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
            xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xv_ft_)

        elif self.activation == 'linear':
            xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xv_ft_)

        elif self.activation == 'linear_norm':
            mins_real = xqk_ft.real.min(dim=-1)[0].unsqueeze(-1).expand(-1, -1, -1, xqk_ft.shape[3])
            xqk_ft_real = xqk_ft.real - mins_real
            sums_real = xqk_ft_real.sum(dim=-1).unsqueeze(-1).expand(-1, -1, -1, xqk_ft.shape[3])
            xqk_ft_real /= sums_real

            mins_imag = xqk_ft.imag.min(dim=-1)[0].unsqueeze(-1).expand(-1, -1, -1, xqk_ft.shape[3])
            xqk_ft_imag = xqk_ft.imag - mins_imag
            sums_imag = xqk_ft_imag.sum(dim=-1).unsqueeze(-1).expand(-1, -1, -1, xqk_ft.shape[3])
            xqk_ft_imag /= sums_imag

            xqkv_ft_real = torch.einsum("bhxy,bhey->bhex", xqk_ft_real, xv_ft_.real)
            xqkv_ft_imag = torch.einsum("bhxy,bhey->bhex", xqk_ft_imag, xv_ft_.imag)
            xqkv_ft = torch.complex(xqkv_ft_real, xqkv_ft_imag)

        elif self.activation == 'linear_norm_abs':
            xqk_ft = xqk_ft.abs() / xqk_ft.abs().sum(dim=-1).unsqueeze(-1).expand(-1, -1, -1, xqk_ft.shape[3])
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
            xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xv_ft_)

        elif self.activation == 'linear_norm_real':
            mins_real = xqk_ft.real.min(dim=-1)[0].unsqueeze(-1).expand(-1, -1, -1, xqk_ft.shape[3])
            xqk_ft_real = xqk_ft.real - mins_real
            sums_real = xqk_ft_real.sum(dim=-1).unsqueeze(-1).expand(-1, -1, -1, xqk_ft.shape[3])
            xqk_ft_real /= sums_real

            xqk_ft = torch.complex(xqk_ft_real, torch.zeros_like(xqk_ft_real))
            xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xv_ft_)

        out = torch.fft.irfft(xqkv_ft, n=L, dim=-1, norm='ortho').permute(0, 3, 1, 2)

        if self.output_attention == False:
            return (out, None)
        else:
            return (out, (xqk_ft_real, xqk_ft_imag))


        
class WaveletAttention(nn.Module):

    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, c=64,
                 k=8, ich=512,
                 L=3,
                 base='legendre',
                 initializer=None, T=1, activation='softmax', output_attention=False,
                 **kwargs):
        super(WaveletAttention, self).__init__()
        print('base', base)

        self.c = c
        self.k = k
        self.L = L
        self.T = T
        self.activation = activation
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0 @ PHI0
        G0r = G0 @ PHI0
        H1r = H1 @ PHI1
        G1r = G1 @ PHI1

        H0r[np.abs(H0r) < 1e-8] = 0
        H1r[np.abs(H1r) < 1e-8] = 0
        G0r[np.abs(G0r) < 1e-8] = 0
        G1r[np.abs(G1r) < 1e-8] = 0
        self.max_item = 3

        self.register_buffer('ec_s', torch.Tensor(
            np.concatenate((H0.T, H1.T), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(
            np.concatenate((G0.T, G1.T), axis=0)))

        self.register_buffer('rc_e', torch.Tensor(
            np.concatenate((H0r, G0r), axis=0)))
        self.register_buffer('rc_o', torch.Tensor(
            np.concatenate((H1r, G1r), axis=0)))

        self.Lk = nn.Linear(ich, c * k)
        self.Lq = nn.Linear(ich, c * k)
        self.Lv = nn.Linear(ich, c * k)
        self.out = nn.Linear(c * k, ich)

        self.output_attention = output_attention

    def forward(self, q, k, v, mask=None):
        B, N, H, E = q.shape  # (B, N, H, E) torch.Size([3, 768, 8, 2])
        _, S, _, _ = k.shape  # (B, S, H, E) torch.Size([3, 96, 8, 2])

        q = q.view(q.shape[0], q.shape[1], -1)  # (B, N, H*E)
        k = k.view(k.shape[0], k.shape[1], -1)
        v = v.view(v.shape[0], v.shape[1], -1)
        q = self.Lq(q)
        q = q.view(q.shape[0], q.shape[1], self.c, self.k)  # (B, N, E, H)
        k = self.Lk(k)
        k = k.view(k.shape[0], k.shape[1], self.c, self.k)
        v = self.Lv(v)
        v = v.view(v.shape[0], v.shape[1], self.c, self.k)

        if N > S:
            zeros = torch.zeros_like(q[:, :(N - S), :]).float()
            v = torch.cat([v, zeros], dim=1)
            k = torch.cat([k, zeros], dim=1)
        else:
            v = v[:, :N, :, :]
            k = k[:, :N, :, :]
        ns = math.floor(np.log2(N))
        nl = pow(2, math.ceil(np.log2(N)))
        extra_q = q[:, 0:nl - N, :, :]
        extra_k = k[:, 0:nl - N, :, :]
        extra_v = v[:, 0:nl - N, :, :]
        q = torch.cat([q, extra_q], 1)
        k = torch.cat([k, extra_k], 1)
        v = torch.cat([v, extra_v], 1)

        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])

        attn_d_list, attn_s_list = [], []

        for i in range(ns - self.L):
            dq, q = self.wavelet_transform(q)
            dk, k = self.wavelet_transform(k)
            dv, v = self.wavelet_transform(v)  # (B, N, E, H)

            scores_d = torch.einsum("bxeh,byeh->bhxy", dq, dk) / math.sqrt(E)

            if self.activation == 'softmax':
                attn_d = F.softmax(scores_d / self.T, dim=-1)  # (B,H,q,k)
            elif self.activation == 'linear':
                attn_d = scores_d  # (B,H,q,k)
            elif self.activation == 'linear_norm':
                attn_d = scores_d  # (B,H,q,k)
                mins = attn_d.min(dim=-1).unsqueeze(-1).expand(-1, -1, -1, attn_d.shape[3])
                attn_d -= mins
                sums = attn_d.sum(dim=-1).unsqueeze(-1).expand(-1, -1, -1, attn_d.shape[3])
                attn_d /= sums
            Ud += [torch.einsum("bhxy,byeh->bxeh", attn_d, dv)]
            attn_d_list.append(attn_d)

            scores_s = torch.einsum("bxeh,byeh->bhxy", q, k) / math.sqrt(E)

            if self.activation == 'softmax':
                attn_s = F.softmax(scores_s / self.T, dim=-1)  # (B,H,q,k)
            elif self.activation == 'linear':
                attn_s = scores_s  # (B,H,q,k)
            elif self.activation == 'linear_norm':
                attn_s = scores_s  # (B,H,q,k)
                mins = attn_s.min(dim=-1).unsqueeze(-1).expand(-1, -1, -1, attn_s.shape[3])
                attn_s -= mins
                sums = attn_s.sum(dim=-1).unsqueeze(-1).expand(-1, -1, -1, attn_s.shape[3])
                attn_s /= sums
            Us += [torch.einsum("bhxy,byeh->bxeh", attn_s, v)]
            attn_s_list.append(attn_s)

        # reconstruct
        for i in range(ns - 1 - self.L, -1, -1):
            v = v + Us[i]
            v = torch.cat((v, Ud[i]), -1)
            v = self.evenOdd(v)
        v = self.out(v[:, :N, :, :].contiguous().view(B, N, -1))
        if self.output_attention == False:
            return (v.contiguous(), None)
        else:
            return (v.contiguous(), (attn_s_list, attn_d_list))

    def wavelet_transform(self, x):
        xa = torch.cat([x[:, ::2, :, :],
                        x[:, 1::2, :, :],
                        ], -1)
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s

    def evenOdd(self, x):
        B, N, c, ich = x.shape  # (B, N, c, k)
        assert ich == 2 * self.k
        x_e = torch.matmul(x, self.rc_e)
        x_o = torch.matmul(x, self.rc_o)

        x = torch.zeros(B, N * 2, c, self.k,
                        device=x.device)
        x[..., ::2, :, :] = x_e
        x[..., 1::2, :, :] = x_o
        return x


