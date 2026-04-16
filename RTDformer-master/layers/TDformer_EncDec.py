import os
import numpy as np
import pandas as pd
import math
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """
    完成子层1：多头注意力机制（Multi-Head Attention），结果为MultiHead(Q, K, V )
    """
    def __init__(self, attention, d_model, n_heads=8, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        ## 最终返回形状 (Shape)：(B, L, d_{model}) 是一个包含相关关系的输入，需要进一步传入FeedForward 层
        ## attn 是一个包含注意力权重的张量，形状为 (B, H, L, S)，其中 B 是批次大小，H 是注意力头的数量，L 是查询序列的长度，S 是键/值序列的长度。这个张量表示了每个查询位置与每个键位置之间的注意力权重，可以用于分析模型关注输入序列的哪些部分。
        return self.out_projection(out), attn


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        ## 使用最大池化层（Max Pooling）来减少序列长度，kernel_size=3 表示池化窗口的大小，stride=2 表示池化窗口每次移动的步长，padding=1 表示在输入序列的两端添加一个元素的零填充，以保持输出序列的长度与输入序列的一半相同。
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=2048, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        ## 升维与降维
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        ## 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        ## 第一阶段：注意力机制（信息的“横向”交流）
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)

        ## 第二阶段：前馈神经网络（信息的“纵向”变换）
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

    
# class SpatioEncoder(nn.Module):
#     def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
#         super(Encoder, self).__init__()
#         self.attn_layers = nn.ModuleList(attn_layers)
#         self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
#         self.norm = norm_layer

#     def forward(self, x, attn_mask=None):
#         # x [B, L, D]
#         attns = []
#         if self.conv_layers is not None:
#             for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
#                 x, attn = attn_layer(x, attn_mask=attn_mask)
#                 x = conv_layer(x)
#                 attns.append(attn)
#             x, attn = self.attn_layers[-1](x)
#             attns.append(attn)
#         else:
#             for attn_layer in self.attn_layers:
#                 x, attn = attn_layer(x, attn_mask=attn_mask)
#                 attns.append(attn)

#         if self.norm is not None:
#             x = self.norm(x)

#         return x, attns

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=2048,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x_o, attn1 = self.self_attention(x, x, x, attn_mask=x_mask)
        x = x + self.dropout(x_o)
        x = self.norm1(x)

        x_o, attn2 = self.cross_attention(x, cross, cross, attn_mask=cross_mask)
        x = x + self.dropout(x_o)

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y), (attn1, attn2)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        attns = []
        for layer in self.layers:
            x, attn = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)

        return x, attns


class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """

    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        # [Batch, Length, Channel] -0维 2维不变，1维进行首尾值padding
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean = torch.cat(moving_mean, dim=-1)
        moving_mean = torch.sum(moving_mean * nn.Softmax(-1)(self.layer(x.unsqueeze(-1))), dim=-1)
        res = x - moving_mean
        return res, moving_mean
    
    
class SeasonalDecomp(nn.Module):
    """
    Further decompose the residual into seasonal and random components using FFT
    """
    def __init__(self):
        super(SeasonalDecomp, self).__init__()
        self.window_size=25
        self.seasonal_weight = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.dynamic_threshold = nn.Parameter(torch.tensor(1.0), requires_grad=True)  # 可学习的动态阈值

    def forward(self, x):
        # Applying FFT
        fft_result = torch.fft.rfft(x, dim=1, norm='ortho')
        
        # Define a mask or filter in frequency domain that captures seasonal peaks
        seasonal_mask = self.define_seasonal_mask(fft_result)
        
        # Apply the seasonal mask to isolate seasonal frequencies
        seasonal_freqs = fft_result * seasonal_mask
        
        # Convert the isolated seasonal frequencies back to the time domain
        seasonal_component = torch.fft.irfft(seasonal_freqs, dim=1, norm='ortho', n=x.shape[1])
        
        # Calculate noise component as the original input minus the seasonal component
        noise_component = x - seasonal_component
        
        return seasonal_component, noise_component
    
    def define_seasonal_mask(self, fft_result):
        # Implement your logic to create a mask that isolates seasonal frequencies
        # Example of a simple threshold-based mask
        magnitude = torch.abs(fft_result)  #取振幅
        local_threshold = self.local_threshold(magnitude, self.window_size)
        
        threshold = local_threshold * self.dynamic_threshold
        ## 通过比较振幅与动态阈值，创建一个布尔掩码，标记出振幅比较大的频段，这些频段对应于季节性成分
        mask = magnitude > threshold
        
        return mask.float()  # Convert boolean mask to float for multiplication
    
    def local_threshold(self, magnitude, window_size):
        # Apply padding to handle borders
         #(1,49,1)
        padded = torch.nn.functional.pad(magnitude, (window_size//2, window_size//2), mode='replicate')
        
        # Use avg_pool1d to compute sliding window mean
        local_means = torch.nn.functional.avg_pool1d(padded, kernel_size=window_size, stride=1, padding=0)
        # Scale the local mean to set a dynamic threshold
        return local_means * 0.8

        
class series_decomp_res(nn.Module):
    """
    Series decomposition block, modified to return trend, seasonal, and residual components.
    """
    def __init__(self, kernel_size):
        super(series_decomp_res, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)
        self.seasonal_decomp_module = SeasonalDecomp()

    def forward(self, x):
        moving_mean = self.moving_avg(x)  # Trend component
        res = x - moving_mean  # Seasonal component as previously defined
        seasonal_component, noise_component = self.seasonal_decomp_module(res)
        
        
        return seasonal_component, moving_mean, noise_component
    
    
class series_decomp_multi_res(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp_multi_res, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size)).double()  
        self.seasonal_decomp_module = SeasonalDecomp()

    def forward(self, x):
        moving_mean = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean = torch.cat(moving_mean, dim=-1)
        moving_mean = torch.sum(moving_mean * nn.Softmax(-1)(self.layer(x.unsqueeze(-1))), dim=-1)
        res = x - moving_mean
        
        seasonal_component, noise_component = self.seasonal_decomp_module(res)
        return seasonal_component, moving_mean,noise_component 
    
    
    
class FourierDecomp(nn.Module):
    def __init__(self):
        super(FourierDecomp, self).__init__()
        pass

    def forward(self, x):
        x_ft = torch.fft.rfft(x, dim=-1)