from layers.TDformer_EncDec import EncoderLayer, Encoder, DecoderLayer, Decoder, AttentionLayer, series_decomp_multi,series_decomp_res,series_decomp_multi_res
import torch.nn as nn
import torch
from layers.Embed import DataEmbedding
from layers.Attention import WaveletAttention, FourierAttention, FullAttention
from layers.RevIN import RevIN
import torch.nn.functional as F

class GatedLayer(nn.Module):
    def __init__(self, d_model):
        super(GatedLayer, self).__init__()
        self.gate = nn.Linear(d_model, d_model)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gate = self.sigmoid(self.gate(x))
        return x * gate
    
class Model(nn.Module):
    """
    3Dformer 累积收益方向预测版：decoder 原生输出单步 [N, c_out]。
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.version = configs.version
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.output_stl = configs.output_stl
        self.device = torch.device(getattr(configs, 'device', 'cpu'))
        # Decomp
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi_res(kernel_size)  
        else:
            self.decomp = series_decomp_res(kernel_size)

        # Embedding 初始化（定义）
        self.enc_seasonal_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.dec_seasonal_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.enc_trend_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.dec_trend_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        
        # Encoder
        if configs.version == 'Wavelet':
            enc_self_attention = WaveletAttention(in_channels=configs.d_model,
                                                  out_channels=configs.d_model,
                                                  seq_len_q=configs.seq_len,
                                                  seq_len_kv=configs.seq_len,
                                                  ich=configs.d_model,
                                                  T=configs.temp,
                                                  activation=configs.activation,
                                                  output_attention=configs.output_attention)
            dec_self_attention = WaveletAttention(in_channels=configs.d_model,
                                                  out_channels=configs.d_model,
                                                  seq_len_q=configs.seq_len // 2 + 1,
                                                  seq_len_kv=configs.seq_len // 2 + 1,
                                                  ich=configs.d_model,
                                                  T=configs.temp,
                                                  activation=configs.activation,
                                                  output_attention=configs.output_attention)
            dec_cross_attention = WaveletAttention(in_channels=configs.d_model,
                                                   out_channels=configs.d_model,
                                                   seq_len_q=configs.seq_len // 2 + 1,
                                                   seq_len_kv=configs.seq_len,
                                                   ich=configs.d_model,
                                                   T=configs.temp,
                                                   activation=configs.activation,
                                                   output_attention=configs.output_attention)
        elif configs.version == 'Fourier':
            enc_self_attention = FourierAttention(T=configs.temp, activation=configs.activation,
                                                  output_attention=configs.output_attention)
            dec_self_attention = FourierAttention(T=configs.temp, activation=configs.activation,
                                                  output_attention=configs.output_attention)
            dec_cross_attention = FourierAttention(T=configs.temp, activation=configs.activation,
                                                   output_attention=configs.output_attention)
        # elif configs.version == 'Time':
        else:
            ## True Flase为是否对未来数据进行掩码（填充为负无穷，最后做softmax就为0
            enc_self_attention = FullAttention(False, T=configs.temp, activation=configs.activation,
                                               attention_dropout=configs.dropout,
                                               output_attention=configs.output_attention)
            dec_self_attention = FullAttention(True, T=configs.temp, activation=configs.activation,
                                               attention_dropout=configs.dropout,
                                               output_attention=configs.output_attention)
            dec_cross_attention = FullAttention(False, T=configs.temp, activation=configs.activation,
                                                attention_dropout=configs.dropout,
                                                output_attention=configs.output_attention)

        ## 季节性（周期）默认用fourier attention
        ## 趋势用full attention

        # Seasonal
        # Encoder
        self.seasonal_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(enc_self_attention, configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)   ## 生成多个层
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Decoder
        self.seasonal_decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(dec_self_attention, configs.d_model, configs.n_heads),
                    AttentionLayer(dec_cross_attention, configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )


        # Trend
        enc_self_attention_trend=FullAttention(False, T=configs.temp, activation=configs.activation,
                                               attention_dropout=configs.dropout,
                                               output_attention=configs.output_attention)
        
        dec_self_attention_trend=FullAttention(True, T=configs.temp, activation=configs.activation,
                                               attention_dropout=configs.dropout,
                                               output_attention=configs.output_attention)
        
        dec_cross_attention_trend=FullAttention(False, T=configs.temp, activation=configs.activation,
                                                attention_dropout=configs.dropout,
                                                output_attention=configs.output_attention)
        
        
        self.trend_encoder = Encoder(
            [EncoderLayer(AttentionLayer(enc_self_attention_trend, configs.d_model, configs.n_heads), configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation) for l in range(configs.e_layers)],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.trend_decoder = Decoder(
            [DecoderLayer(AttentionLayer(dec_self_attention_trend, configs.d_model, configs.n_heads), AttentionLayer(dec_cross_attention_trend, configs.d_model, configs.n_heads), configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation) for l in range(configs.d_layers)],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )


        # Residual
        self.revin_residual = RevIN(configs.enc_in).to(self.device)
        self.revin_volatility = RevIN(configs.d_model).to(self.device)
        self.residual_lstm = nn.LSTM(configs.enc_in, configs.d_model, batch_first=True)
        self.residual_proj = nn.Linear(configs.d_model, configs.c_out)
        
        # Residual MLP: seq_len → d_model → d_model → 1 (单步输出)
        self.residual_mlp = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, 1)
        )
        
        # Gated Layers
        self.seasonal_gate = GatedLayer(configs.c_out)
        self.trend_gate = GatedLayer(configs.c_out)
        self.residual_gate = GatedLayer(configs.c_out)
        
    def _build_dec_mark(self, x_mark_dec):
        """从完整 y_mark 中裁剪出 decoder 所需的时间特征: label_len 天已知 + 1 天待预测"""
        return torch.cat([
            x_mark_dec[:, :self.label_len, :],
            x_mark_dec[:, -1:, :],
        ], dim=1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        执行函数：
        1、序列分解 (Decomposition)：将x_enc分解为seasonal(x-trend后，保留快速傅里叶变换的高频波段，转为时域)、trend（Avgpool前后取平均）、residual_enc（x-trend-seasonal）
        2、对seasonal_enc进行embedding（特征升维），得到 enc_out ，之后进行seasonal_encoder，得到enc_out
        3、将dec_out进行embedding（特征升维），得到 dec_out .接着与enc_out一同进入 seasonal_decoder ，得到 seasonal_out
        """
        ## Seasonal（使用fourier attention）    
        seasonal_enc, trend_enc, residual_enc = self.decomp(x_enc)

        enc_out = self.enc_seasonal_embedding(seasonal_enc, x_mark_enc)  
        enc_out, attn_e = self.seasonal_encoder(enc_out, attn_mask=enc_self_mask) 

        seasonal_dec = F.pad(seasonal_enc[:, -self.label_len:, :], (0, 0, 0, 1))
        seasonal_mark = self._build_dec_mark(x_mark_dec)

        dec_out = self.dec_seasonal_embedding(seasonal_dec, seasonal_mark)
        seasonal_out, attn_d = self.seasonal_decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        seasonal_out = seasonal_out[:, -1, :]
        seasonal_out = self.seasonal_gate(seasonal_out)


        ## === Trend（full attention）===
        trend_dec = F.pad(trend_enc[:, -self.label_len:, :], (0, 0, 0, 1))
        trend_enc_out = self.enc_trend_embedding(trend_enc, x_mark_enc)
        trend_enc_out, _ = self.trend_encoder(trend_enc_out, attn_mask=None)
        trend_dec_out = self.dec_trend_embedding(trend_dec, seasonal_mark)
        trend_out, _ = self.trend_decoder(trend_dec_out, trend_enc_out, x_mask=None, cross_mask=None)
        trend_out = trend_out[:, -1, :]
        trend_out = self.trend_gate(trend_out)


        ## === Residual（RevIN → MLP → LSTM → RevIN → proj）===
        residual_out = self.revin_residual(residual_enc, 'norm')
        residual_out = self.residual_mlp(residual_out.permute(0, 2, 1)).permute(0, 2, 1)

        residual_out, _ = self.residual_lstm(residual_out)
        residual_out = self.revin_volatility(residual_out, 'norm')
        residual_out = self.residual_proj(residual_out)
        residual_out = self.residual_gate(residual_out).squeeze(1)

        dec_out = trend_out + residual_out + seasonal_out
        output = F.log_softmax(dec_out, dim=-1)
        return output
