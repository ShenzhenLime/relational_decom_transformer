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
    Transformer for seasonality, MLP for trend
    """
    def __init__(self, configs):
        """
        整个函数都是在传参，定义模型的结构和组件
        """
        super(Model, self).__init__()
        self.version = configs.version
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.output_stl = configs.output_stl
        self.device = torch.device(getattr(configs, 'device', 'cpu'))
        self.top_k = getattr(configs, 'top_k', 0)

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
                                                  seq_len_q=configs.seq_len // 2 + configs.pred_len,
                                                  seq_len_kv=configs.seq_len // 2 + configs.pred_len,
                                                  ich=configs.d_model,
                                                  T=configs.temp,
                                                  activation=configs.activation,
                                                  output_attention=configs.output_attention)
            dec_cross_attention = WaveletAttention(in_channels=configs.d_model,
                                                   out_channels=configs.d_model,
                                                   seq_len_q=configs.seq_len // 2 + configs.pred_len,
                                                   seq_len_kv=configs.seq_len,
                                                   ich=configs.d_model,
                                                   T=configs.temp,
                                                   activation=configs.activation,
                                                   output_attention=configs.output_attention)
        elif configs.version == 'Fourier':
            enc_self_attention = FourierAttention(T=configs.temp, activation=configs.activation,
                                                  output_attention=configs.output_attention)#
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
        
        # Residual MLP
        self.residual_mlp = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.pred_len)
        )
        # self.residual_Linear=nn.Linear(configs.seq_len,configs.pred_len)
        
        self.full_attention = FullAttention(mask_flag=False, T=configs.temp, activation=configs.activation, output_attention=False)

        self.projector2 = nn.Linear(configs.c_out, 2, bias=True)
        
        
        # Gated Layers
        self.seasonal_gate = GatedLayer(configs.c_out)
        self.trend_gate = GatedLayer(configs.c_out)
        self.residual_gate = GatedLayer(configs.c_out)
        
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

        # # 防止 NaN 值，设置一个最小值（缩放比率
        # seasonal_mean_abs = seasonal_enc.abs().mean(dim=1)  # 季节性成分的平均绝对值
        # min_value = 1e-6
        # seasonal_mean_abs = torch.where(seasonal_mean_abs == 0, torch.tensor(min_value).to(self.device), seasonal_mean_abs)
        # seasonal_ratio = seasonal_enc.abs().mean(dim=1) / seasonal_out.abs().mean(dim=1)
        # seasonal_ratio = seasonal_ratio.unsqueeze(1).expand(-1, self.pred_len, -1)

        ## 补0占位，用于解码矩阵中填充预测值。[Batch, Time, Feature] 
        # 第一对0，0表示Feature不填充，第二队0，pred_len表示Time维度的末尾填充pred_len个0
        # 原本是[32, 96, 4] -> [32, 96+48, 4]
        seasonal_dec = F.pad(seasonal_enc[:, -self.label_len:, :], (0, 0, 0, self.pred_len))   

        ## 嵌入解码器
        dec_out = self.dec_seasonal_embedding(seasonal_dec, x_mark_dec)    

        ## 开始解码
        seasonal_out, attn_d = self.seasonal_decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        
        ##保留预测的那48行，不要练手几行
        seasonal_out = seasonal_out[:, -self.pred_len:, :]

        ## 门控，调整不同成分的权重
        seasonal_out = self.seasonal_gate(seasonal_out)


        ## Trend（与Seasonal的区别就是使用全注意力机制）
        trend_dec=F.pad(trend_enc[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        
        trend_enc_out = self.enc_trend_embedding(trend_enc, x_mark_enc)
        trend_enc_out, _ = self.trend_encoder(trend_enc_out, attn_mask=None)
        trend_dec_out = self.dec_trend_embedding(trend_dec, x_mark_dec)
        trend_out, _ = self.trend_decoder(trend_dec_out, trend_enc_out, x_mask=None, cross_mask=None)
        trend_out = trend_out[:, -self.pred_len:, :]
        
        # Apply gate to trend output
        trend_out = self.trend_gate(trend_out)
        

        ## Residual（REVIN -> MLP -> LSTM -> REVIN -> MLP ）
        residual_out=self.revin_residual(residual_enc, 'norm')          # Eq.18
        residual_out = self.residual_mlp(residual_out.permute(0, 2, 1)).permute(0, 2, 1)  # Eq.19-20

        # LSTM
        residual_out, _ = self.residual_lstm(residual_out)           # Eq.21
        residual_out = self.revin_volatility(residual_out, 'norm')   # Eq.22 second RevIN
        residual_out = self.residual_proj(residual_out)              # Eq.22 MLP
        
        if self.pred_len > seasonal_enc.shape[1]:
            residual_out = F.interpolate(residual_out.permute(0, 2, 1), size=self.pred_len, mode='linear', align_corners=False).permute(0, 2, 1)
        else:
            residual_out = residual_out[:, -self.pred_len:, :]
            
        # Apply gate to residual output
        residual_out = self.residual_gate(residual_out)
        
        dec_out= trend_out + residual_out + seasonal_out
        
        ## Inter-Stock Correlation Attention，把股票当作一个token，进行全局注意力机制，捕捉不同股票之间的相关性
        dec_out = dec_out.permute(1, 0, 2)
        B, L, D = dec_out.shape
        dec_out = dec_out.unsqueeze(2)
        dec_out, _ = self.full_attention(dec_out, dec_out, dec_out, attn_mask=None, top_k=self.top_k)
        
        dec_out = dec_out.squeeze(2) 
        
        dec_out = dec_out.permute(1, 0, 2) # Reshape back to original shape


        # ACC
        dec_out=self.projector2(dec_out)
        output = F.elu(dec_out)
        output=F.log_softmax(output, dim=2)
        return output
