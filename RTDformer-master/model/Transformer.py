import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np

class SpatialAttention(nn.Module):
    def __init__(self, d_model, n_heads, attention_dropout=0.1):
        super(SpatialAttention, self).__init__()
        self.full_attention = FullAttention(mask_flag=False, attention_dropout=attention_dropout, output_attention=False)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

    def forward(self, x):
        B, L, D = x.size()
        H = self.n_heads
        # Reshape for multi-head attention
        x = x.view(B, L, H, self.d_k).permute(2, 0, 1, 3).contiguous().view(H, B, L, self.d_k)
        queries, keys, values = x, x, x
        output, _ = self.full_attention(queries, keys, values, attn_mask=None)
        output = output.view(H, B, L, self.d_k).permute(1, 2, 0, 3).contiguous().view(B, L, D)
        return output

class Model(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        if configs.channel_independence:
            self.enc_in = 1
            self.dec_in = 1
            self.c_out = 1
        else:
            self.enc_in = configs.enc_in
            self.dec_in = configs.dec_in
            self.c_out = configs.c_out

        # Embedding
        self.enc_embedding = DataEmbedding(self.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.dec_embedding = DataEmbedding(self.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
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
        self.projector2=nn.Linear(configs.c_out,configs.d_model, bias=True)
        self.spatial_attention = SpatialAttention(configs.d_model, configs.n_heads, configs.dropout)
        self.projector3 = nn.Linear(configs.d_model,configs.c_out,  bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
#         dec_out=self.projector2(dec_out)
#         # Apply spatial attention
#         dec_out = self.spatial_attention(dec_out)
#         dec_out=self.projector3(dec_out)
        
        output = (dec_out[:, -self.pred_len:, :])
#         output = (output[:, -1, :])
#         print(output.shape)  (155,2)
        
        
        output = F.elu((output))
        output = F.log_softmax(output, dim=1)
        
        
        return output
