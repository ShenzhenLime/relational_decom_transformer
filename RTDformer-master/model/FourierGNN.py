import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, configs):
#     def __init__(self, pre_length, embed_size,
#                  feature_size, seq_length, hidden_size, hard_thresholding_fraction=1, hidden_size_factor=1, sparsity_threshold=0.01):
        super(Model, self).__init__()
        self.embed_size = configs.d_model
        self.hidden_size = configs.d_model
        self.number_frequency = 1
        self.pre_length =configs.pred_len
        self.feature_size = configs.enc_in
        self.seq_length = configs.seq_len
        self.frequency_size = self.embed_size // self.number_frequency
        self.hidden_size_factor = 1
        self.sparsity_threshold = 0.01
        self.hard_thresholding_fraction = 1
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        #('--embed_size', type=int, default=128, help='hidden dimensions')

        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size, self.frequency_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor, self.frequency_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size))
        self.w3 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size,
                                     self.frequency_size * self.hidden_size_factor))
        self.b3 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.embeddings_10 = nn.Parameter(torch.randn(self.seq_length, 8))
        self.fc = nn.Sequential(
            nn.Linear(self.embed_size * 8, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length)
        )
        self.to('cuda:1')
        self.projector2 = nn.Linear(configs.enc_in, 2, bias=True)

    def tokenEmb(self, x):
        x = x.unsqueeze(2)  #在第三个维度上，添加一个维度，大小为1
        
        # [B,L×D,1] ×[1,128]-->[B,L×D,128]
        y = self.embeddings  #torch.Size([1, 128])

        return x * y

    # FourierGNN
    def fourierGC(self, x, B, N, L):
        
        # 128*
        o1_real = torch.zeros([B, (N*L)//2 + 1, self.frequency_size * self.hidden_size_factor],
                              device=x.device)
        o1_imag = torch.zeros([B, (N*L)//2 + 1, self.frequency_size * self.hidden_size_factor],
                              device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        o3_real = torch.zeros(x.shape, device=x.device)
        o3_imag = torch.zeros(x.shape, device=x.device)
        
        o1_real = F.relu(
            torch.einsum('bli,ii->bli', x.real, self.w1[0]) - \
            torch.einsum('bli,ii->bli', x.imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag = F.relu(
            torch.einsum('bli,ii->bli', x.imag, self.w1[0]) + \
            torch.einsum('bli,ii->bli', x.real, self.w1[1]) + \
            self.b1[1]
        )

        # 1 layer
        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)

        o2_real = F.relu(
            torch.einsum('bli,ii->bli', o1_real, self.w2[0]) - \
            torch.einsum('bli,ii->bli', o1_imag, self.w2[1]) + \
            self.b2[0]
        )

        o2_imag = F.relu(
            torch.einsum('bli,ii->bli', o1_imag, self.w2[0]) + \
            torch.einsum('bli,ii->bli', o1_real, self.w2[1]) + \
            self.b2[1]
        )

        # 2 layer
        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = x + y

        o3_real = F.relu(
                torch.einsum('bli,ii->bli', o2_real, self.w3[0]) - \
                torch.einsum('bli,ii->bli', o2_imag, self.w3[1]) + \
                self.b3[0]
        )

        o3_imag = F.relu(
                torch.einsum('bli,ii->bli', o2_imag, self.w3[0]) + \
                torch.einsum('bli,ii->bli', o2_real, self.w3[1]) + \
                self.b3[1]
        )

        # 3 layer
        z = torch.stack([o3_real, o3_imag], dim=-1)
        z = F.softshrink(z, lambd=self.sparsity_threshold)
        z = z + x
        z = torch.view_as_complex(z)
        return z

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        x=x_enc
        x = x.permute(0, 2, 1).contiguous()
        B, N, L = x.shape
        x = x.reshape(B, -1)

        x = self.tokenEmb(x)
        
#         print(x.shape) #torch.Size([32, 1680, 128])

        x = torch.fft.rfft(x, dim=1, norm='ortho')  #对输入张量x的第二个维度执行一维实数快速傅里叶变换，并且采用正交归一化。
        #实数序列的傅里叶变换中，输出的长度是输入长度的一半加一
#         print(x.shape)  #torch.Size([32, 841, 128])
        x = x.reshape(B, (N*L)//2+1, self.frequency_size)

        bias = x

        # FourierGNN
        x = self.fourierGC(x, B, N, L)

        x = x + bias
        
#         print(x.shape)torch.Size([32, 841, 128])

        x = x.reshape(B, (N*L)//2+1, self.embed_size)
    
#         print("1",x.shape)  # torch.Size([32, 841, 128])

        # ifft
        x = torch.fft.irfft(x, n=N*L, dim=1, norm="ortho")
        
#         print("2",x.shape) #2 torch.Size([32, 1680, 128])

        x = x.reshape(B, N, L, self.embed_size)
        x = x.permute(0, 1, 3, 2)  # B, N, D, L
        
#         print("3",x.shape) #3 torch.Size([32, 140, 128, 12])

        # projection
        x = torch.matmul(x, self.embeddings_10)
        
#         print("4",x.shape)  #4 torch.Size([32, 140, 128, 8])
        x = x.reshape(B, N, -1)  #
        
#         print("5",x.shape)5 torch.Size([32, 140, 1024])
        x = self.fc(x)
    
        x=x.permute(0,2,1)
    
        x=self.projector2(x)
    
        x = F.log_softmax(x, dim=1)
        
#         print("6",x.shape)
        
#         print(x.shape)  #torch.Size([32, 140, 12])

        return x

