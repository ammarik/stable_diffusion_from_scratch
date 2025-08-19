import torch

from torch import nn
from torch.nn import functional as F

from attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self, n_embed: int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, 4 * n_embed)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, 320)
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        
        # (1, 1280)
        return x


class Diffusion(nn.Module):
    # U-Net
    def __init__(self):
        #super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNETOutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        # latent: (batch_size, 4, height/8, width/8) = output shape of VAE Encoder
        # context: (batch_size, seq_len, dim) = CLIP output shape
        # time: (1, 320)

        # (1, 320) -> (1, 1280) 
        time = self.time_embedding(time)

        # (batch, 4, height/8, width/8) -> (batch, 320, height/8, width/8)
        output = self.unet(latent, context, time)

        # (batch, 320, height/8, width/8) -> (batch, 4, height/8, width/8)
        output = self.final(output)

        return output




