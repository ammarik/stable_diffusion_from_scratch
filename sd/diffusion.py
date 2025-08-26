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


class UNETResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_time: int=1280):
        super().__init__()

        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feature: torch.Tensor, time):
        # feature (latent): (batch_size, in_channels, height, width)
        # time embedding: (1, 1280)
        residue = feature

        feature = self.groupnorm_feature(feature)

        feature = F.silu(feature)

        feature = self.conv_feature(feature)

        time = F.silu(time)

        time = self.linear_time(time)

        merged = feature + time.unsqueeze(-1).unsqueeze(-1) # (Time doesen't have batch and channel dimensions, so we add them here with unsqueeze)

        merged = self.groupnorm_merged(merged)

        merged = F.silu(merged)

        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, channels, height, width) -> (batch_size, channels, height * 2, width * 2)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class SwitchSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNETAttentionBlock):
                x = layer(x, context) # Cross attention beween our latents and the prompt
            elif isinstance(layer, UNETResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class UNET(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoders = nn.ModuleList(
            [
            # (batch_size, 4, height/8, width/8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),

            SwitchSequential(UNETResidualBLock(320, 320), UNETAttentionBlock(8, 40)),

            SwitchSequential(UNETResidualBLock(320, 320), UNETAttentionBlock(8,40)),

            # (batch_size, 320, height/8, width/8) -> (batch_size, 320, height/16, width/16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNETResidualBLock(320, 640), UNETAttentionBlock(8, 80)),

            SwitchSequential(UNETResidualBLock(640, 640), UNETAttentionBlock(8, 80)),

            # (batch_size, 640, height/16, width/16) -> (batch_size, 640, height/32, width/32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNETResidualBLock(640, 1280), UNETAttentionBlock(8, 160)),

            SwitchSequential(UNETResidualBLock(1280, 1280), UNETAttentionBlock(8, 160)),

            # (batch_size, 1280, height/32, width/32) -> (batch_size, 1280, height/64, width/64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNETResidualBLock(1280, 1280)),

            # (batch_size, 1280, height/64, width/64) -> (batch_size, 1280, height/64, width/64)
            SwitchSequential(UNETResidualBLock(1280, 1280)),
        ])

        self.bottleneck = SwitchSequential(
            UNETResidualBlock(1280, 1280),
            UNETAttentionBlock(8, 160),
            UNETResidualBlock(1280, 1280),
        )

        self.decoders = nn.ModuleList([
            SwitchSequential(UNETResidualBlock(2560, 1280)),
            SwitchSequential(UNETResidualBlock(2560, 1280)),
            SwitchSequential(UNETResidualBlock(2560, 1280), Upsample(1280)),
            SwitchSequential(UNETResidualBLock(2560, 1280), UNETAttentionBlock(8, 160)),
            SwitchSequential(UNETResidualBLock(2560, 1280), UNETAttentionBlock(8, 160)),
            SwitchSequential(UNETResidualBLock(1920, 1280), UNETAttentionBlock(8, 160), Upsample(1280)),
            SwitchSequential(UNETResidualBLock(1920, 640), UNETAttentionBlock(8, 80)),
            SwitchSequential(UNETResidualBLock(1280, 640), UNETAttentionBlock(8, 80)),
            SwitchSequential(UNETResidualBLock(640, 640), UNETAttentionBlock(8, 80), Upsample(640)),
            SwitchSequential(UNETResidualBLock(960, 320), UNETAttentionBlock(8, 40)),
            SwitchSequential(UNETResidualBLock(640, 320), UNETAttentionBlock(8, 80)),
            SwitchSequential(UNETResidualBLock(640, 320), UNETAttentionBlock(8, 40)),
        ])


class UNETOutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.gorupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # We need to convert (batch, 320, height/8, width/8) to (batch, 4, height/8, width/8)
        # x: (batch, 320, height/8, width/8)

        x = self.gorupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
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




