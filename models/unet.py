import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels, 2 * channels, kernel_size=1),
            nn.ELU()
        )

    def forward(self, x):
        res = self.block(x)
        gated_res = self.gate(res)
        return gated_res


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, base_channels=64, num_layers=5):
        super(UNet, self).__init__()

        # Encoder (Bottom-Up Path)
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for _ in range(num_layers):
            self.encoder.append(nn.Sequential(
                nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
                nn.ELU(),
                ResidualBlock(base_channels)
            ))
            in_channels = base_channels
            base_channels *= 2

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels // 2, base_channels, kernel_size=3, padding=1),
            nn.ELU(),
            ResidualBlock(base_channels)
        )

        # Decoder (Top-Down Path)
        self.decoder = nn.ModuleList()
        base_channels //= 2
        for _ in range(num_layers):
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ELU(),
                ResidualBlock(base_channels)
            ))
            base_channels //= 2

        # Final Convolution
        self.final_conv = nn.Conv2d(base_channels * 2, out_channels, kernel_size=3, padding=1)

    def forward(self, x, return_latent=False):
        # Encoder
        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
            x = self.pool(x)

        # Bottleneck
        latent = self.bottleneck(x)

        # Decoder
        for decode, skip in zip(self.decoder, reversed(skips)):
            x = decode(x)
            x = torch.cat([x, skip], dim=1)

        # Final Convolution
        x = self.final_conv(x)

        # Extract center pixel prediction
        batch_size, num_classes, h, w = x.shape
        center_prediction = x[:, :, h // 2, w // 2]

        if return_latent:
            return latent, center_prediction
        return center_prediction
