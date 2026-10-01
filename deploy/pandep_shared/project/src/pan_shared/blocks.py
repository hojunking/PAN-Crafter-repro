"""Fixed PAN-Crafter-paper numerical blocks, with no conditional modules.

Derived from KAIST-VICLab PAN-Crafter (2025). See vendor_reference/model_source
for the unmodified source, parameter mapping and non-commercial license notice.
"""
import torch
from torch import nn


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ResBlock(nn.Module):
    def __init__(self, channels, out_channels=None):
        super().__init__()
        out = out_channels or channels
        self.in_layers = nn.Sequential(ChannelLayerNorm(channels), nn.SiLU(),
                                       nn.Conv2d(channels, out, 3, padding=1))
        self.out_layers = nn.Sequential(ChannelLayerNorm(out), nn.SiLU(),
                                        nn.Dropout(0.0), nn.Conv2d(out, out, 3, padding=1))
        self.skip_connection = nn.Identity() if channels == out else nn.Conv2d(channels, out, 1)
        nn.init.zeros_(self.out_layers[3].weight)
        nn.init.zeros_(self.out_layers[3].bias)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers[0](h)
        h = self.out_layers[1:](h)
        return self.skip_connection(x) + h


class DownConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, 2, stride=2, padding=0)

    def forward(self, x):
        return self.conv(x)


class SharedTrunk(nn.Module):
    def __init__(self, width, depth=(1, 2, 2)):
        super().__init__()
        if tuple(depth) != (1, 2, 2):
            raise ValueError('Only the registered D122 trunk is supported')
        d0, d1, d2 = depth
        self.encoder1 = nn.ModuleList([ResBlock(width) for _ in range(d0)])
        self.down1 = DownConv(width)
        self.encoder2 = nn.ModuleList([ResBlock(width) for _ in range(d1)])
        self.down2 = DownConv(width)
        self.middle = nn.ModuleList([ResBlock(width) for _ in range(d2)])
        self.up2 = UpConv(width)
        self.decoder2 = nn.ModuleList([ResBlock(width * 2, width)] + [ResBlock(width) for _ in range(d1 - 1)])
        self.up1 = UpConv(width)
        self.decoder1 = nn.ModuleList([ResBlock(width * 2, width)] + [ResBlock(width) for _ in range(d0 - 1)])
        self.output_norm = ChannelLayerNorm(width)
        self.output_act = nn.SiLU()

    def forward(self, x):
        for block in self.encoder1:
            x = block(x)
        skip1 = x
        x = self.down1(x)
        for block in self.encoder2:
            x = block(x)
        skip2 = x
        x = self.down2(x)
        for block in self.middle:
            x = block(x)
        x = torch.cat((self.up2(x), skip2), dim=1)
        for block in self.decoder2:
            x = block(x)
        x = torch.cat((self.up1(x), skip1), dim=1)
        for block in self.decoder1:
            x = block(x)
        return self.output_act(self.output_norm(x))
