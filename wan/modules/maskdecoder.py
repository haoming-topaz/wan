import torch
import torch.nn as nn


class ResidualBlock2D_on_3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(1,3,3), padding=(0,1,1))
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(1,3,3), padding=(0,1,1))
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=(1,1,1)),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out

class MaskDecoder3D(nn.Module):
    def __init__(self, in_channels, base_channels=256, num_upsample=3, res_blocks_per_up=2):
        super().__init__()

        self.res_stack = nn.Sequential(
            ResidualBlock2D_on_3D(in_channels, base_channels),
            ResidualBlock2D_on_3D(base_channels, base_channels)
        )

        self.up_blocks = nn.ModuleList()
        for i in range(num_upsample):
            up_layers = [
                nn.ConvTranspose3d(base_channels, base_channels // 2, kernel_size=(1,2,2), stride=(1,2,2))
            ]
            for _ in range(res_blocks_per_up):
                up_layers.append(ResidualBlock2D_on_3D(base_channels // 2, base_channels // 2))

            self.up_blocks.append(nn.Sequential(*up_layers))
            base_channels = base_channels // 2

        self.final_conv = nn.Conv3d(base_channels, 1, kernel_size=(1,1,1))

    def forward(self, z):
        assert z.shape[2] == 1, "This decoder assumes T=1!"

        x = self.res_stack(z)

        for up_block in self.up_blocks:
            x = up_block(x)

        mask_logits = self.final_conv(x)
        return mask_logits
