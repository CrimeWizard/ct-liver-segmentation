# code/models/unet_binary_roi.py
# Fixed lightweight U-Net for binary tumor ROI segmentation

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Building blocks
# ------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_transpose=False):
        super().__init__()
        if use_transpose:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.reduce = nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, bias=False)

        self.conv = ConvBlock(in_ch, out_ch)
        self.use_transpose = use_transpose

    def forward(self, x, skip):
        if self.use_transpose:
            x = self.up(x)
        else:
            x = self.up(x)
            x = self.reduce(x)

        # pad if needed (odd sizes)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ------------------------------------------------------------
# U-Net architecture
# ------------------------------------------------------------
class UNetBinaryROI(nn.Module):
    def __init__(self, in_channels=1, base_ch=32, dropout=0.1, use_transpose=False):
        super().__init__()
        self.inc = ConvBlock(in_channels, base_ch)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_ch, base_ch * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_ch * 2, base_ch * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_ch * 4, base_ch * 8))
        self.bot = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_ch * 8, base_ch * 16))

        self.up1 = UpBlock(base_ch * 16, base_ch * 8, use_transpose)
        self.up2 = UpBlock(base_ch * 8, base_ch * 4, use_transpose)
        self.up3 = UpBlock(base_ch * 4, base_ch * 2, use_transpose)
        self.up4 = UpBlock(base_ch * 2, base_ch, use_transpose)

        self.outc = nn.Conv2d(base_ch, 1, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        xb = self.bot(x4)

        x = self.up1(xb, x4)
        x = self.dropout(x)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = UNetBinaryROI(in_channels=1, base_ch=32)
    x = torch.randn(2, 1, 256, 256)
    y = m(x)
    print("Output:", y.shape, "Params (M):", round(count_parameters(m) / 1e6, 2))
