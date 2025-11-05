# code/models/unet.py
# Baseline 2D U-Net (lightweight) for liver/tumor segmentation
# Returns RAW LOGITS (no sigmoid); use BCEWithLogits + Dice in training.

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Building blocks
# ------------------------------------------------------------
class ConvBlock(nn.Module):
    """Two consecutive conv-BN-ReLU layers (optionally with dropout)."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.do = nn.Dropout2d(dropout) if (dropout and dropout > 0) else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.do(self.relu(self.bn2(self.conv2(x))))
        return x


class Down(nn.Module):
    """Downscale with max-pool then double conv."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float | None = None):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ConvBlock(in_ch, out_ch, dropout=None if dropout is None else dropout * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class Up(nn.Module):
    """Upscale + skip-concat + double conv."""
    def __init__(self, in_ch: int, out_ch: int, use_transpose: bool = False, dropout: float | None = None):
        super().__init__()
        self.use_transpose = use_transpose
        if use_transpose:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.reduce = nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, bias=False)
            nn.init.kaiming_normal_(self.reduce.weight, nonlinearity="relu")

        self.block = ConvBlock(in_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.use_transpose:
            x = self.up(x)
        else:
            x = self.up(x)
            x = self.reduce(x)

        # Handle size mismatches (odd inputs)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([skip, x], dim=1)
        return self.block(x)


# ------------------------------------------------------------
# U-Net architecture
# ------------------------------------------------------------
class UNet(nn.Module):
    """
    Baseline 2-D U-Net (4 encoder + 4 decoder levels).

    Args:
        in_channels: 1 for CT
        out_channels: 1 (raw logits; apply sigmoid externally)
        base_ch: starting feature width (32 fits MX450)
        dropout: dropout prob in decoder blocks (e.g., 0.5)
        use_transpose: if True, use ConvTranspose2d for upsampling
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_ch: int = 32,
        dropout: float = 0.5,
        use_transpose: bool = False,
    ):
        super().__init__()

        # Encoder
        self.inc = ConvBlock(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2, dropout)
        self.down2 = Down(base_ch * 2, base_ch * 4, dropout)
        self.down3 = Down(base_ch * 4, base_ch * 8, dropout)

        # Bottleneck
        self.bot = ConvBlock(base_ch * 8, base_ch * 16, dropout)

        # Decoder
        self.up1 = Up(base_ch * 16, base_ch * 8, use_transpose, dropout)
        self.up2 = Up(base_ch * 8,  base_ch * 4, use_transpose, dropout)
        self.up3 = Up(base_ch * 4,  base_ch * 2, use_transpose, dropout)
        self.up4 = Up(base_ch * 2,  base_ch,     use_transpose, dropout)

        # Output head
        self.outc = nn.Conv2d(base_ch, out_channels, kernel_size=1)
        nn.init.xavier_normal_(self.outc.weight)
        if self.outc.bias is not None:
            nn.init.zeros_(self.outc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Bottleneck
        xb = self.bot(x4)

        # Decoder with skip connections
        x = self.up1(xb, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)

        logits = self.outc(x)
        return logits


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = UNet(in_channels=1, out_channels=1, base_ch=32, dropout=0.5, use_transpose=False)
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    print("Output:", y.shape)
    print("Params (M):", round(count_parameters(model) / 1e6, 3))
