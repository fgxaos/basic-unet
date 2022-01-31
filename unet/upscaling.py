### LIBRARIES ###
# Global libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom libraries
from unet.convolutions import DoubleConvolution

### CLASS DEFINITION ###
class Up(nn.Module):
    """Upscaling module."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        """Initiates the upscaling module.

        Args:
            in_channels: int
                number of input channels
            out_channels: int
                number of output channels
            bilinear: bool
                whether to use 'normal' convolutions to reduce the number of channels
        """
        super(Up, self).__init__()

        # If `bilinear`, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConvolution(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConvolution(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """Forward function of the upscaling module.

        Args:
            x1: torch.Tensor
                tensor to upscale
            x2: torch.Tensor
                residual tensor
        Returns:
            _: torch.Tensor
                upscaled tensor
        """
        # Upscale the tensor x1
        x1 = self.up(x1)

        # Concatenate the residual x2
        diff_x = x2.shape[3] - x1.shape[3]
        diff_y = x2.shape[2] - x1.shape[2]

        x1 = F.pad(
            x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )
        x = torch.cat([x2, x1], dim=1)

        # Apply convolution to the newly formed tensor
        return self.conv(x)
