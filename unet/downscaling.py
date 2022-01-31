### LIBRARIES ###
# Global libraries
import torch
import torch.nn as nn

# Custom libraries
from unet.convolutions import DoubleConvolution

### CLASS DEFINITION ###
class Down(nn.Module):
    """Downscaling with maxpool, then double convolution."""

    def __init__(self, in_channels: int, out_channels: int):
        """Initiates the downscaling module.

        Args:
            in_channels: int
                number of input channels
            out_channels: int
                number of output channels
        """
        super(Down, self).__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConvolution(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor):
        """Forward function of the downscaling module.

        Args:
            x: torch.Tensor
        Returns:
            _: torch.Tensor
        """
        return self.maxpool_conv(x)
