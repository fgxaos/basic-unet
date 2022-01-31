### LIBRARIES ###
# Global libraries
import torch
import torch.nn as nn

### CLASS DEFINITION ###
class OutputConvolution(nn.Module):
    """Output convolution module."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1):
        """Initiates the output convolution module.

        Args:
            in_channels: int
                number of input channels
            out_channels: int
                number of output channels
            kernel_size: int
                size of the kernel to use for the convolution
        """
        super(OutputConvolution, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor):
        """Forward function of the output convolution module.

        Args:
            x: torch.Tensor
        """
        return self.conv(x)
