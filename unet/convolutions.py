### LIBRARIES ###
# Global libraries
import torch
import torch.nn as nn

### CLASS DEFINITIONS ###
class DoubleConvolution(nn.Module):
    """Executes two convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = None,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = False,
    ):
        """Initiates the Double Convolution module.

        Args:
            in_channels: int
                number of input channels
            out_channels: int
                number of output channels
            mid_channels: int | None
                number of intermediate channels
            kernel_size: int
                size of the kernel to use for convolution
            padding: int
                padding to use for convolution
            bias: bool
                whether to use bias in the convolution
        """

        super(DoubleConvolution, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        """Forward function of the double convolution module.

        Args:
            x: torch.Tensor

        Returns:
            _: torch.Tensor
        """
        return self.double_conv(x)
