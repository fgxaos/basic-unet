### LIBRARIES ###
# Global libraries
import torch
import torch.nn as nn

import pytorch_lightning as pl

# Custom libraries
from unet.convolutions import DoubleConvolution
from unet.downscaling import Down
from unet.upscaling import Up
from unet.out_convolution import OutputConvolution

### CLASS DEFINITION ###
class UNet(pl.LightningModule):
    def __init__(self, n_channels: int, n_classes: int, bilinear: bool = True):
        """Initiates the U-Net module.

        Args:
            n_channels: int
                number of input channels
            n_classes: int
                number of classes to predict
            bilinear: bool
                whether to use 'normal' convolutions to reduce the number of channels
        """
        super(UNet, self).__init__()

        factor = 2 if bilinear else 1

        # Encoding the input
        self.inc = DoubleConvolution(n_channels, 64)

        # Downscaling the input
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        # Upscaling the input
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # Return the output
        self.outc = OutputConvolution(64, n_classes)

        # Training loss function
        if n_classes > 1:
            self.train_loss = nn.CrossEntropyLoss()
        else:
            self.train_loss = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor):
        """Forward function of the U-Net model.

        Args:
            x: torch.Tensor
        """
        # Encoding the input
        x_enc = self.inc(x)

        # Downscaling the input
        x_down1 = self.down1(x_enc)
        x_down2 = self.down2(x_down1)
        x_down3 = self.down3(x_down2)
        x_down4 = self.down4(x_down3)

        # Upscaling the input
        x_up = self.up1(x_down4, x_down3)
        x_up = self.up2(x_up, x_down2)
        x_up = self.up3(x_up, x_down1)
        x_up = self.up4(x_up, x_enc)

        return self.outc(x_up)

    def training_step(self, batch, batch_idx):
        x, ground_truth = batch

        # Compute predictions
        pred = self(x)

        # Compute the loss
        print("Ground_truth: ", ground_truth)
        loss = self.train_loss(pred, ground_truth.long())

        # Log the results
        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=0.1, weight_decay=1e-8)
