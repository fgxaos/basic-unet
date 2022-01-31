### LIBRARIES ###
# Global libraries
from pl_bolts.datamodules import CityscapesDataModule

import pytorch_lightning as pl

# Custom libraries
from unet.unet import UNet

### MAIN CODE ###
model = UNet(3, 30)

cityscapes_dataset = CityscapesDataModule(
    data_dir="cityscapes",
    quality_mode="fine",
    batch_size=5,
    num_workers=12,
    target_type="semantic",
)

trainer = pl.Trainer()
trainer.fit(model, cityscapes_dataset)
