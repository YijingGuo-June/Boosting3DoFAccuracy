import os
import torch 
import lightning.pytorch as pl

# Set environment variables similar to caltech256.py if needed
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4" 
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["GDAL_NUM_THREADS"] = "4"

from src.datasets.vigor import VIGOR

# Default ImageNet statistics, you may want to calculate VIGOR specific values
MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])

class VIGORDataModule(pl.LightningDataModule):
    mean = MEAN 
    std = STD

    def __init__(self, root="/public/home/shiyj2-group/image_localization/VIGOR", 
                 batch_size=32, num_workers=0, transforms=None):
        super(VIGORDataModule, self).__init__()
        self.root = root
        self.train_batch_size = batch_size
        self.eval_batch_size = batch_size 
        self.num_workers = num_workers
        self.transforms = transforms

    def setup(self, stage="fit", drop_last=False):
        """Method to setup dataset and corresponding splits."""
        # Setup train and val splits (VIGOR only has train/val)
        for split in ["train", "val"]:
            ds = VIGOR(split=split, transforms=self.transforms)
            setattr(self, f"{split}_dataset", ds)
            
        self.drop_last = drop_last

    def train_dataloader(self):
        """Return training dataset loader."""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        """Return validation dataset loader."""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers, 
            shuffle=False,
            pin_memory=True,
            drop_last=self.drop_last,
        )