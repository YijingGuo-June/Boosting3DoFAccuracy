import os
from typing import Callable, Optional, cast
import numpy as np
import torch
from torchvision import transforms
from matplotlib.figure import Figure
from torch import Tensor
from torchgeo.datasets.geo import NonGeoDataset
import matplotlib.pyplot as plt
from PIL import Image

class VIGOR(NonGeoDataset):
    splits = ["train", "val"]
    
    def __init__(
        self,
        split: str = "train",
        transforms=None,
        type: str = "ground"  # Add type parameter
    ) -> None:
        """Initialize a new VIGOR dataset instance.

        Args:
            split: one of "train" or "val"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            type: one of "satellite" or "ground"

        Raises:
            AssertionError: if ``split`` or ``type`` argument is invalid
        """
        self.base_dir = "/data/dataset/VIGOR"
        assert split in self.splits
        assert type in ["satellite", "ground"]
        self.split = split
        self.type = type
        self.transforms = transforms

        # Define city list
        self.city_list = ["Chicago", "NewYork", "SanFrancisco", "Seattle"]
        self.image_list = []

        # Load image list for each city
        for city in self.city_list:
            if self.type == "satellite":
                satellite_dir = os.path.join(self.base_dir, city, 'satellite')
                for filename in os.listdir(satellite_dir):
                    if filename.endswith('.png'):
                        self.image_list.append(os.path.join(satellite_dir, filename))
            else:  # ground
                panorama_dir = os.path.join(self.base_dir, city, 'panorama')
                for filename in os.listdir(panorama_dir):
                    if filename.endswith('.jpg'):
                        self.image_list.append(os.path.join(panorama_dir, filename))
                

        # Split into train and val (80:20)
        num_total = len(self.image_list)
        num_train = int(0.8 * num_total)
        
        if split == "train":
            self.image_list = self.image_list[:num_train]
        else:  # val
            self.image_list = self.image_list[num_train:]

    def __len__(self):
        """Return the number of data points in the dataset."""
        return len(self.image_list)

    def __getitem__(self, index: int):
        """Return an index within the dataset.

        Args:
            index: index to return
        Returns:
            data and label at that index
        """
        img_name = self.image_list[index]
        # print(f"Image name: {img_name}")
        
        if self.type == "satellite":
            img_path = os.path.join(img_name)
        else:
            img_path = os.path.join(img_name)

        # Load and convert image to RGB
        image = Image.open(img_path).convert('RGB')
        to_tensor = transforms.ToTensor()
        image = to_tensor(image)  # 自动将像素值归一化到[0,1]
        
        # print(f"Image tensor shape before additional_transforms: {image.shape}")
        # print(f"Image tensor type before additional_transforms: {type(image)}")
        # print("self.transforms: ", self.transforms)

        if self.transforms:
            image = self.transforms({"image": image})["image"] # 将张量包装在字典中

        # print(f"Image tensor shape after additional_transforms: {image.shape}")

        return {"image": image}

    def plot(
        self,
        sample,
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by __getitem__
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        fig, ax = plt.subplots(figsize=(5, 5))
        
        ax.imshow(sample["image"].permute(1, 2, 0))
        ax.axis("off")
        if show_titles:
            ax.set_title(f"{self.type.capitalize()} View")

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
