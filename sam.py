from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import torch
from segment_anything import sam_model_registry
from torch import nn
import torch.nn.functional as F
from .dpt import DPT



class SAM(nn.Module):
    def __init__(self, arch="vit_b", output="dense", layer=-1, return_multilayer=False,
                 output_channels = 320):
        super().__init__()

        assert output in ["gap", "dense"], "Options: [gap, dense]"
        self.output = output
        self.checkpoint_name = f"sam_{arch}"
        ckpt_paths = {
            "vit_b": "sam_vit_b_01ec64.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_h": "sam_vit_h_4b8939.pth",
        }

        ckpt_file = ckpt_paths[arch]
        ckpt_path = Path(__file__).parent / "checkpoint_weights" / ckpt_file

        if not ckpt_path.exists():
            download_path = (
                f"https://dl.fbaipublicfiles.com/segment_anything/{ckpt_file}"
            )
            urlretrieve(download_path, ckpt_path)

        sam = sam_model_registry[arch](checkpoint=ckpt_path)
        vit = sam.image_encoder

        feat_dim = vit.neck[0].in_channels
        emb_h, emb_w = vit.pos_embed.shape[1:3]
        self.patch_size = vit.patch_embed.proj.kernel_size[0]
        self.image_size = (emb_h * self.patch_size, emb_w * self.patch_size)
        assert self.patch_size == 16

        vit.pos_embed = nn.Parameter(
            torch.zeros(1, 32, 32, 768)
        )

        self.vit = vit

        # frozen
        for param in self.vit.parameters():
            param.requires_grad = False

        num_layers = len(self.vit.blocks)
        multilayers = [
            num_layers // 4 - 1,
            num_layers // 2 - 1,
            num_layers // 4 * 3 - 1,
            num_layers - 1,
        ]

        if return_multilayer:
            self.feat_dim = [feat_dim, feat_dim, feat_dim, feat_dim]
            self.multilayers = multilayers
        else:
            self.feat_dim = feat_dim
            layer = multilayers[-1] if layer == -1 else layer
            self.multilayers = [layer]

        # define layer name (for logging)
        self.layer = "-".join(str(_x) for _x in self.multilayers)

        self.flatten = nn.Flatten(2)
        self.mlp = MLP(input_dim=768, hidden_dim=512, output_dim=320)
        self.unflatten = nn.Unflatten(2, (32, 32))
        self.conv_down = nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

        # self.dpt = DPT(self.feat_dim, output_channels)

    def resize_pos_embed(self, image_size):
        # get embed size
        h, w = image_size
        h = h // self.patch_size
        w = w // self.patch_size

        # resize embed
        pos_embed = self.vit.pos_embed.data.permute(0, 3, 1, 2)
        pos_embed = torch.nn.functional.interpolate(
            pos_embed, size=(h, w), mode="bicubic"
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        self.vit.pos_embed.data = pos_embed
        self.image_size = image_size

    def forward(self, x):
        # with torch.no_grad():
        _, _, h, w = x.shape
        assert h % self.patch_size == 0 and w % self.patch_size == 0, f"{h}, {w}"

        # if h != self.image_size[0] or w != self.image_size[1]:
        #     self.resize_pos_embed(image_size=(h, w))

        # run vit
        x = self.vit.patch_embed(x)
        if self.vit.pos_embed is not None:
            x = x + self.vit.pos_embed

        embeds = []
        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if i in self.multilayers:
                embeds.append(x)
                if len(embeds) == len(self.multilayers):
                    break

        # feat shape is batch x feat_dim x height x width
        embeds = [_emb.permute(0, 3, 1, 2).contiguous() for _emb in embeds]

        if self.output == "gap":
            embeds = [x.mean(dim=(-2, -1)) for x in embeds]

        # pure_sam = embeds[0]
        # embeds = self.dpt(embeds)
        # output_tensor = F.interpolate(embeds, size=(16, 16), mode='bilinear', align_corners=True)
        # return pure_sam, output_tensor

        x = embeds[0]
        fmap = x
        x = self.flatten(x)  # 形状变为 [16, 768, 1024]
        x = x.permute(0, 2, 1)  # 形状变为 [16, 1024, 768] 以便应用 MLP
        x = self.mlp(x)  # 形状变为 [16, 1024, 320]
        x = x.permute(0, 2, 1)  # 形状变为 [16, 320, 1024]

        x = self.unflatten(x)  # 形状变为 [16, 320, 16, 16]
        x = self.conv_down(x)
        x = self.relu(x)
        embeds[0] = x


        # return embeds[0] if len(embeds) == 1 else embeds
        return fmap, x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x