import time

import torch

import einops as E  

from .utils.utils import center_padding, tokens_to_output
from torch.nn.functional import interpolate
import torch.nn as nn
import torch.nn.functional as F

from .dpt import DPT

class DINO(torch.nn.Module):
    def __init__(
        self,
        dino_name="dinov2",
        model_name="vitb14",
        output="dense",
        layer=-1,
        return_multilayer=True,
        output_channels=112,
        hidden_channels1=544,
        hidden_channels2=465,
        down_sample=False,
    ):
        super().__init__()
        feat_dims = {
            "vitb8": 768,
            "vitb16": 768,
            "vitb14": 768,
            "vitb14_reg": 768,
            "vitl14": 1024,
            "vitg14": 1536,
        }

        # get model
        self.model_name = dino_name
        self.down_sample = down_sample
        self.checkpoint_name = f"{dino_name}_{model_name}"
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        dino_vit = torch.hub.load(f"facebookresearch/{dino_name}", self.checkpoint_name)
        self.vit = dino_vit.eval().to(torch.float32)
        for param in dino_vit.parameters():
            param.requires_grad = False
        self.has_registers = "_reg" in model_name

        self.flatten = nn.Flatten(2)
        self.mlp = MLP(input_dim=768, hidden_dim=512, output_dim=320)
        self.unflatten = nn.Unflatten(2, (37, 37))
        self.conv_down = nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()



        assert output in ["cls", "gap", "dense", "dense-cls"]
        self.output = output
        self.patch_size = self.vit.patch_embed.proj.kernel_size[0]

        feat_dim = feat_dims[model_name]
        feat_dim = feat_dim * 2 if output == "dense-cls" else feat_dim

        num_layers = len(self.vit.blocks)
        # print("num_layers: ", num_layers)
        # multilayers = [
        #     num_layers // 2 - 1,  # dinov2 vitb14 num_layers=12
        #     num_layers // 2 + 1,
        #     num_layers // 4 * 3,
        #     num_layers - 1,
        # ]
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

        self.dpt = DPT(self.feat_dim, output_channels)

        # define layer name (for logging)
        self.layer = "-".join(str(_x) for _x in self.multilayers)

        def tokens_to_output(output_type, dense_tokens, cls_token, feat_hw):
            if output_type == "cls":
                assert cls_token is not None
                output = cls_token
            elif output_type == "gap":
                output = dense_tokens.mean(dim=1)
            elif output_type == "dense":
                h, w = feat_hw
                dense_tokens = E.rearrange(dense_tokens, "b (h w) c -> b c h w", h=h, w=w)
                output = dense_tokens.contiguous()
            elif output_type == "dense-cls":
                assert cls_token is not None
                h, w = feat_hw
                dense_tokens = E.rearrange(dense_tokens, "b (h w) c -> b c h w", h=h, w=w)
                cls_token = cls_token[:, :, None, None].repeat(1, 1, h, w)
                output = torch.cat((dense_tokens, cls_token), dim=1).contiguous()
            else:
                raise ValueError()

            return output
        
        

    def forward(self, images):

        images = center_padding(images, self.patch_size)
        h, w = images.shape[-2:]
        h, w = h // self.patch_size, w // self.patch_size

        if self.model_name == "dinov2":
            x = self.vit.prepare_tokens_with_masks(images, None)
        else:
            x = self.vit.prepare_tokens(images)

        embeds = []
        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if i in self.multilayers:
                embeds.append(x)
                if len(embeds) == len(self.multilayers):
                    break

        num_spatial = h * w
        outputs = []
        for i, x_i in enumerate(embeds):
            cls_tok = x_i[:, 0]
            # ignoring register tokens
            spatial = x_i[:, -1 * num_spatial :]
            x_i = tokens_to_output(self.output, spatial, cls_tok, (h, w))
            outputs.append(x_i)

        res = self.dpt(outputs)
        x = F.interpolate(res, size=(16,16), mode='bilinear', align_corners=True)

        # x = outputs[0] # shape (16, 768, 37,37)
        # x = self.flatten(x)  # 形状变为 [16, 768, 1024]
        # x = x.permute(0, 2, 1)  # 形状变为 [16, 1024, 768] 以便应用 MLP
        # x = self.mlp(x)  # 形状变为 [16, 1024, 320]
        # x = x.permute(0, 2, 1)  # 形状变为 [16, 320, 1024]
        #
        # x = self.unflatten(x)  # 形状变为 [16, 320, 16, 16]
        # x = self.conv_down(x)
        # x = self.relu(x)
        # x = self.conv_down(x)
        # x = self.relu(x)
        # x = F.interpolate(x, size=(16, 16), mode='bilinear', align_corners=True)
        # print(f"dino feat shape: {x.shape}")


        return outputs[0], x

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