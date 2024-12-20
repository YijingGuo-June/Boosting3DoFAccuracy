import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from downsamplers import SimpleDownsampler
from SLR import add_extra_weights, ScaledLowRankAdapter, ScaledLowRankConvAdapter, ScaledLowRankConfigTimmViT

class DINOv2(nn.Module):
    def __init__(self, arch='vits14', patch_size=14, feat_type='patch', use_adapter=True):
        super().__init__()
        self.arch = arch
        self.patch_size = patch_size
        self.feat_type = feat_type
        self.use_adapter = use_adapter

        torch.hub.set_dir('/home/yijing/workspace/torch_cache')
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', trust_repo=True)
        for param in self.model.parameters():
            param.requires_grad = False

        if use_adapter:
            config = ScaledLowRankConfigTimmViT()

        add_extra_weights(
            self.model,
            config,
            adapter=ScaledLowRankAdapter,
            conv_adapter=ScaledLowRankConvAdapter,
            trainable=True,
            only_scaler_trainable=False
        )

        self.conf_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_dim, 1, 1),  # 384是DINOv2输出的特征通道数
                nn.Sigmoid()
            ) for out_dim in [256, 128, 64]
        ])

    def center_padding(self, x):
        _, _, h, w = x.shape
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        
        pad_t = pad_h // 2
        pad_b = pad_h - pad_t
        pad_l = pad_w // 2
        pad_r = pad_w - pad_l
        
        return F.pad(x, (pad_l, pad_r, pad_t, pad_b))
    

    def forward(self, x):  # x: [B, 3, 512, 512]
        # 计算填充值
        _, _, h, w = x.shape
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        
        pad_t = pad_h // 2
        pad_b = pad_h - pad_t
        pad_l = pad_w // 2
        pad_r = pad_w - pad_l

        # 添加填充到[518, 518]
        x = self.center_padding(x)
        print('center_padding x.shape: ', x.shape)
        feat_raw = self.model(x) # [B, 384, 37, 37]
        print('feat_raw.shape: ', feat_raw.shape)
        # 获取高分辨率特征图
        hr_feats = self.model(x)  # [B, 384, 518, 518]

        # 裁剪回原始大小
        hr_feats = hr_feats[:, :, pad_t:pad_t+512, pad_l:pad_l+512]
        
        # 生成多尺度特征
        sat_feat_list = []
        sat_conf_list = []
        
        # 生成多尺度特征
        # level 0: H/8 x W/8

        
        # level 1: H/4 x W/4

        
        # level 2: H/2 x W/2


        return sat_feat_list, sat_conf_list









