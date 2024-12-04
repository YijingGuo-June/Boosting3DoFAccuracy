import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from downsamplers import SimpleDownsampler
from SLR import add_extra_weights, ScaledLowRankAdapter, ScaledLowRankConvAdapter, ScaledLowRankConfigTimmViT

class DINOv2Featurizer(nn.Module):
    def __init__(self, arch='vits14', patch_size=14, feat_type='patch', use_adapter=True):
        super().__init__()
        self.arch = arch
        self.patch_size = patch_size
        self.feat_type = feat_type

        # 设置模型缓存目录
        torch.hub.set_dir('/home/yijing/workspace/torch_cache')
        # 加载能输出高分辨率特征的模型
        self.model = torch.hub.load("mhamilton723/FeatUp", 'dinov2', trust_repo=True)
        for param in self.model.parameters():
            param.requires_grad = False

        self.use_adapter = use_adapter

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

        # 创建三个下采样器，同时进行空间和通道的降维
        self.downsamplers = nn.ModuleList([
            # level 0: 512->64, 384->256
            SimpleDownsampler(
                in_dim=384, 
                out_dim=256,  # VGG level 0输出通道数
                kernel_size=8,
                final_size=64
            ),
            # level 1: 512->128, 384->128
            SimpleDownsampler(
                in_dim=384,
                out_dim=128,  # VGG level 1输出通道数
                kernel_size=4,
                final_size=128
            ),
            # level 2: 512->256, 384->64
            SimpleDownsampler(
                in_dim=384,
                out_dim=64,   # VGG level 2输出通道数
                kernel_size=2,
                final_size=256
            )
        ])

        for downsampler in self.downsamplers:
            add_extra_weights(
                downsampler,
                config,
                adapter=ScaledLowRankAdapter,
                conv_adapter=ScaledLowRankConvAdapter,
                trainable=True,
                only_scaler_trainable=False
            )
        
        # 添加三个尺度的置信度预测头
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

        # 添加填充到[514, 514]
        x = self.center_padding(x)
        # 获取高分辨率特征图
        hr_feats = self.model(x)  # [B, 384, 512, 512]

        # 裁剪回原始大小
        hr_feats = hr_feats[:, :, pad_t:pad_t+512, pad_l:pad_l+512]
        
        # 生成多尺度特征
        sat_feat_list = []
        sat_conf_list = []
        
        # 使用下采样器生成多尺度特征
        # level 0: H/8 x W/8
        feat0 = self.downsamplers[0](hr_feats, None)  # [B, 256, 64, 64]
        sat_feat_list.append(feat0)
        sat_conf_list.append(self.conf_heads[0](feat0))
        
        # level 1: H/4 x W/4
        feat1 = self.downsamplers[1](hr_feats, None)  # [B, 128, 128, 128]
        sat_feat_list.append(feat1)
        sat_conf_list.append(self.conf_heads[1](feat1))
        
        # level 2: H/2 x W/2
        feat2 = self.downsamplers[2](hr_feats, None)  # [B, 64, 256, 256]
        sat_feat_list.append(feat2)
        sat_conf_list.append(self.conf_heads[2](feat2))

        return sat_feat_list, sat_conf_list
    


class Encoder(nn.Module):
    def __init__(self, use_adapter=True):
        super().__init__()
        
        # 加载DINOv2主干网络
        torch.hub.set_dir('/home/yijing/workspace/torch_cache')
        self.backbone = torch.hub.load("mhamilton723/FeatUp", 'dinov2', trust_repo=True)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.patch_size = 14
        

        # 创建三个下采样器，对齐VGG的特征维度和尺寸
        self.downsamplers = nn.ModuleList([
            # x15: [B, 256, H/8, W/8] (40, 80)
            SimpleDownsampler(
                in_dim=384,
                out_dim=256,  # VGG x15的通道数
                kernel_size=8,
                final_size=(40, 80)  # 320/8=40, 640/8=80
            ),
            # x8: [B, 128, H/4, W/4] (80, 160)
            SimpleDownsampler(
                in_dim=384,
                out_dim=128,  # VGG x8的通道数
                kernel_size=4,
                final_size=(80, 160)  # 320/4=80, 640/4=160
            ),
            # x3: [B, 64, H/2, W/2] (160, 320)
            SimpleDownsampler(
                in_dim=384,
                out_dim=64,   # VGG x3的通道数
                kernel_size=2,
                final_size=(160, 320)  # 320/2=160, 640/2=320
            )
        ])

        if use_adapter:
            config = ScaledLowRankConfigTimmViT()
            
            # 为backbone添加适配器
            add_extra_weights(
                self.backbone,
                config,
                adapter=ScaledLowRankAdapter,
                conv_adapter=ScaledLowRankConvAdapter,
                trainable=True
            )
            
            # 为downsamplers添加适配器
            for downsampler in self.downsamplers:
                add_extra_weights(
                    downsampler,
                    config,
                    adapter=ScaledLowRankAdapter,
                    conv_adapter=ScaledLowRankConvAdapter,
                    trainable=True
                )

    def center_padding(self, x):
        _, _, h, w = x.shape
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        
        pad_t = pad_h // 2
        pad_b = pad_h - pad_t
        pad_l = pad_w // 2
        pad_r = pad_w - pad_l
        
        return F.pad(x, (pad_l, pad_r, pad_t, pad_b))

    def forward(self, x):  # x: [B, 3, 320, 640]
        # 计算填充值
        _, _, h, w = x.shape
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        
        pad_t = pad_h // 2
        pad_b = pad_h - pad_t
        pad_l = pad_w // 2
        pad_r = pad_w - pad_l

        # 添加填充到[322, 644]
        x = self.center_padding(x)
        
        # 获取高分辨率特征图
        hr_feats = self.backbone(x)  # [B, 384, 322, 644]
        
        # 裁剪回原始大小
        hr_feats = hr_feats[:, :, pad_t:pad_t+320, pad_l:pad_l+640]  # [B, 384, 320, 640]
        
        # 生成多尺度特征
        # 使用下采样器生成与VGG对齐的特征
        x15 = self.downsamplers[0](hr_feats, None)  # [B, 256, 40, 80]   H/8, W/8
        x8 = self.downsamplers[1](hr_feats, None)   # [B, 128, 80, 160]  H/4, W/4
        x3 = self.downsamplers[2](hr_feats, None)   # [B, 64, 160, 320]  H/2, W/2

        return x15, x8, x3  # 与VGG Encoder的输出完全对齐
