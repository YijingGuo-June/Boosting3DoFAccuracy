import torch
import einops as E
import torch.nn as nn
import torch.nn.functional as F
import src.trainers
import src.models
import src.sat_mae.models_mae

class EncoderOnly(torch.nn.Module):
    """只包含encoder部分的MAE模型"""
    def __init__(self, img_size, patch_size=14, in_chans=3):
        super().__init__()
        
        # 只保留encoder相关的组件
        model = src.sat_mae.models_mae.mae_dinov2_vits14(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            # embed_dim=embed_dim,
            # depth=depth,
            # num_heads=num_heads,
            # mlp_ratio=mlp_ratio
        )
        
        # 只复制encoder部分
        self.patch_embed = model.patch_embed
        self.cls_token = model.cls_token
        self.pos_embed = model.pos_embed
        self.blocks = model.blocks
        self.norm = model.norm
        self.mask_token = model.mask_token


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


class FrozenMultiScaleFeatures_Grd(nn.Module):
    def __init__(self, feat_dim=384):
        super().__init__()
        self.num_patches = 23  # 518/14=37.xx -> 37
        
        # 用于调整通道数的1x1卷积
        self.channel_adjust = nn.ModuleList([
            nn.Conv2d(feat_dim, 256, 1),  # x15: 最小尺度
            nn.Conv2d(feat_dim, 128, 1),  # x8: 中等尺度
            nn.Conv2d(feat_dim, 64, 1)    # x3: 最大尺度
        ])
        
        # 冻结参数
        for param in self.channel_adjust.parameters():
            param.requires_grad = False

    def reshape_tokens_to_features(self, x):
        """将tokens重组为特征图"""
        B = x.shape[0]
        return x.transpose(1, 2).reshape(B, -1, self.num_patches, self.num_patches*2)

    def forward(self, features):
        if not isinstance(features, list):
            features = [features]
            
        # 特征重组和尺度变换
        multi_scale_features = []
        for idx, feat in enumerate(features[-3:]):  # 只使用最后三层特征
            # 重组为特征图 [B, N, C] -> [B, C, H, W]
            feat = self.reshape_tokens_to_features(feat)
            
            # 调整通道数
            feat = self.channel_adjust[idx](feat)  # [B, C, 37, 37]
            
            # 设置目标尺寸
            if idx == 0:  # x15: 最小尺度
                target_size = (40, 80)     # [B, 256, 40, 80]
            elif idx == 1:  # x8: 中等尺度
                target_size = (80, 160)    # [B, 128, 80, 160]
            else:  # x3: 最大尺度
                target_size = (160, 320)   # [B, 64, 160, 320]
            
            # 上采样到目标尺寸
            feat = F.interpolate(
                feat, 
                size=target_size, 
                mode='bilinear', 
                align_corners=False
            )
            multi_scale_features.append(feat)
        
        return tuple(multi_scale_features)  # ([B,256,40,80], [B,128,80,160], [B,64,160,320])
    

class MAE_Grd(torch.nn.Module):
    def __init__(
        self,
        output="dense",
        layer=-1,
        return_multilayer=True,
        output_channels=112,
        hidden_channels1=544,
        hidden_channels2=465,
        down_sample=False,
    ):
        super().__init__()
        
        # 创建只有encoder的模型
        self.model = EncoderOnly(
            img_size=(322, 644),
            patch_size=14,
            in_chans=3
        )
        
        # 添加adapter
        self.model = src.models.add_adapter(
            self.model,
            type="low-rank-scaling",
            shared=False,
            scale=1.0, # type: ignore
            hidden_dim=16,
            patch_embed_adapter=True,
            adapter_trainable=True,
            norm_trainable=True,
            only_scaler_trainable=False,
        )
        
        # 加载预训练权重
        checkpoint = torch.load("/home/yijing/workspace/Boosting3DoFAccuracy/checkpoint/pretrain_ground.pth")
        for param in checkpoint['model'].parameters():
            param.requires_grad = False
        self.model.load_state_dict(checkpoint['model'], strict=True)
        
        # 冻结参数
        for param in self.model.parameters():
            param.requires_grad = False
            
        # # 下游处理层
        # self.flatten = nn.Flatten(2)
        # self.mlp = MLP(input_dim=384, hidden_dim=512, output_dim=320)
        # # self.unflatten = nn.Unflatten(2, (37, 37))
        # self.unflatten = nn.Unflatten(2, (23, 46))
        # self.conv_down = nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, stride=2, padding=1)
        # self.relu = nn.ReLU()
            
        # 特征提取设置
        assert output in ["cls", "gap", "dense", "dense-cls"]
        self.output = output
        self.patch_size = 14
        self.feat_dim = 384  # MAE的embedding维度
        
        # 多层特征提取设置
        num_layers = len(self.model.blocks)
        self.multilayers = [
            num_layers // 4 - 1,
            num_layers // 2 - 1,
            num_layers // 4 * 3 - 1,
        ]
        
        # 替换DPT为MultiScale
        self.multiscale = FrozenMultiScaleFeatures_Grd(feat_dim=self.feat_dim)

        # 设置参数的可训练状态
        self.set_trainable_params()
        
        # 定义层名（用于日志）
        self.layer = "-".join(str(_x) for _x in self.multilayers)

    def set_trainable_params(self):
        """设置参数的可训练状态"""
        # 1. 冻结主干网络
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 2. 激活需要训练的参数
        for name, param in self.model.named_parameters():
            # adapter参数
            if any(x in name for x in ['up', 'down', 'scaler']):
                param.requires_grad = True
            # norm层参数 (包括norm1和norm2)
            elif any(x in name for x in ['norm1', 'norm2', 'norm.']):
                param.requires_grad = True
            # token相关参数
            elif any(x in name for x in ['cls_token', 'pos_embed', 'mask_token']):
                param.requires_grad = True
            # backbone参数保持冻结
            else:
                param.requires_grad = False
        
        # 3. 冻结channel_adjust
        for param in self.multiscale.channel_adjust.parameters():
            param.requires_grad = True
        
        # # 4. 确保conf_heads可训练
        # for param in self.multiscale.conf_heads.parameters():
        #     param.requires_grad = True
            
        # # 5. 确保下游任务层可训练
        # for param in self.downstream.parameters():
        #     param.requires_grad = True

    def print_trainable_params(self):
        """打印可训练参数的信息和总数"""
        print("\n=== Trainable Parameters ===")
        total_params = 0
        adapter_params = 0
        norm_params = 0
        token_params = 0
        channel_adjust_params = 0
        conf_params = 0
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.shape}")
                # 统计adapter参数
                if any(x in name for x in ['up', 'down', 'scaler']):
                    adapter_params += param.numel()
                # 统计norm参数
                elif any(x in name for x in ['norm1', 'norm2', 'norm.']):
                    norm_params += param.numel()
                # 统计token相关参数
                elif any(x in name for x in ['cls_token', 'pos_embed', 'mask_token']):
                    token_params += param.numel()
                # 统计channel_adjust参数
                elif 'channel_adjust' in name:
                    channel_adjust_params += param.numel()
                # 统计conf_heads参数
                # elif 'conf_heads' in name:
                #     conf_params += param.numel()
                total_params += param.numel()
        
        print(f"\nAdapter parameters: {adapter_params:,}")
        print(f"Norm parameters: {norm_params:,}")
        print(f"Token parameters: {token_params:,}")
        print(f"Channel adjust parameters: {channel_adjust_params:,}")
        # print(f"Confidence head parameters: {conf_params:,}")
        print(f"Total trainable parameters: {total_params:,}")
        print("==========================\n")


    def tokens_to_output(self, output_type, dense_tokens, cls_token, feat_hw):
        """将tokens转换为所需的输出格式"""
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
            raise ValueError(f"Unsupported output type: {output_type}")
        return output

    def center_padding(self, x):
        """对输入图像进行中心填充"""
        _, _, h, w = x.shape
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        
        pad_t = pad_h // 2
        pad_b = pad_h - pad_t
        pad_l = pad_w // 2
        pad_r = pad_w - pad_l
        
        return F.pad(x, (pad_l, pad_r, pad_t, pad_b))

    def forward(self, images):
        # 中心填充
        images = self.center_padding(images)
        # print(images.shape)
        h, w = images.shape[-2:]
        h, w = h // self.patch_size, w // self.patch_size
        # print(h, w)
        
        # 获取patch embeddings
        x = self.model.patch_embed(images)
        print(x.shape)
        
        # 添加cls token和position embedding
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
        # print(cls_token.shape)
        x = torch.cat((cls_token, x), dim=1)
        # print(x.shape)
        x = x + self.model.pos_embed
        # print(x.shape)
        
        # 通过transformer blocks提取特征
        embeds = []
        for i, blk in enumerate(self.model.blocks):
            x = blk(x)
            # print(i)
            # print(x.shape)
            if i in self.multilayers:
                embeds.append(x)
                
        
        # 处理提取的特征
        num_spatial = h * w
        outputs = []
        for i, x_i in enumerate(embeds):
            cls_tok = x_i[:, 0]
            spatial = x_i[:, 1:num_spatial+1]  # 不包括cls token
            x_i = self.tokens_to_output(self.output, spatial, cls_tok, (h, w))
            outputs.append(x_i)
        
        # 使用MultiScale处理特征
        multi_scale_features = self.multiscale(outputs)
        
        return multi_scale_features  # 直接返回多尺度特征
    


class FrozenMultiScaleFeatures_Sat(nn.Module):
    def __init__(self, feat_dim=384):
        super().__init__()
        self.num_patches = 37  # 518/14=37.xx -> 37
        
        # 用于调整通道数的1x1卷积
        self.channel_adjust = nn.ModuleList([
            nn.Conv2d(feat_dim, 256, 1),  # for 64x64
            nn.Conv2d(feat_dim, 128, 1),  # for 128x128
            nn.Conv2d(feat_dim, 64, 1)    # for 256x256
        ])

        # 添加置信度预测头
        self.conf_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 1, 1),  # 最小尺度
                nn.Sigmoid()
            ),
            nn.Sequential(
                nn.Conv2d(128, 1, 1),  # 中等尺度
                nn.Sigmoid()
            ),
            nn.Sequential(
                nn.Conv2d(64, 1, 1),   # 最大尺度
                nn.Sigmoid()
            )
        ])
        
        # 冻结参数
        for param in self.channel_adjust.parameters():
            param.requires_grad = False
        for param in self.conf_heads.parameters():
            param.requires_grad = False

    def reshape_tokens_to_features(self, x):
        """将tokens重组为特征图"""
        B = x.shape[0]
        return x.transpose(1, 2).reshape(B, -1, self.num_patches, self.num_patches)

    def forward(self, features):
        if not isinstance(features, list):
            features = [features]
            
        # 特征重组和尺度变换
        multi_scale_features = []
        confidence_scores = []
        for idx, feat in enumerate(features[-3:]):  # 只使用最后三层特征
            # 重组为特征图 [B, N, C] -> [B, C, H, W]
            feat = self.reshape_tokens_to_features(feat)
            
            # 调整通道数
            feat = self.channel_adjust[idx](feat)  # [B, C, 37, 37]
            
            # 设置目标尺寸
            if idx == 0:  # 256维特征
                target_size = (64, 64)
            elif idx == 1:  # 128维特征
                target_size = (128, 128)
            else:  # 64维特征
                target_size = (256, 256)
            
            # 上采样到目标尺寸
            feat = F.interpolate(
                feat, 
                size=target_size, 
                mode='bilinear', 
                align_corners=False
            )
            # 计算置信度
            conf = self.conf_heads[idx](feat)
            multi_scale_features.append(feat)
            confidence_scores.append(conf)
        
        return tuple(multi_scale_features), tuple(confidence_scores)  # ([B,256,64,64], [B,128,128,128], [B,64,256,256])
    

class MAE_Sat(torch.nn.Module):
    def __init__(
        self,
        output="dense",
        layer=-1,
        return_multilayer=True,
        output_channels=112,
        hidden_channels1=544,
        hidden_channels2=465,
        down_sample=False,
    ):
        super().__init__()
        
        # 创建只有encoder的模型
        self.model = EncoderOnly(
            img_size=(518, 518),
            patch_size=14,
            in_chans=3
        )
        
        # 添加adapter
        self.model = src.models.add_adapter(
            self.model,
            type="low-rank-scaling",
            shared=False,
            scale=1.0, # type: ignore
            hidden_dim=16,
            patch_embed_adapter=True,
            adapter_trainable=True,
            norm_trainable=True,
            only_scaler_trainable=False,
        )
        
        # 加载预训练权重
        checkpoint = torch.load("/home/yijing/workspace/Boosting3DoFAccuracy/checkpoint/pretrain_sat.pth")
        self.model.load_state_dict(checkpoint['model'], strict=True)
        for param in self.model.parameters():
            param.requires_grad = False
            
        # # 下游处理层
        # self.flatten = nn.Flatten(2)
        # self.mlp = MLP(input_dim=384, hidden_dim=512, output_dim=320)
        # # self.unflatten = nn.Unflatten(2, (37, 37))
        # self.unflatten = nn.Unflatten(2, (37, 37))
        # self.conv_down = nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, stride=2, padding=1)
        # self.relu = nn.ReLU()
            
        # 特征提取设置
        assert output in ["cls", "gap", "dense", "dense-cls"]
        self.output = output
        self.patch_size = 14
        self.feat_dim = 384  # MAE的embedding维度
        
        # 多层特征提取设置
        num_layers = len(self.model.blocks)
        self.multilayers = [
            num_layers // 4 - 1,
            num_layers // 2 - 1,
            num_layers // 4 * 3 - 1,
        ]
        
        # 替换DPT为MultiScale
        self.multiscale = FrozenMultiScaleFeatures_Sat(feat_dim=self.feat_dim)

        # 设置参数的可训练状态
        self.set_trainable_params()
        
        # 定义层名（用于日志）
        self.layer = "-".join(str(_x) for _x in self.multilayers)

    def set_trainable_params(self):
        """设置参数的可训练状态"""
        # 1. 冻结主干网络
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 2. 激活需要训练的参数
        for name, param in self.model.named_parameters():
            # adapter参数
            if any(x in name for x in ['up', 'down', 'scaler']):
                param.requires_grad = True
            # norm层参数 (包括norm1和norm2)
            elif any(x in name for x in ['norm1', 'norm2', 'norm.']):
                param.requires_grad = True
            # token相关参数
            elif any(x in name for x in ['cls_token', 'pos_embed', 'mask_token']):
                param.requires_grad = True
            # backbone参数保持冻结
            else:
                param.requires_grad = False
        
        # 3. 冻结channel_adjust
        for param in self.multiscale.channel_adjust.parameters():
            param.requires_grad = True
        
        # 4. 确保conf_heads可训练
        for param in self.multiscale.conf_heads.parameters():
            param.requires_grad = True
            
        # # 5. 确保下游任务层可训练
        # for param in self.downstream.parameters():
        #     param.requires_grad = True
            
    def print_trainable_params(self):
        """打印可训练参数的信息和总数"""
        print("\n=== Trainable Parameters ===")
        total_params = 0
        adapter_params = 0
        norm_params = 0
        token_params = 0
        channel_adjust_params = 0
        conf_params = 0
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.shape}")
                # 统计adapter参数
                if any(x in name for x in ['up', 'down', 'scaler']):
                    adapter_params += param.numel()
                # 统计norm参数
                elif any(x in name for x in ['norm1', 'norm2', 'norm.']):
                    norm_params += param.numel()
                # 统计token相关参数
                elif any(x in name for x in ['cls_token', 'pos_embed', 'mask_token']):
                    token_params += param.numel()
                # 统计channel_adjust参数
                elif 'channel_adjust' in name:
                    channel_adjust_params += param.numel()
                # 统计conf_heads参数
                elif 'conf_heads' in name:
                    conf_params += param.numel()
                total_params += param.numel()
        
        print(f"\nAdapter parameters: {adapter_params:,}")
        print(f"Norm parameters: {norm_params:,}")
        print(f"Token parameters: {token_params:,}")
        print(f"Channel adjust parameters: {channel_adjust_params:,}")
        print(f"Confidence head parameters: {conf_params:,}")
        print(f"Total trainable parameters: {total_params:,}")
        print("==========================\n")


    def tokens_to_output(self, output_type, dense_tokens, cls_token, feat_hw):
        """将tokens转换为所需的输出格式"""
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
            raise ValueError(f"Unsupported output type: {output_type}")
        return output

    def center_padding(self, x):
        """对输入图像进行中心填充"""
        _, _, h, w = x.shape
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        
        pad_t = pad_h // 2
        pad_b = pad_h - pad_t
        pad_l = pad_w // 2
        pad_r = pad_w - pad_l
        
        return F.pad(x, (pad_l, pad_r, pad_t, pad_b))

    def forward(self, images):
        # 中心填充
        images = self.center_padding(images)
        # print(images.shape)
        h, w = images.shape[-2:]
        h, w = h // self.patch_size, w // self.patch_size
        # print(h, w)
        
        # 获取patch embeddings
        x = self.model.patch_embed(images)
        # print(x.shape)
        
        # 添加cls token和position embedding
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
        # print(cls_token.shape)
        x = torch.cat((cls_token, x), dim=1)
        # print(x.shape)
        x = x + self.model.pos_embed
        # print(x.shape)
        
        # 通过transformer blocks提取特征
        embeds = []
        for i, blk in enumerate(self.model.blocks):
            x = blk(x)
            # print(i)
            # print(x.shape)
            if i in self.multilayers:
                embeds.append(x)
                
        
        # 处理提取的特征
        num_spatial = h * w
        outputs = []
        for i, x_i in enumerate(embeds):
            cls_tok = x_i[:, 0]
            spatial = x_i[:, 1:num_spatial+1]  # 不包括cls token
            x_i = self.tokens_to_output(self.output, spatial, cls_tok, (h, w))
            outputs.append(x_i)
        
        # 使用MultiScale处理特征
        multi_scale_features, confidence_scores = self.multiscale(outputs)
        
        return multi_scale_features, confidence_scores  # 直接返回多尺度特征