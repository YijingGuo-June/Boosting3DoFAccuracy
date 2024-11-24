import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d


class SimpleDownsampler(torch.nn.Module):
    def get_kernel(self):
        k = self.kernel_params.unsqueeze(0).unsqueeze(0).abs()
        k /= k.sum()
        return k

    def __init__(self, in_dim, out_dim, kernel_size, final_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = kernel_size
        self.final_size = final_size
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # 空间下采样的卷积核参数
        self.kernel_params = torch.nn.Parameter(torch.ones(kernel_size, kernel_size))
        
        # 通道降维的1x1卷积
        self.dim_reduction = torch.nn.Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=1,
            bias=False
        )

    def forward(self, imgs, guidance):
        b, c, h, w = imgs.shape
        
        # 1. 先进行空间下采样
        input_imgs = imgs.reshape(b * c, 1, h, w)
        stride = (h - self.kernel_size) // (self.final_size - 1)
        
        spatial_down = F.conv2d(
            input_imgs,
            self.get_kernel(),
            stride=stride
        ).reshape(b, c, self.final_size, self.final_size)
        
        # 2. 再进行通道降维
        return self.dim_reduction(spatial_down)  # [B, out_dim, final_size, final_size]
    

class Grd_SimpleDownsampler(torch.nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, final_size):
        super().__init__()
        
        # 支持final_size为元组或单个数字
        if isinstance(final_size, tuple):
            self.final_h, self.final_w = final_size
        else:
            self.final_h = self.final_w = final_size
            
        # 计算stride以达到目标尺寸
        self.stride_h = None
        self.stride_w = None
        
        self.conv = torch.nn.Conv2d(
            in_dim, 
            out_dim,
            kernel_size=kernel_size,
            stride=1,  # 先用1，后面用interpolate调整尺寸
            padding=kernel_size//2
        )
        
    def forward(self, x, mask=None):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        
        # 先做卷积
        x = self.conv(x)  # [B, out_dim, H, W]
        
        # 然后用interpolate调整到目标尺寸
        x = F.interpolate(
            x,
            size=(self.final_h, self.final_w),
            mode='bilinear',
            align_corners=True
        )
        
        return x



class AttentionDownsampler(torch.nn.Module):

    def __init__(self, dim, kernel_size, final_size, blur_attn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = kernel_size
        self.final_size = final_size
        self.in_dim = dim
        self.attention_net = torch.nn.Sequential(
            torch.nn.Dropout(p=.2),
            torch.nn.Linear(self.in_dim, 1)
        )
        self.w = torch.nn.Parameter(torch.ones(kernel_size, kernel_size).cuda()
                                    + .01 * torch.randn(kernel_size, kernel_size).cuda())
        self.b = torch.nn.Parameter(torch.zeros(kernel_size, kernel_size).cuda()
                                    + .01 * torch.randn(kernel_size, kernel_size).cuda())
        self.blur_attn = blur_attn

    def forward_attention(self, feats, guidance):
        return self.attention_net(feats.permute(0, 2, 3, 1)).squeeze(-1).unsqueeze(1)

    def forward(self, hr_feats, guidance):
        b, c, h, w = hr_feats.shape

        if self.blur_attn:
            inputs = gaussian_blur2d(hr_feats, 5, (1.0, 1.0))
        else:
            inputs = hr_feats

        stride = (h - self.kernel_size) // (self.final_size - 1)

        patches = torch.nn.Unfold(self.kernel_size, stride=stride)(inputs) \
            .reshape(
            (b, self.in_dim, self.kernel_size * self.kernel_size, self.final_size, self.final_size * int(w / h))) \
            .permute(0, 3, 4, 2, 1)

        patch_logits = self.attention_net(patches).squeeze(-1)

        b, h, w, p = patch_logits.shape
        dropout = torch.rand(b, h, w, 1, device=patch_logits.device) > 0.2

        w = self.w.flatten().reshape(1, 1, 1, -1)
        b = self.b.flatten().reshape(1, 1, 1, -1)

        patch_attn_logits = (patch_logits * dropout) * w + b
        patch_attention = F.softmax(patch_attn_logits, dim=-1)

        downsampled = torch.einsum("bhwpc,bhwp->bchw", patches, patch_attention)

        return downsampled[:, :c, :, :]
