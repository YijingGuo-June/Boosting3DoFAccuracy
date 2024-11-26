import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d


class SimpleDownsampler(torch.nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, final_size):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        
        # 支持final_size为元组或单个数字
        if isinstance(final_size, tuple):
            self.final_h, self.final_w = final_size
        else:
            self.final_h = self.final_w = final_size
            
        # 创建可学习的卷积核参数
        self.kernel_params = torch.nn.Parameter(torch.ones(kernel_size, kernel_size))
        
        # 通道变换层
        self.channel_conv = torch.nn.Conv2d(in_dim, out_dim, 1)
        
    def get_kernel(self):
        # 获取归一化的卷积核
        k = self.kernel_params.unsqueeze(0).unsqueeze(0).abs()  # [1, 1, k, k]
        k /= k.sum()  # 归一化确保权重和为1
        return k
        
    def forward(self, x, guidance=None):
        b, c, h, w = x.shape
        
        # 重塑输入以便对每个通道独立应用相同的卷积核
        input_x = x.reshape(b * c, 1, h, w)  # [b*c, 1, h, w]
        
        # 计算步长以达到目标尺寸
        stride_h = (h - self.kernel_size) // (self.final_h - 1)
        stride_w = (w - self.kernel_size) // (self.final_w - 1)
        
        # 使用可学习的卷积核进行降采样
        x = F.conv2d(
            input_x,
            self.get_kernel(),
            stride=(stride_h, stride_w)
        ).reshape(b, c, self.final_h, self.final_w)
        
        # 通道变换
        x = self.channel_conv(x)
        
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
