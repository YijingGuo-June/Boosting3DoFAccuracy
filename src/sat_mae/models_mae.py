# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

"""
让我详细分析 518×518 输入，patch_size=14 时的维度变化：

1. **输入图像**：
```python
input_size = [B, 3, 518, 518]  # B是batch_size
```

2. **Patch Embedding**：
```python
# 计算patch数量
H = W = 518
patch_size = 14
num_patches = (H // patch_size) * (W // patch_size)  # 37 * 37 = 1,369 patches
# 注意：518/14 ≈ 37.0，最后一些像素会被裁剪掉

# PatchEmbed 输出
patch_tokens = [B, 1369, embed_dim]  # embed_dim 通常是 384 (ViT-Small)
```

3. **Encoder处理**：
```python
# 1. 添加位置编码
x = patch_tokens + pos_embed  # [B, 1369, 384]

# 2. 随机mask (假设mask_ratio=0.75)
num_keep = int(1369 * (1-0.75))  # = 342 tokens保留
x_masked = [B, 342, 384]  # 只保留25%的patches

# 3. 添加cls_token
x = torch.cat([cls_token, x_masked], dim=1)  # [B, 343, 384]

# 4. Transformer blocks处理
# 维度保持不变
encoder_output = [B, 343, 384]
```

4. **Decoder处理**：
```python
# 1. 投影到decoder维度
x = self.decoder_embed(encoder_output)  # [B, 343, decoder_embed_dim]

# 2. 移除cls_token并添加mask tokens
visible_tokens = x[:, 1:, :]  # [B, 342, decoder_embed_dim]
mask_tokens = self.mask_token.repeat(...)  # [B, 1027, decoder_embed_dim]
# 1027 = 1369 - 342 (被mask的数量)

# 3. 合并并还原顺序
x = torch.cat([visible_tokens, mask_tokens], dim=1)  # [B, 1369, decoder_embed_dim]
x = gather操作还原顺序  # [B, 1369, decoder_embed_dim]

# 4. Transformer blocks处理
# 维度保持不变

# 5. 最后的预测层
x = self.decoder_pred(x)  # [B, 1369, 14*14*3]
# 14*14*3 = 588 是每个patch的像素值
```

5. **Unpatchify**：
```python
# 输入: [B, 1369, 588]

# 1. 重塑为patch网格
h = w = 37  # sqrt(1369)
x = x.reshape(B, h, w, patch_size, patch_size, 3)  # [B, 37, 37, 14, 14, 3]

# 2. 调整维度顺序
x = torch.einsum('nhwpqc->nchpwq', x)  # [B, 3, 37, 14, 37, 14]

# 3. 合并为最终图像
output = x.reshape(B, 3, 37*14, 37*14)  # [B, 3, 518, 518]
```

注意事项：
1. 实际输出是 518×518（37×14 = 518）
2. 如果输入不能被patch_size整除，通常会：
   - 要么在输入时进行padding
   - 要么在边缘裁剪掉一些像素
3. 整个过程中的关键维度变化：
   - 518×518 → 1369个patch tokens
   - 1369 → 342（mask后）
   - 342 → 1369（重建）
   - 1369 → 518×518（unpatchify）

这就是为什么通常建议使用能被patch_size整除的输入尺寸，以避免信息损失。

"""

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from src.sat_mae.util.pos_embed import get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_from_grid


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
    ):
        super().__init__()

        self.in_c = in_chans

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    # qk_scale=None,
                    norm_layer=norm_layer,
                    init_values=1e-5
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    # qk_scale=None,
                    norm_layer=norm_layer,
                    init_values=None # decoder部分不使用layer Scale，保持原始 MAE 的设计，专注于重建任务
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        grid_size = self.patch_embed.grid_size  # (23, 46)
        # pos_embed = get_2d_sincos_pos_embed_from_grid(
        #     self.pos_embed.shape[-1],
        #     grid_size,
        #     cls_token=True,
        # )
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            # int(self.patch_embed.num_patches**0.5),
            grid_size,
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            # int(self.patch_embed.num_patches**0.5),
            grid_size,
            cls_token=True,
        )
        # decoder_pos_embed = get_2d_sincos_pos_embed_from_grid(
        #     self.decoder_pos_embed.shape[-1],
        #     grid_size,
        #     cls_token=True,
        # )   
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs, p, c):
        """
        imgs: (N, C, H, W)
        p: Patch embed patch size
        c: Num channels
        x: (N, L, patch_size**2 *C)
        """
        # p = self.patch_embed.patch_size[0]
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        # 移除方形检查，分别计算高度和宽度的patch数
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        # c = self.in_c
        # h = w = imgs.shape[2] // p
        h = imgs.shape[2] // p  # 高度方向的patch数
        w = imgs.shape[3] // p  # 宽度方向的patch数
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
        return x

    def unpatchify(self, x, p, c):
        """
        x: (N, L, patch_size**2 *C)
        p: Patch embed patch size
        c: Num channels
        imgs: (N, C, H, W)
        """
        # c = self.in_c
        # p = self.patch_embed.patch_size[0]
        h = self.patch_embed.grid_size[0]
        w = self.patch_embed.grid_size[1]
        # h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask, on_all_patches=False):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        # target = imgs[:, :3, :, :]
        # pred = self.unpatchify(pred, self.patch_embed.patch_size[0], self.in_c)
        # pred = self.patchify(pred[:, :3, :, :], self.patch_embed.patch_size[0], 3)
        # target = self.patchify(target, self.patch_embed.patch_size[0], 3)
        target = self.patchify(imgs, self.patch_embed.patch_size[0], self.in_c)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        if not on_all_patches:
            # compute loss only on masked patches
            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        else:
            # average over all patches (including visible)
            loss = loss.mean()
        return loss

    def forward(self, imgs, mask_ratio=0.75, on_all_patches=False):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask, on_all_patches=on_all_patches)
        return loss, pred, mask

    def forward_features(self, imgs):
        latent, _, _ = self.forward_encoder(imgs, mask_ratio=0.0)
        latent = latent[:, 1:]  # remove cls token

        # latent = latent.mean(dim=1) # avg pool
        return latent


def mae_dinov2_vits14(**kwargs):
    model = MaskedAutoencoderViT(
        # patch_size=14,
        embed_dim=384,
        depth=12,
        num_heads=6,
        decoder_embed_dim=384,
        decoder_depth=12,
        decoder_num_heads=6,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        embed_dim=1280,
        depth=32,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
