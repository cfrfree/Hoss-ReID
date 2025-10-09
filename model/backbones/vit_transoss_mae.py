# model/backbones/vit_transoss_mae.py

import torch
import torch.nn as nn
from functools import partial

# 从您现有的 vit_transoss.py 中导入必要的模块
# 确保下面的导入路径是正确的
from .vit_transoss import TransOSS, Block, PatchEmbed_overlap, _no_grad_trunc_normal_, trunc_normal_


# =================================================================================
# 1. 辅助函数：用于随机遮盖 (Masking)
# =================================================================================
def random_masking(x, mask_ratio):
    """
    对输入的patch序列进行随机遮盖。
    x: [N, L, D], N是batch size, L是序列长度, D是维度
    """
    N, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # 噪音用于随机排序

    # 排序并获取保留和移除的ids
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # 保留的ids
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # 生成mask, 1表示移除, 0表示保留
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # 恢复到原始顺序
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


# =================================================================================
# 2. TransOSS MAE 模型定义
# =================================================================================
class TransOSS_MAE(nn.Module):
    """
    基于TransOSS的跨模态Masked Autoencoder模型
    """

    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.mask_ratio = cfg.MODEL.MAE_MASK_RATIO

        # --- 1. 编码器 (Encoder) ---
        # 直接复用您已经写好的TransOSS作为编码器
        # 注意：这里的 num_classes 可以设为0，因为预训练阶段我们不需要分类头
        self.encoder = TransOSS(
            img_size=cfg.INPUT.SIZE_TRAIN,
            stride_size=cfg.MODEL.STRIDE_SIZE,
            drop_path_rate=cfg.MODEL.DROP_PATH,
            drop_rate=cfg.MODEL.DROP_OUT,
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
            camera=2,  # MIE embedding for Optical and SAR
            mie_coe=cfg.MODEL.MIE_COE,
            sse=cfg.MODEL.SSE,
            embed_dim=cfg.MODEL.MAE_ENCODER_DIM,
            depth=cfg.MODEL.MAE_ENCODER_DEPTH,
            num_heads=cfg.MODEL.MAE_ENCODER_NUM_HEADS,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

        encoder_dim = self.encoder.embed_dim
        decoder_dim = cfg.MODEL.MAE_DECODER_DIM

        # --- 2. 解码器 (Decoder) ---
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        trunc_normal_(self.mask_token, std=0.02)

        # 解码器的position embedding需要能够容纳两个模态的所有patches
        num_patches = self.encoder.patch_embed.num_patches
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches * 2, decoder_dim))
        trunc_normal_(self.decoder_pos_embed, std=0.02)

        self.decoder_blocks = nn.ModuleList(
            [
                Block(decoder_dim, cfg.MODEL.MAE_DECODER_NUM_HEADS, mlp_ratio=4.0, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
                for _ in range(cfg.MODEL.MAE_DECODER_DEPTH)
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim)

        # --- 3. 重建头 (Prediction Head) ---
        # 输出的维度 = patch_size * patch_size * channels
        patch_size = self.encoder.patch_embed.patch_size[0]
        self.decoder_pred = nn.Linear(decoder_dim, patch_size**2 * 3, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        # 初始化解码器和预测头的权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 * 3)
        """
        p = self.encoder.patch_embed.patch_size[0]
        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq -> nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def forward_encoder(self, x_optical, x_sar):
        # 1. 将图片转换为Patches
        x_opt_patches = self.encoder.patch_embed(x_optical)
        x_sar_patches = self.encoder.patch_embed_SAR(x_sar)

        # 2. 分别进行随机遮盖
        x_opt_masked, mask_opt, ids_restore_opt = random_masking(x_opt_patches, self.mask_ratio)
        x_sar_masked, mask_sar, ids_restore_sar = random_masking(x_sar_patches, self.mask_ratio)

        # 3. 为可见Patches添加模态信息
        B, _, C = x_opt_masked.shape
        # 为光学(cam_id=0)和SAR(cam_id=1)添加MIE
        opt_mie = self.encoder.mie_embed[0].expand(B, -1, -1)
        sar_mie = self.encoder.mie_embed[1].expand(B, -1, -1)

        x_opt_masked = x_opt_masked + opt_mie
        x_sar_masked = x_sar_masked + sar_mie

        # 4. 拼接两个模态的可见Patches
        x_concat = torch.cat([x_opt_masked, x_sar_masked], dim=1)

        # 5. 添加CLS token 和 位置编码
        cls_token = self.encoder.cls_token
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x_concat), dim=1)

        # 使用部分位置编码
        pos_embed_opt = self.encoder.pos_embed[:, 1 : x_opt_masked.shape[1] + 1, :]
        pos_embed_sar = self.encoder.pos_embed[:, 1 : x_sar_masked.shape[1] + 1, :]
        pos_embed_cat = torch.cat([pos_embed_opt, pos_embed_sar], dim=1)
        x[:, 1:, :] = x[:, 1:, :] + pos_embed_cat
        x[:, :1, :] = x[:, :1, :] + self.encoder.pos_embed[:, :1, :]

        # 6. 送入Transformer编码器
        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.encoder.norm(x)

        return x, mask_opt, mask_sar, ids_restore_opt, ids_restore_sar

    def forward_decoder(self, x, ids_restore_opt, ids_restore_sar):
        # 1. 嵌入到解码器维度
        x = self.decoder_embed(x)

        # 2. 分离CLS token和Patches
        cls_token = x[:, :1, :]
        x_patches = x[:, 1:, :]  # (B, len_keep_opt + len_keep_sar, D_decoder)

        # 3. 准备mask tokens
        len_keep_opt = int(self.encoder.patch_embed.num_patches * (1 - self.mask_ratio))

        # 光学部分
        x_patches_opt = x_patches[:, :len_keep_opt, :]
        mask_tokens_opt = self.mask_token.repeat(x.shape[0], ids_restore_opt.shape[1] - len_keep_opt, 1)
        x_opt_full = torch.cat([x_patches_opt, mask_tokens_opt], dim=1)
        x_opt_full = torch.gather(x_opt_full, dim=1, index=ids_restore_opt.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        # SAR部分
        x_patches_sar = x_patches[:, len_keep_opt:, :]
        mask_tokens_sar = self.mask_token.repeat(x.shape[0], ids_restore_sar.shape[1] - len_keep_opt, 1)
        x_sar_full = torch.cat([x_patches_sar, mask_tokens_sar], dim=1)
        x_sar_full = torch.gather(x_sar_full, dim=1, index=ids_restore_sar.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        # 4. 拼接完整的序列
        x_full = torch.cat([cls_token, x_opt_full, x_sar_full], dim=1)

        # 5. 添加解码器位置编码
        x_full = x_full + self.decoder_pos_embed

        # 6. 送入解码器
        for blk in self.decoder_blocks:
            x_full = blk(x_full)
        x_full = self.decoder_norm(x_full)

        # 7. 重建头
        x_pred = self.decoder_pred(x_full)

        # 8. 分离光学和SAR的重建结果
        num_patches = self.encoder.patch_embed.num_patches
        pred_opt = x_pred[:, 1 : num_patches + 1, :]
        pred_sar = x_pred[:, num_patches + 1 :, :]

        return pred_opt, pred_sar

    def forward_loss(self, imgs_opt, imgs_sar, pred_opt, pred_sar, mask_opt, mask_sar):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target_opt = self.patchify(imgs_opt)
        target_sar = self.patchify(imgs_sar)

        loss_opt = (pred_opt - target_opt) ** 2
        loss_opt = loss_opt.mean(dim=-1)  # (N, L), mean loss per patch
        loss_opt = (loss_opt * mask_opt).sum() / mask_opt.sum()  # 只计算被mask掉的patch的loss

        loss_sar = (pred_sar - target_sar) ** 2
        loss_sar = loss_sar.mean(dim=-1)  # (N, L), mean loss per patch
        loss_sar = (loss_sar * mask_sar).sum() / mask_sar.sum()

        return loss_opt + loss_sar

    def forward(self, imgs_opt, imgs_sar):
        latent, mask_opt, mask_sar, ids_restore_opt, ids_restore_sar = self.forward_encoder(imgs_opt, imgs_sar)
        pred_opt, pred_sar = self.forward_decoder(latent, ids_restore_opt, ids_restore_sar)
        loss = self.forward_loss(imgs_opt, imgs_sar, pred_opt, pred_sar, mask_opt, mask_sar)
        return loss
