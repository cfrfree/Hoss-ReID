# model/backbones/vit_transoss_simsiam.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

# 从您现有的 vit_transoss.py 中导入必要的模块
from .vit_transoss import TransOSS, trunc_normal_


class SimSiam(nn.Module):
    """
    基于TransOSS的跨模态SimSiam模型
    """

    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg

        # --- 1. 编码器 (Encoder) ---
        # 直接复用您已经写好的TransOSS作为编码器
        self.encoder = TransOSS(
            img_size=cfg.INPUT.SIZE_TRAIN,
            stride_size=cfg.MODEL.STRIDE_SIZE,
            drop_path_rate=cfg.MODEL.DROP_PATH,
            drop_rate=cfg.MODEL.DROP_OUT,
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
            camera=2,  # MIE embedding for Optical and SAR
            mie_coe=cfg.MODEL.MIE_COE,
            sse=cfg.MODEL.SSE,
            embed_dim=cfg.MODEL.SIMSIAM_DIM,
            depth=cfg.MODEL.SIMSIAM_ENCODER_DEPTH,
            num_heads=cfg.MODEL.SIMSIAM_ENCODER_HEADS,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        encoder_dim = self.encoder.embed_dim

        # --- 2. 投影头 (Projection Head) ---
        # 将编码器输出的特征映射到新的空间
        proj_hidden_dim = cfg.MODEL.SIMSIAM_PROJ_HIDDEN_DIM
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, encoder_dim, bias=False),
            nn.BatchNorm1d(encoder_dim, affine=False),
        )

        # --- 3. 预测头 (Prediction Head) ---
        # 这是SimSiam的核心，只在一个分支上使用
        pred_hidden_dim = cfg.MODEL.SIMSIAM_PRED_HIDDEN_DIM
        self.predictor = nn.Sequential(
            nn.Linear(encoder_dim, pred_hidden_dim, bias=False),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hidden_dim, encoder_dim),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x_optical, x_sar):
        """
        输入光学和SAR图像对
        """
        # 编码特征
        f_opt = self.encoder(x_optical, cam_label=torch.zeros(x_optical.shape[0], dtype=torch.long, device=x_optical.device))
        f_sar = self.encoder(x_sar, cam_label=torch.ones(x_sar.shape[0], dtype=torch.long, device=x_sar.device))

        # 投影特征
        z_opt = self.projector(f_opt)
        z_sar = self.projector(f_sar)

        # 预测
        p_opt = self.predictor(z_opt)
        p_sar = self.predictor(z_sar)

        # 计算对称损失，关键在于stop-gradient
        loss = -(F.cosine_similarity(p_opt, z_sar.detach(), dim=-1).mean() + F.cosine_similarity(p_sar, z_opt.detach(), dim=-1).mean()) * 0.5

        return loss
