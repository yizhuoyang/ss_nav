#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random

import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models


class DepthResNet18Encoder(nn.Module):
    def __init__(self, out_dim=256, pretrained=True):
        super().__init__()

        try:
            backbone = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
        except Exception:
            backbone = models.resnet18(pretrained=pretrained)

        old_conv1 = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=old_conv1.bias is not None,
        )

        if pretrained:
            with torch.no_grad():
                backbone.conv1.weight[:] = old_conv1.weight.mean(dim=1, keepdim=True)

        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  
        self.out_dim = out_dim
        self.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        """
        x: (B, 1, 128, 128)
        """
        x = self.features(x)              # (B, 512, H', W')
        x = self.global_pool(x)           # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)         # (B, 512)
        x = self.fc(x)                    # (B, out_dim)
        return x

class RGBResNet18Encoder(nn.Module):
    def __init__(self, out_dim=256, pretrained=True):
        super().__init__()

        try:
            backbone = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
        except Exception:
            backbone = models.resnet18(pretrained=pretrained)

        old_conv1 = backbone.conv1
        backbone.conv1 =old_conv1
        if pretrained:
            with torch.no_grad():
                backbone.conv1.weight[:] = old_conv1.weight.mean(dim=1, keepdim=True)

        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  
        self.out_dim = out_dim
        self.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        """
        x: (B, 1, 128, 128)
        """
        x = self.features(x)              # (B, 512, H', W')
        x = self.global_pool(x)           # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)         # (B, 512)
        x = self.fc(x)                    # (B, out_dim)
        return x

class SpecEncoderGlobal(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        channels=(16, 32, 64),
        dropout: float = 0.1,
        out_dim: int = 256,
        use_compress: bool = True,
    ):
        super().__init__()
        self.use_compress = use_compress
        self.out_dim = out_dim
        if use_compress:
            conv_channels = channels
        else:
            conv_channels = (16, 32, 64, 64, 64)

        conv_blocks = []
        prev_c = in_channels

        for ch in conv_channels:
            block = nn.Sequential(
                nn.Conv2d(prev_c, ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),  # H,W 都 /2
                nn.Dropout2d(dropout),
            )
            conv_blocks.append(block)
            prev_c = ch

        self.conv = nn.Sequential(*conv_blocks)
        flattened_dim = prev_c * 8 * 3
        self.fc = nn.Linear(flattened_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        if self.use_compress:
            assert x.shape[2] == 65 and x.shape[3] == 26, \
                f"expect (B,2,65,26) when use_compress=True, got {tuple(x.shape)}"
        else:
            assert x.shape[2] == 257 and x.shape[3] == 101, \
                f"expect (B,2,257,101) when use_compress=False, got {tuple(x.shape)}"
        x = self.conv(x)          # -> (B, C_last, 8, 3)
        x = x.contiguous().view(B, -1)         # -> (B, C_last*8*3)
        x = self.fc(x)            # -> (B, out_dim)
        return x

class AudioGoalPredictor(nn.Module):
    def __init__(self, predict_label=True, predict_location=True):
        super(AudioGoalPredictor, self).__init__()
        self.input_shape_printed = False

        self.spec_encoder = SpecEncoderGlobal(
            in_channels=2,
            channels=(16, 32, 64),
            dropout=0.3,
            out_dim=256,
            use_compress=False
        )

        self.depth_encoder = DepthResNet18Encoder(
            out_dim=64,
            pretrained=True,
        )

        self.rgb_encoder = RGBResNet18Encoder(
            out_dim=64,
            pretrained=True,
        )

        # Freeze encoders (feature extractors)
        for p in self.depth_encoder.features.parameters():
            p.requires_grad = False
        for p in self.rgb_encoder.features.parameters():
            p.requires_grad = False

        # ===== Two FiLMs: (depth -> spec) and (rgb -> spec) =====
        self.film_gamma_d = nn.Linear(64, 256)
        self.film_beta_d  = nn.Linear(64, 256)

        self.film_gamma_r = nn.Linear(64, 256)
        self.film_beta_r  = nn.Linear(64, 256)

        # ===== Audio-conditioned gating: decide trust depth vs rgb =====
        # gate in (0,1), scalar per sample (can also make it 256-dim if you want)
        self.gate_mlp = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        self.fusion_fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.doa_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 360),
        )

        self.distance_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 120),
        )

        # Modality dropout to avoid over-reliance
        self.drop_depth_prob = 0.5
        self.drop_rgb_prob   = 0.5   # 可调：RGB更“强”，建议小一点

    @staticmethod
    def _to_tensor(x, device):
        # x: numpy or tensor
        if torch.is_tensor(x):
            return x.to(device=device)
        return torch.from_numpy(x).to(device=device)

    def forward(self, audio_feature):
        if not self.input_shape_printed:
            logging.info(f'Audiogoal predictor input spectrogram shape: {audio_feature["spectrogram"].shape}')
            logging.info(f'Audiogoal predictor input depth shape: {audio_feature["depth"].shape}')
            if "rgb" in audio_feature:
                logging.info(f'Audiogoal predictor input rgb shape: {audio_feature["rgb"].shape}')
            self.input_shape_printed = True

        device = next(self.parameters()).device

        # ---- inputs ----
        audio_obs = audio_feature["spectrogram"]  # expected: (B?, H, W, C=2) or numpy
        depth_obs = audio_feature["depth"]        # expected: (B?, 1, H, W) or similar
        rgb_obs   = audio_feature.get("rgb", None)

        audio_obs = self._to_tensor(audio_obs, device)
        depth_obs = self._to_tensor(depth_obs, device)
        if rgb_obs is not None:
            rgb_obs = self._to_tensor(rgb_obs, device)

        # add batch dim if needed (your original code did unsqueeze(0))
        if audio_obs.dim() == 3:
            audio_obs = audio_obs.unsqueeze(0)
        if depth_obs.dim() == 3:
            depth_obs = depth_obs.unsqueeze(0)
        if rgb_obs is not None and rgb_obs.dim() == 3:
            rgb_obs = rgb_obs.unsqueeze(0)

        # spectrogram: (B, H, W, C) -> (B, C, H, W)
        if audio_obs.dim() == 4 and audio_obs.shape[-1] in (1, 2, 4):
            audio_obs = audio_obs.permute(0, 3, 1, 2).contiguous()

        # ---- encoders ----
        spec_feat  = self.spec_encoder(audio_obs)      # (B, 256)
        depth_feat = self.depth_encoder(depth_obs)     # (B, 64)

        if rgb_obs is None:
            # 如果暂时没有rgb输入，也能跑：退化为仅depth FiLM
            rgb_feat = torch.zeros_like(depth_feat)
        else:
            rgb_feat = self.rgb_encoder(rgb_obs)       # (B, 64)

        # ---- modality dropout (train only) ----
        if self.training:
            if torch.rand(1).item() < self.drop_depth_prob:
                depth_feat = torch.zeros_like(depth_feat)
            if torch.rand(1).item() < self.drop_rgb_prob:
                rgb_feat = torch.zeros_like(rgb_feat)

        # ---- FiLM: depth -> spec ----
        gamma_d = self.film_gamma_d(depth_feat)   # (B,256)
        beta_d  = self.film_beta_d(depth_feat)    # (B,256)

        gamma_d = 1.0 + 0.1 * torch.tanh(gamma_d)
        beta_d  = 0.1 * torch.tanh(beta_d)
        spec_film_d = gamma_d * spec_feat + beta_d

        # ---- FiLM: rgb -> spec ----
        gamma_r = self.film_gamma_r(rgb_feat)     # (B,256)
        beta_r  = self.film_beta_r(rgb_feat)      # (B,256)

        gamma_r = 1.0 + 0.1 * torch.tanh(gamma_r)
        beta_r  = 0.1 * torch.tanh(beta_r)
        spec_film_r = gamma_r * spec_feat + beta_r

        # ---- Audio-conditioned gate (scalar) ----
        gate = torch.sigmoid(self.gate_mlp(spec_feat))  # (B,1)
        spec_film = gate * spec_film_d + (1.0 - gate) * spec_film_r  # broadcast over 256

        # ---- head ----
        out_feat = self.fusion_fc(spec_film)

        doa_logits = torch.sigmoid(self.doa_head(out_feat))         # (B,360)
        distance_logits = torch.sigmoid(self.distance_head(out_feat))  # (B,120)

        return doa_logits, distance_logits
