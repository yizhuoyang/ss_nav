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

        for p in self.depth_encoder.features.parameters():
            p.requires_grad = False

        self.film_gamma = nn.Linear(64, 256)
        self.film_beta  = nn.Linear(64, 256)

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
        # self.num_distance_bins = num_distance_bins
        self.distance_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 120),
        )
        self.drop_depth_prob = 0.5

    def forward(self, audio_feature):
        if not self.input_shape_printed:
            logging.info('Audiogoal predictor input audio feature shape: {}'.format(audio_feature["spectrogram"].shape))
            self.input_shape_printed = True
        audio_observations = audio_feature['spectrogram']
        depth_observations = audio_feature['depth']
        if not torch.is_tensor(audio_observations):
            audio_observations = torch.from_numpy(audio_observations).to(device='cuda:0').unsqueeze(0)
            depth_observations = torch.from_numpy(depth_observations).to(device='cuda:0').unsqueeze(0)

        audio_observations = audio_observations.permute(0, 3, 1, 2)
        spec_feat = self.spec_encoder(audio_observations)

        depth_feat = self.depth_encoder(depth_observations) 

        if self.training and torch.rand(1).item() < self.drop_depth_prob:
            depth_feat = torch.zeros_like(depth_feat)

        gamma = self.film_gamma(depth_feat)            # (B, spec_out_dim)
        beta  = self.film_beta(depth_feat)             # (B, spec_out_dim)

        gamma = 1.0 + 0.1 * torch.tanh(gamma)
        beta  = 0.1 * torch.tanh(beta)

        spec_film = gamma * spec_feat + beta           # (B, spec_out_dim)

        out_feat = self.fusion_fc(spec_film)           # (B, fusion_out_dim)

        # TODO
        doa_logits = self.doa_head(out_feat)
        doa_logits = torch.sigmoid(doa_logits)

        distance_logits = self.distance_head(out_feat)
        distance_logits = torch.sigmoid(distance_logits)

        return doa_logits, distance_logits

