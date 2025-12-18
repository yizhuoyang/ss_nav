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

#
# class AudioGoalPredictor(nn.Module):
#     def __init__(self, predict_label=True, predict_location=True):
#         super(AudioGoalPredictor, self).__init__()
#         self.input_shape_printed = False
#         self.predictor = models.resnet18(pretrained=True)
#         self.predictor.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         # output_size = (21 if predict_label else 0) + (2 if predict_location else 0)
#         self.predictor.fc = nn.Identity()
#         self.doa_head = nn.Linear(512,360)
#         self.dis_head = nn.Linear(512,120)
#         self.last_global_coords = None
#
#     def forward(self, audio_feature):
#         if not self.input_shape_printed:
#             logging.info('Audiogoal predictor input audio feature shape: {}'.format(audio_feature["spectrogram"].shape))
#             self.input_shape_printed = True
#         audio_observations = audio_feature['spectrogram']
#         if not torch.is_tensor(audio_observations):
#             audio_observations = torch.from_numpy(audio_observations).to(device='cuda:0').unsqueeze(0)
#         # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
#         audio_observations = audio_observations.permute(0, 3, 1, 2)
#         audio_feature = self.predictor(audio_observations)
#         predicted_doa = self.doa_head(audio_feature)
#         # predicted_doa = torch.sigmoid(predicted_doa)
#         predicted_dis = self.dis_head(audio_feature)
#         predicted_dis = torch.sigmoid(predicted_dis)
#         return predicted_doa,predicted_dis
#         # return self.predictor(audio_observations)
#
#     def update(self, observations, envs, predict_location):
#         """
#         update the current observations with estimated location in the agent's current coordinate frame
#         if spectrogram in the current obs is zero, transform last estimate to agent's current coordinate frame
#         """
#         num_env = envs.num_envs
#         if self.last_global_coords is None:
#             self.last_global_coords = [None] * num_env
#
#         for i in range(num_env):
#             if observations[i]['spectrogram'].sum() != 0:
#                 if predict_location:
#                     pred_location = self.forward(observations[i])[0, -2:].cpu().numpy()
#                 else:
#                     offsets = [0, +1, -1, +2, -2]
#                     gt_location = observations[i]['pointgoal_with_gps_compass']
#                     pred_location = np.array([gt_location[1] + random.choice(offsets),
#                                           -gt_location[0] + random.choice(offsets)])
#                 self.last_global_coords[i] = envs.call_at(i, 'egocentric_to_global', {'pg': pred_location})
#             else:
#                 pred_location = envs.call_at(i, 'global_to_egocentric', {'pg': self.last_global_coords[i]})
#                 if not predict_location:
#                     pred_location = np.array([-pred_location[1], pred_location[0]])
#             observations[i]['pointgoal_with_gps_compass'] = pred_location

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

    def forward(self, audio_feature):
        if not self.input_shape_printed:
            logging.info('Audiogoal predictor input audio feature shape: {}'.format(audio_feature["spectrogram"].shape))
            self.input_shape_printed = True
        audio_observations = audio_feature['spectrogram']
        if not torch.is_tensor(audio_observations):
            audio_observations = torch.from_numpy(audio_observations).to(device='cuda:0').unsqueeze(0)

        audio_observations = audio_observations.permute(0, 3, 1, 2)
        spec_feat = self.spec_encoder(audio_observations)

        doa_logits = self.doa_head(spec_feat)
        doa_logits = torch.sigmoid(doa_logits)

        distance_logits = self.distance_head(spec_feat)
        distance_logits = torch.sigmoid(distance_logits)

        return doa_logits, distance_logits
        # return self.predictor(audio_observations)