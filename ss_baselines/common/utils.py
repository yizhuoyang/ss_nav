#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
from collections import defaultdict
from typing import Dict, List, Optional
import random
import copy
import numbers
import json

import numpy as np
import cv2
from scipy.io import wavfile
import torch
import torch.nn as nn
import torch.nn.functional as f
import moviepy.editor as mpy
from gym.spaces import Box
from moviepy.audio.AudioClip import CompositeAudioClip, AudioArrayClip
import moviepy.editor as mpy
from moviepy.audio.AudioClip import AudioArrayClip
from habitat.utils.visualizations.utils import images_to_video
from habitat import logger
from habitat_sim.utils.common import d3_40_colors_rgb
from ss_baselines.common.tensorboard_utils import TensorboardWriter
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import draw_collision


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


class CustomFixedCategorical(torch.distributions.Categorical):
    def sample(self, sample_shape=torch.Size()):
        return super().sample(sample_shape).unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class CategoricalNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)

        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = self.linear(x)
        return CustomFixedCategorical(logits=x)


class CategoricalNetWithMask(nn.Module):
    def __init__(self, num_inputs, num_outputs, masking):
        super().__init__()
        self.masking = masking

        self.linear = nn.Linear(num_inputs, num_outputs)

        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, features, action_maps):
        probs = f.softmax(self.linear(features))
        if self.masking:
            probs = probs * torch.reshape(action_maps, (action_maps.shape[0], -1)).float()

        return CustomFixedCategorical(probs=probs)


def linear_decay(epoch: int, total_num_updates: int) -> float:
    r"""Returns a multiplicative factor for linear value decay

    Args:
        epoch: current epoch number
        total_num_updates: total number of epochs

    Returns:
        multiplicative factor that decreases param value linearly
    """
    return 1 - (epoch / float(total_num_updates))


def exponential_decay(epoch: int, total_num_updates: int, decay_lambda: float) -> float:
    r"""Returns a multiplicative factor for linear value decay

    Args:
        epoch: current epoch number
        total_num_updates: total number of epochs
        decay_lambda: decay lambda

    Returns:
        multiplicative factor that decreases param value linearly
    """
    return np.exp(-decay_lambda * (epoch / float(total_num_updates)))


def to_tensor(v):
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)


def batch_obs(
    observations: List[Dict], device: Optional[torch.device] = None, skip_list = []
) -> Dict[str, torch.Tensor]:
    r"""Transpose a batch of observation dicts to a dict of batched
    observations.

    Args:
        observations:  list of dicts of observations.
        device: The torch.device to put the resulting tensors on.
            Will not move the tensors if None

    Returns:
        transposed dict of lists of observations.
    """
    batch = defaultdict(list)

    for obs in observations:
        for sensor in obs:
            if sensor in skip_list:
                continue

            batch[sensor].append(to_tensor(obs[sensor]).float())

    for sensor in batch:
        batch[sensor] = torch.stack(batch[sensor], dim=0).to(
            device=device, dtype=torch.float
        )

    return batch


def poll_checkpoint_folder(
    checkpoint_folder: str, previous_ckpt_ind: int, eval_interval: int
) -> Optional[str]:
    r""" Return (previous_ckpt_ind + 1)th checkpoint in checkpoint folder
    (sorted by time of last modification).

    Args:
        checkpoint_folder: directory to look for checkpoints.
        previous_ckpt_ind: index of checkpoint last returned.
        eval_interval: number of checkpoints between two evaluation

    Returns:
        return checkpoint path if (previous_ckpt_ind + 1)th checkpoint is found
        else return None.
    """
    assert os.path.isdir(checkpoint_folder), (
        f"invalid checkpoint folder " f"path {checkpoint_folder}"
    )
    models_paths = list(
        filter(os.path.isfile, glob.glob(checkpoint_folder + "/*"))
    )
    models_paths.sort(key=os.path.getmtime)
    ind = previous_ckpt_ind + eval_interval
    if ind < len(models_paths):
        return models_paths[ind]
    return None

def ensure_2T(audio: np.ndarray) -> np.ndarray:
    """Return (2,T) float32. Accept (2,T) or (T,2)."""
    a = np.asarray(audio)
    if a.ndim == 1:
        raise ValueError("audio is (T,), but stereo (2,T) is required.")
    if a.shape[0] == 2:
        return a.astype(np.float32)
    if a.shape[-1] == 2:
        return a.T.astype(np.float32)
    raise ValueError(f"Unsupported audio shape {a.shape}, expected (2,T) or (T,2).")


def make_stereo_waveform_panel(
    audio_2T: np.ndarray,    # (2, Tseg)
    panel_h: int,
    panel_w: int,
    pad: int = 12,
    gain: float = 1.0,
    ch_names=("L", "R"),
    show_grid: bool = True,
) -> np.ndarray:
    """
    Create an RGB panel visualizing stereo waveforms (no RMS).
    Top half: ch0, bottom half: ch1.
    """
    panel_bgr = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    a = ensure_2T(audio_2T) * float(gain)
    a0, a1 = a[0], a[1]

    def _norm(y):
        peak = float(np.max(np.abs(y)) + 1e-9)
        return (y / peak).astype(np.float32)

    y0 = _norm(a0)
    y1 = _norm(a1)

    def _resample(y):
        if y.size <= 1:
            return np.zeros((panel_w,), np.float32)
        xs = np.linspace(0, y.size - 1, panel_w).astype(np.int32)
        return y[xs]

    y0w = _resample(y0)
    y1w = _resample(y1)

    h_half = panel_h // 2
    mid0 = h_half // 2
    mid1 = h_half + (h_half // 2)
    amp0 = (h_half // 2) - pad
    amp1 = (h_half // 2) - pad

    if show_grid:
        # vertical grid
        step = max(1, panel_w // 10)
        for x in range(0, panel_w, step):
            cv2.line(panel_bgr, (x, 0), (x, panel_h - 1), (35, 35, 35), 1)
        # separator
        cv2.line(panel_bgr, (0, h_half), (panel_w - 1, h_half), (55, 55, 55), 1)

    # midlines
    cv2.line(panel_bgr, (0, mid0), (panel_w - 1, mid0), (75, 75, 75), 1)
    cv2.line(panel_bgr, (0, mid1), (panel_w - 1, mid1), (75, 75, 75), 1)

    col0 = (220, 220, 220)   # ch0
    col1 = (200, 230, 255)   # ch1 (slightly blue-ish)

    for x in range(panel_w):
        v0 = float(y0w[x])
        y_top0 = int(mid0 - v0 * amp0)
        y_bot0 = int(mid0 + v0 * amp0)
        cv2.line(panel_bgr, (x, y_top0), (x, y_bot0), col0, 1)

        v1 = float(y1w[x])
        y_top1 = int(mid1 - v1 * amp1)
        y_bot1 = int(mid1 + v1 * amp1)
        cv2.line(panel_bgr, (x, y_top1), (x, y_bot1), col1, 1)

    # labels
    cv2.putText(panel_bgr, str(ch_names[0]), (pad, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (210, 210, 210), 1)
    cv2.putText(panel_bgr, str(ch_names[1]), (pad, h_half + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (210, 210, 210), 1)

    return cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB)


def images_to_video_with_stereo_audio_and_top_vis(
    images,                 # List[np.ndarray] RGB HxWx3
    output_dir: str,
    video_name: str,
    audios,                 # List[np.ndarray] (2,T) or (T,2)
    sr: int,
    fps: int = 1,
    multiplier: float = 0.5,
    top_panel_h: int = 90,   # audio panel height on top
    gain: float = 1.0,
    ch_names=("L", "R"),
):
    """
    - 不遮挡原图：把 audio waveform 画在上方 panel，再拼到原图上方
    - 不显示 RMS
    - 音频双通道写入视频
    """
    os.makedirs(output_dir, exist_ok=True)
    video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"

    assert len(images) == len(audios), "images and audios must have same length"
    seg_len = int(sr * (1.0 / fps))

    out_frames = []
    audio_clips = []

    for i, frame in enumerate(images):
        frame_rgb = frame.copy()
        H, W = frame_rgb.shape[:2]

        a = ensure_2T(audios[i])  # (2,T)
        a_seg = a[:, :seg_len]
        if a_seg.shape[1] < seg_len:
            a_seg = np.pad(a_seg, ((0, 0), (0, seg_len - a_seg.shape[1])), mode="constant")

        # ---- top waveform panel (match width W) ----
        panel_rgb = make_stereo_waveform_panel(
            audio_2T=a_seg,
            panel_h=int(top_panel_h),
            panel_w=int(W),
            gain=gain,
            ch_names=ch_names,
            show_grid=True,
        )

        merged = np.concatenate([panel_rgb, frame_rgb], axis=0)  # stack vertically
        out_frames.append(merged)

        # ---- audio clip (stereo) ----
        a_T2 = (a_seg * float(multiplier)).astype(np.float32).T  # (T,2) for moviepy
        clip = AudioArrayClip(a_T2, fps=sr).set_start((1.0 / fps) * i)
        audio_clips.append(clip)

    composite_audio = CompositeAudioClip(audio_clips)
    video_clip = mpy.ImageSequenceClip(out_frames, fps=fps)
    video_with_audio = video_clip.set_audio(composite_audio)

    out_path = os.path.join(output_dir, video_name)
    video_with_audio.write_videofile(out_path, fps=fps)
    print("[OK]", out_path)


def generate_video(
    video_option: List[str],
    video_dir: Optional[str],
    images: List[np.ndarray],
    scene_name: str,
    sound: str,
    sr: int,
    episode_id: int,
    checkpoint_idx: int,
    metric_name: str,
    metric_value: float,
    tb_writer: TensorboardWriter,
    fps: int = 10,
    audios: List[str] = None
) -> None:
    r"""Generate video according to specified information.

    Args:
        video_option: string list of "tensorboard" or "disk" or both.
        video_dir: path to target video directory.
        images: list of images to be converted to video.
        episode_id: episode id for video naming.
        checkpoint_idx: checkpoint index for video naming.
        metric_name: name of the performance metric, e.g. "spl".
        metric_value: value of metric.
        tb_writer: tensorboard writer object for uploading video.
        fps: fps for generated video.
        audios: raw audio files
    Returns:
        None
    """
    if len(images) < 1:
        return

    video_name = f"{scene_name}_{episode_id}_{sound}_{metric_name}{metric_value:.2f}"
    if "disk" in video_option:
        assert video_dir is not None
        if audios is None:
            images_to_video(images, video_dir, video_name)
        else:
            # images_to_video_with_audio(images, video_dir, video_name, audios, sr, fps=fps)
            images_to_video_with_stereo_audio_and_top_vis(images, video_dir, video_name, audios, sr, fps=fps)
    if "tensorboard" in video_option:
        tb_writer.add_video_from_np_images(
            f"episode{episode_id}", checkpoint_idx, images, fps=fps
        )

def plot_top_down_map(info, dataset='replica', pred=None, source_world=None):
    top_down_map = info["top_down_map"]["map"]
    top_down_map = maps.colorize_topdown_map(
        top_down_map, info["top_down_map"]["fog_of_war_mask"]
    )
    map_agent_pos = info["top_down_map"]["agent_map_coord"]  # (row, col) in map
    # print("here")
    if dataset == 'replica':
        agent_radius_px = top_down_map.shape[0] // 16
    else:
        agent_radius_px = top_down_map.shape[0] // 50

    # 画 agent
    top_down_map = maps.draw_agent(
        image=top_down_map,
        agent_center_coord=map_agent_pos,
        agent_rotation=info["top_down_map"]["agent_angle"],
        agent_radius_px=agent_radius_px
    )

    # ====== 1) 你原来的 pred-based 预测点（保留） ======
    if pred is not None:
        from habitat.utils.geometry_utils import quaternion_rotate_vector

        source_rotation = info["top_down_map"]["agent_rotation"]

        rounded_pred = np.round(pred[1])
        # pred: [front, right]，转成 agent 坐标系下的 3D 向量
        direction_vector_agent = np.array([rounded_pred[1], 0, -rounded_pred[0]])
        # 旋转到世界坐标
        direction_vector = quaternion_rotate_vector(source_rotation, direction_vector_agent)

        grid_size = (
            (maps.COORDINATE_MAX - maps.COORDINATE_MIN) / 10000,
            (maps.COORDINATE_MAX - maps.COORDINATE_MIN) / 10000,
        )
        delta_x = int(-direction_vector[0] / grid_size[0])
        delta_y = int(direction_vector[2] / grid_size[1])

        x = int(np.clip(map_agent_pos[0] + delta_x, 0, top_down_map.shape[0] - 1))
        y = int(np.clip(map_agent_pos[1] + delta_y, 0, top_down_map.shape[1] - 1))

        point_padding = 20
        for m in range(x - point_padding, x + point_padding + 1):
            for n in range(y - point_padding, y + point_padding + 1):
                if (
                    np.linalg.norm(np.array([m - x, n - y])) <= point_padding
                    and 0 <= m < top_down_map.shape[0]
                    and 0 <= n < top_down_map.shape[1]
                ):
                    # 预测点画成黄色
                    top_down_map[m, n] = (0, 255, 255)

        if np.linalg.norm(rounded_pred) < 1:
            assert delta_x == 0 and delta_y == 0

    # ====== 2) 真实 source 位置标注（关键部分） ======
    if source_world is not None:
        # source_world: [x, y, z] in world coordinates
        source_world = np.array(source_world, dtype=np.float32)

        # agent 的世界坐标（你自己根据实际情况替换这个字段）
        agent_world = np.array(info["top_down_map"]["agent_world_pos"], dtype=np.float32)

        # 计算 source 相对 agent 的位移（世界坐标）
        rel_vec = source_world - agent_world   # [dx, dy, dz]
        # 同样用 grid_size 把 metric → pixel offset
        grid_size = (
            (maps.COORDINATE_MAX - maps.COORDINATE_MIN) / 10000,
            (maps.COORDINATE_MAX - maps.COORDINATE_MIN) / 10000,
        )

        # 注意方向跟 pred 那里保持一致：
        # world +X → 图像左边，所以用 -rel_vec[0]
        delta_x = int(-rel_vec[0] / grid_size[0])
        # world +Z → 图像下方，所以用 +rel_vec[2]
        delta_y = int(rel_vec[2] / grid_size[1])

        sx = int(np.clip(map_agent_pos[0] + delta_x, 0, top_down_map.shape[0] - 1))
        sy = int(np.clip(map_agent_pos[1] + delta_y, 0, top_down_map.shape[1] - 1))

        # 画一个小圆点表示真实 source（比如红色）
        point_padding = 50
        for m in range(sx - point_padding, sx + point_padding + 1):
            for n in range(sy - point_padding, sy + point_padding + 1):
                if (
                    np.linalg.norm(np.array([m - sx, n - sy])) <= point_padding
                    and 0 <= m < top_down_map.shape[0]
                    and 0 <= n < top_down_map.shape[1]
                ):
                    top_down_map[m, n] = (255, 0, 0)  # red for GT source

    # 旋转到更直观的方向（你原来的逻辑）
    if top_down_map.shape[0] > top_down_map.shape[1]:
        top_down_map = np.rot90(top_down_map, 1)

    return top_down_map

def images_to_video_with_audio(
    images: List[np.ndarray],
    output_dir: str,
    video_name: str,
    audios: List[str],
    sr: int,
    fps: int = 1,
    quality: Optional[float] = 5,
    **kwargs
):
    r"""Calls imageio to run FFMPEG on a list of images. For more info on
    parameters, see https://imageio.readthedocs.io/en/stable/format_ffmpeg.html
    Args:
        images: The list of images. Images should be HxWx3 in RGB order.
        output_dir: The folder to put the video in.
        video_name: The name for the video.
        audios: raw audio files
        fps: Frames per second for the video. Not all values work with FFMPEG,
            use at your own risk.
        quality: Default is 5. Uses variable bit rate. Highest quality is 10,
            lowest is 0.  Set to None to prevent variable bitrate flags to
            FFMPEG so you can manually specify them using output_params
            instead. Specifying a fixed bitrate using ‘bitrate’ disables
            this parameter.
    """
    assert 0 <= quality <= 10
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"

    audio_clips = []
    multiplier = 0.5
    for i, audio in enumerate(audios):
        audio_clip = AudioArrayClip(audio.T[:int(sr * 1 / fps)] * multiplier, fps=sr)
        audio_clip = audio_clip.set_start(1 / fps * i)
        audio_clips.append(audio_clip)
    composite_audio_clip = CompositeAudioClip(audio_clips)
    video_clip = mpy.ImageSequenceClip(images, fps=fps)
    video_with_new_audio = video_clip.set_audio(composite_audio_clip)
    video_with_new_audio.write_videofile(os.path.join(output_dir, video_name))

def resize_observation(observations, model_resolution):
    for observation in observations:
        observation['rgb'] = cv2.resize(observation['rgb'], (model_resolution, model_resolution))
        observation['depth'] = np.expand_dims(cv2.resize(observation['depth'], (model_resolution, model_resolution)),
                                              axis=-1)

def convert_semantics_to_rgb(semantics):
    r"""Converts semantic IDs to RGB images.
    """
    semantics = semantics.long() % 40
    mapping_rgb = torch.from_numpy(d3_40_colors_rgb).to(semantics.device)
    semantics_r = torch.take(mapping_rgb[:, 0], semantics)
    semantics_g = torch.take(mapping_rgb[:, 1], semantics)
    semantics_b = torch.take(mapping_rgb[:, 2], semantics)
    semantics_rgb = torch.stack([semantics_r, semantics_g, semantics_b], -1)

    return semantics_rgb


class ResizeCenterCropper(nn.Module):
    def __init__(self, size, channels_last: bool = False):
        r"""An nn module the resizes and center crops your input.
        Args:
            size: A sequence (w, h) or int of the size you wish to resize/center_crop.
                    If int, assumes square crop
            channels_list: indicates if channels is the last dimension
        """
        super().__init__()
        if isinstance(size, numbers.Number):
            size = (int(size), int(size))
        assert len(size) == 2, "forced input size must be len of 2 (w, h)"
        self._size = size
        self.channels_last = channels_last

    def transform_observation_space(
        self, observation_space, trans_keys=["rgb", "depth", "semantic"]
    ):
        size = self._size
        observation_space = copy.deepcopy(observation_space)
        if size:
            for key in observation_space.spaces:
                if (
                    key in trans_keys
                    and observation_space.spaces[key].shape != size
                ):
                    logger.info(
                        "Overwriting CNN input size of %s: %s" % (key, size)
                    )
                    observation_space.spaces[key] = overwrite_gym_box_shape(
                        observation_space.spaces[key], size
                    )
        self.observation_space = observation_space
        return observation_space

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._size is None:
            return input

        return center_crop(
            image_resize_shortest_edge(
                input, max(self._size), channels_last=self.channels_last
            ),
            self._size,
            channels_last=self.channels_last,
        )


def image_resize_shortest_edge(
    img, size: int, channels_last: bool = False
) -> torch.Tensor:
    """Resizes an img so that the shortest side is length of size while
        preserving aspect ratio.

    Args:
        img: the array object that needs to be resized (HWC) or (NHWC)
        size: the size that you want the shortest edge to be resize to
        channels: a boolean that channel is the last dimension
    Returns:
        The resized array as a torch tensor.
    """
    img = to_tensor(img)
    no_batch_dim = len(img.shape) == 3
    if len(img.shape) < 3 or len(img.shape) > 5:
        raise NotImplementedError()
    if no_batch_dim:
        img = img.unsqueeze(0)  # Adds a batch dimension
    if channels_last:
        h, w = img.shape[-3:-1]
        if len(img.shape) == 4:
            # NHWC -> NCHW
            img = img.permute(0, 3, 1, 2)
        else:
            # NDHWC -> NDCHW
            img = img.permute(0, 1, 4, 2, 3)
    else:
        # ..HW
        h, w = img.shape[-2:]

    # Percentage resize
    scale = size / min(h, w)
    h = int(h * scale)
    w = int(w * scale)
    img = torch.nn.functional.interpolate(
        img.float(), size=(h, w), mode="area"
    ).to(dtype=img.dtype)
    if channels_last:
        if len(img.shape) == 4:
            # NCHW -> NHWC
            img = img.permute(0, 2, 3, 1)
        else:
            # NDCHW -> NDHWC
            img = img.permute(0, 1, 3, 4, 2)
    if no_batch_dim:
        img = img.squeeze(dim=0)  # Removes the batch dimension
    return img


def center_crop(img, size, channels_last: bool = False):
    """Performs a center crop on an image.

    Args:
        img: the array object that needs to be resized (either batched or unbatched)
        size: A sequence (w, h) or a python(int) that you want cropped
        channels_last: If the channels are the last dimension.
    Returns:
        the resized array
    """
    if channels_last:
        # NHWC
        h, w = img.shape[-3:-1]
    else:
        # NCHW
        h, w = img.shape[-2:]

    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    assert len(size) == 2, "size should be (h,w) you wish to resize to"
    cropx, cropy = size

    startx = w // 2 - (cropx // 2)
    starty = h // 2 - (cropy // 2)
    if channels_last:
        return img[..., starty : starty + cropy, startx : startx + cropx, :]
    else:
        return img[..., starty : starty + cropy, startx : startx + cropx]


def overwrite_gym_box_shape(box: Box, shape) -> Box:
    if box.shape == shape:
        return box
    shape = list(shape) + list(box.shape[len(shape) :])
    low = box.low if np.isscalar(box.low) else np.min(box.low)
    high = box.high if np.isscalar(box.high) else np.max(box.high)
    return Box(low=low, high=high, shape=shape, dtype=box.dtype)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def normalize01(x, eps=1e-12):
    x = x.astype(np.float32)
    x = x - x.min()
    x = x / (x.max() + eps)
    return x

def bilinear_sample(image, u, v, outside_value=0.0):
    """
    image: (Hs,Ws) float
    u,v: (H,W) float (col,row)
    """
    Hs, Ws = image.shape
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = u0 + 1
    v1 = v0 + 1

    inside = (u0 >= 0) & (v0 >= 0) & (u1 < Ws) & (v1 < Hs)

    u0c = np.clip(u0, 0, Ws - 1)
    u1c = np.clip(u1, 0, Ws - 1)
    v0c = np.clip(v0, 0, Hs - 1)
    v1c = np.clip(v1, 0, Hs - 1)

    Ia = image[v0c, u0c]
    Ib = image[v0c, u1c]
    Ic = image[v1c, u0c]
    Id = image[v1c, u1c]

    du = u - u0.astype(np.float32)
    dv = v - v0.astype(np.float32)

    out = ((1 - du) * (1 - dv) * Ia +
           du * (1 - dv) * Ib +
           (1 - du) * dv * Ic +
           du * dv * Id).astype(np.float32)

    out[~inside] = outside_value
    return out

def overlay_heat_on_rgb(img_rgb, heat01, alpha=0.45, cmap=cv2.COLORMAP_JET):
    """
    img_rgb: (H,W,3) uint8
    heat01: (H,W) float [0,1]
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    heat_u8 = np.clip(heat01 * 255.0, 0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cmap)  # BGR
    out_bgr = cv2.addWeighted(img_bgr, 1-alpha, heat_color, alpha, 0.0)
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

def make_sound_bounds_from_center(center_xz, map_size_m):
    """
    center_xz: (2,) [cx, cz]
    map_size_m: float, soundmap 覆盖的世界长度（米），例如 60.0
    返回 bounds: ((minx,0,minz),(maxx,0,maxz))
    """
    cx, cz = float(center_xz[0]), float(center_xz[1])
    half = map_size_m * 0.5
    minx, maxx = cx - half, cx + half
    minz, maxz = cz - half, cz + half
    return ((minx, 0.0, minz), (maxx, 0.0, maxz))

def resample_sound60_to_topdown(
    sound60, sound_bounds,
    top_bounds, top_H, top_W,
    flip_z_top=True, flip_z_sound=True
):
    """
    sound60: (60,60) probability grid over sound_bounds
    输出 heat: (top_H, top_W) aligned with topdown raw map pixel coords
    """
    sound60 = sound60.astype(np.float32)
    Hs, Ws = sound60.shape
    # assert Hs == 60 and Ws == 60, "soundmap should be 60x60"

    (tminx, _, tminz), (tmaxx, _, tmaxz) = top_bounds
    (sminx, _, sminz), (smaxx, _, smaxz) = sound_bounds

    # top pixel -> world (x,z)
    cols = np.linspace(0, top_W - 1, top_W, dtype=np.float32)
    rows = np.linspace(0, top_H - 1, top_H, dtype=np.float32)
    cc, rr = np.meshgrid(cols, rows)  # (H,W)

    xw = tminx + (cc / (top_W - 1 + 1e-9)) * (tmaxx - tminx)
    if flip_z_top:
        zw = tmaxz - (rr / (top_H - 1 + 1e-9)) * (tmaxz - tminz)
    else:
        zw = tminz + (rr / (top_H - 1 + 1e-9)) * (tmaxz - tminz)

    # world -> sound uv (col,row in sound grid)
    u = (xw - sminx) / (smaxx - sminx + 1e-9) * (Ws - 1)
    if flip_z_sound:
        v = (smaxz - zw) / (smaxz - sminz + 1e-9) * (Hs - 1)
    else:
        v = (zw - sminz) / (smaxz - sminz + 1e-9) * (Hs - 1)

    heat = bilinear_sample(sound60, u, v, outside_value=0.0)
    return heat

def world_to_map_rc(xz, bounds, mpp_x, mpp_z, H, W, flip_z=True):
    (minx, _, minz), (maxx, _, maxz) = bounds
    x = xz[..., 0]
    z = xz[..., 1]
    col = (x - minx) / (mpp_x + 1e-12)
    if flip_z:
        row = (maxz - z) / (mpp_z + 1e-12)
    else:
        row = (z - minz) / (mpp_z + 1e-12)
    row = np.clip(row, 0, H - 1)
    col = np.clip(col, 0, W - 1)
    return np.stack([row, col], axis=-1)

def pick_best_flip(world_xz_hist, map_rc_hist, bounds, mpp_x, mpp_z, H, W):
    pred0 = world_to_map_rc(world_xz_hist, bounds, mpp_x, mpp_z, H, W, flip_z=False)
    pred1 = world_to_map_rc(world_xz_hist, bounds, mpp_x, mpp_z, H, W, flip_z=True)
    e0 = float(np.mean(np.linalg.norm(pred0 - map_rc_hist, axis=1)))
    e1 = float(np.mean(np.linalg.norm(pred1 - map_rc_hist, axis=1)))
    flip = (e1 < e0)
    print(f"[flip_z_top] False err={e0:.3f}, True err={e1:.3f} -> use {flip}")
    return flip

def overlay_sound60_on_topdown(
    info, sim, render_h,
    sound60, sound_bounds,
    world_xz_hist=None, map_rc_hist=None,
    alpha=0.45
):
    """
    info: infos[0]
    sim: habitat_env.sim
    render_h: render_frame.shape[0]
    sound60: (60,60)
    sound_bounds: ((minx,0,minz),(maxx,0,maxz)) for soundmap
    """
    td = info["top_down_map"]
    top_map_raw = td["map"]
    H, W = top_map_raw.shape[:2]

    # topdown bounds + mpp from sim.pathfinder
    bounds = sim.pathfinder.get_bounds()
    (minx, miny, minz), (maxx, maxy, maxz) = bounds
    mpp_x = (maxx - minx) / float(W)
    mpp_z = (maxz - minz) / float(H)

    # auto pick flip_z_top if history provided
    flip_z_top = False
    if world_xz_hist is not None and map_rc_hist is not None and len(world_xz_hist) >= 3:
        flip_z_top = pick_best_flip(world_xz_hist, map_rc_hist, bounds, mpp_x, mpp_z, H, W)

    # render habitat topdown (fit to height)
    td_rgb_fit = maps.colorize_draw_agent_and_fit_to_height(td, render_h)
    Hfit, Wfit = td_rgb_fit.shape[:2]

    # resample sound -> raw topdown size
    heat_raw = resample_sound60_to_topdown(
        sound60=normalize01(sound60),
        sound_bounds=sound_bounds,
        top_bounds=bounds,
        top_H=H, top_W=W,
        flip_z_top=flip_z_top,
        flip_z_sound=False  # 如果你 soundmap 行方向相反，改 False
    )
    heat01_raw = normalize01(heat_raw)

    # resize to fit size
    heat01_fit = cv2.resize(heat01_raw, (Wfit, Hfit), interpolation=cv2.INTER_LINEAR)

    # overlay
    td_rgb_overlay = overlay_heat_on_rgb(td_rgb_fit, heat01_fit, alpha=alpha)
    return td_rgb_overlay


def _ensure_xyz(pos):
    """Accept (3,) xyz or (2,) xz, return (x,y,z)."""
    if pos is None:
        return None
    p = np.asarray(pos, dtype=np.float32).reshape(-1)
    if p.size == 3:
        return p
    if p.size == 2:
        return np.array([p[0], 0.0, p[1]], dtype=np.float32)
    raise ValueError(f"pos must be (3,) or (2,), got shape {p.shape}")

def world_to_topdown_rc(world_pos, sim, H, W, flip_z=True):
    """
    world_pos: (x,y,z) or (x,z)
    return: (row, col) int
    """
    p = _ensure_xyz(world_pos)
    bounds = sim.pathfinder.get_bounds()
    (minx, _, minz), (maxx, _, maxz) = bounds

    x, z = float(p[0]), float(p[2])

    col = (x - minx) / (maxx - minx + 1e-9) * (W - 1)
    if flip_z:
        row = (maxz - z) / (maxz - minz + 1e-9) * (H - 1)
    else:
        row = (z - minz) / (maxz - minz + 1e-9) * (H - 1)

    r = int(np.clip(np.round(row), 0, H - 1))
    c = int(np.clip(np.round(col), 0, W - 1))
    return (r, c)

def draw_filled_star_bgr(img_bgr, center_xy, outer_r=10, inner_r=None, color=(0,0,255), rotation_deg=-90):
    """
    在 BGR 图上画填充五角星
    center_xy: (x,y) = (col,row)
    color: BGR (红色是 (0,0,255))
    """
    if inner_r is None:
        inner_r = int(outer_r * 0.5)

    cx, cy = center_xy
    pts = []
    # 10 个点交替外/内
    for k in range(10):
        ang = np.deg2rad(rotation_deg + k * 36.0)
        r = outer_r if (k % 2 == 0) else inner_r
        x = cx + r * np.cos(ang)
        y = cy + r * np.sin(ang)
        pts.append([x, y])

    pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(img_bgr, [pts], color)

def draw_goal_pred_on_topdown(
    topdown_rgb,
    sim,
    gt_position=None,      # world (x,y,z) or (x,z)
    pred_position=None,    # world (x,y,z) or (x,z)
    flip_z=True,
    star_outer_r=10,
    pred_radius=7,
):
    """
    GT: 红色五角星
    Pred: 绿色圆
    """
    img_bgr = cv2.cvtColor(topdown_rgb, cv2.COLOR_RGB2BGR)
    H, W = img_bgr.shape[:2]

    # --- GT star (red) ---
    if gt_position is not None:
        r, c = world_to_topdown_rc(gt_position, sim, H, W, flip_z=flip_z)
        draw_filled_star_bgr(
            img_bgr,
            center_xy=(c, r),
            outer_r=star_outer_r,
            inner_r=int(star_outer_r * 0.5),
            color=(0, 0, 255),   # red (BGR)
            rotation_deg=-90
        )

    # # --- Pred circle (green) ---
    # if pred_position is not None:
    #     r, c = world_to_topdown_rc(pred_position, sim, H, W, flip_z=flip_z)
    #     cv2.circle(img_bgr, (c, r), int(pred_radius), (0, 180, 0), -1)  # darker green

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def observations_to_image(observation, info,sound_map=None,sim=None,sound_bounds=None,goal_position=None,pred_position=None):
    r"""Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    render_obs_images: List[np.ndarray] = []
    for sensor_name in observation:
        if "rgb" in sensor_name:
            rgb = observation[sensor_name]
            if not isinstance(rgb, np.ndarray):
                rgb = rgb.cpu().numpy()

            render_obs_images.append(rgb)
        elif "depth" in sensor_name:
            depth_map = observation[sensor_name].squeeze() * 255.0
            if not isinstance(depth_map, np.ndarray):
                depth_map = depth_map.cpu().numpy()

            depth_map = depth_map.astype(np.uint8)
            depth_map = np.stack([depth_map for _ in range(3)], axis=2)
            render_obs_images.append(depth_map)

    # add image goal if observation has image_goal info
    if "imagegoal" in observation:
        rgb = observation["imagegoal"]
        if not isinstance(rgb, np.ndarray):
            rgb = rgb.cpu().numpy()

        render_obs_images.append(rgb)

    assert (
        len(render_obs_images) > 0
    ), "Expected at least one visual sensor enabled."

    shapes_are_equal = len(set(x.shape for x in render_obs_images)) == 1
    if not shapes_are_equal:
        render_frame = tile_images(render_obs_images)
    else:
        render_frame = np.concatenate(render_obs_images, axis=1)

    # draw collision
    if "collisions" in info and info["collisions"]["is_collision"]:
        render_frame = draw_collision(render_frame)

    if "top_down_map" in info:
        if sound_map is None or sim is None or sound_bounds is None or goal_position is None or pred_position is None:
            top_down_map = info["top_down_map"]["map"]
            top_down_map = maps.colorize_topdown_map(
                top_down_map, info["top_down_map"]["fog_of_war_mask"]
            )
            map_agent_pos = info["top_down_map"]["agent_map_coord"]
            top_down_map = maps.draw_agent(
                image=top_down_map,
                agent_center_coord=map_agent_pos,
                agent_rotation=info["top_down_map"]["agent_angle"],
                agent_radius_px=top_down_map.shape[0] // 16,
            )
        else:
            top_down_map = overlay_sound60_on_topdown(
                info=info,
                sim=sim,
                render_h=render_frame.shape[0],
                sound60=sound_map,
                sound_bounds=sound_bounds,
                alpha=0.5
            )
            top_down_map = draw_goal_pred_on_topdown(
            topdown_rgb=top_down_map,
            sim=sim,
            gt_position=goal_position,     
            pred_position=pred_position, 
            flip_z=False,                 
            star_outer_r=10,
            pred_radius=7,
        )
        render_frame = np.concatenate((render_frame, top_down_map), axis=1)

    return render_frame

