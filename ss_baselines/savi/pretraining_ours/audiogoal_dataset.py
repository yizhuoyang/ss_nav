import os
import pickle
from itertools import product
import logging
import copy
import random

import librosa
import numpy as np
from torch.utils.data import Dataset
import networkx as nx
from tqdm import tqdm
from scipy.io import wavfile
from scipy.signal import fftconvolve
from skimage.measure import block_reduce

from ss_baselines.common.utils import to_tensor
from soundspaces.mp3d_utils import CATEGORY_INDEX_MAPPING


class AudioGoalDataset(Dataset):
    def __init__(
        self,
        scene_graphs,
        scenes,
        split,
        use_polar_coordinates=False,
        use_cache=False,
        filter_rule='',
        # ===== 新增：磁盘 cache 参数 =====
        cache_dir="/home/Disk/yyz/sound-spaces/cache",
        cache_name=None,
        rebuild_cache=False,
    ):
        self.use_cache = use_cache
        self.files = []
        self.goals = []

        self.binaural_rir_dir = 'data/binaural_rirs/mp3d'
        self.source_sound_dir = f'data/sounds/semantic_splits/{split}'
        self.source_sound_dict = {}
        self.rir_sampling_rate = 16000
        self.num_samples_per   = 10000

        # ===== 1) 磁盘 cache 路径 =====
        os.makedirs(cache_dir, exist_ok=True)
        if cache_name is None:
            # 默认：用 split + 采样参数做名字（你也可以自定义更详细）
            cache_name = f"mp3d_{split}_pairs{self.num_samples_per}_polar{int(use_polar_coordinates)}.npz"
        cache_path = os.path.join(cache_dir, cache_name)

        # ===== 2) 如果 cache 存在且不 rebuild：直接读取 =====
        if (not rebuild_cache) and os.path.exists(cache_path):
            print(f"[AudioGoalDataset] Load cache: {cache_path}")
            data = np.load(cache_path, allow_pickle=True)

            # files: object array of tuples (rir_file, sound_file)
            self.files = data["files"].tolist()

            # goals: float32 array (N,3)
            goals_np = data["goals"].astype(np.float32)
            self.goals = [to_tensor(g) for g in goals_np]

            # 可选保存一些 meta 以做一致性检查
            # meta = data["meta"].item() if "meta" in data else {}
        else:
            # ===== 3) 否则：生成并保存 cache =====
            sound_files = os.listdir(self.source_sound_dir)
            if len(sound_files) == 0:
                raise RuntimeError(f"No sound files in {self.source_sound_dir}")
            print(f"[AudioGoalDataset] Build cache: {cache_path}")
            print("[AudioGoalDataset] sound files:", sound_files)

            for scene in tqdm(scenes, desc=f"Build ({split})"):
                scene_graph = scene_graphs[scene]
                goals = []

                subgraphs = list(nx.connected_components(scene_graph))
                sr_pairs = []
                for subgraph in subgraphs:
                    sr_pairs += list(product(subgraph, subgraph))

                random.shuffle(sr_pairs)

                kept = 0
                for s, r in sr_pairs:
                    if kept >= self.num_samples_per:
                        break
                    if s == r:
                        continue

                    sound_file = random.choice(sound_files)
                    cat = sound_file[:-4]
                    if cat not in CATEGORY_INDEX_MAPPING:
                        continue
                    index = CATEGORY_INDEX_MAPPING[cat]

                    angle = random.choice([0, 90, 180, 270])
                    rir_file = os.path.join(self.binaural_rir_dir, scene, str(angle), f"{r}_{s}.wav")

                    if not os.path.exists(rir_file):
                        continue

                    self.files.append((rir_file, sound_file))

                    delta_x = scene_graph.nodes[s]['point'][0] - scene_graph.nodes[r]['point'][0]
                    delta_y = scene_graph.nodes[s]['point'][2] - scene_graph.nodes[r]['point'][2]
                    goal_xy = self._compute_goal_xy(delta_x, delta_y, angle, use_polar_coordinates)

                    goal = to_tensor(np.zeros(3, dtype=np.float32))
                    goal[0] = float(index)
                    goal[1:] = goal_xy
                    goals.append(goal)

                    kept += 1

                self.goals += goals

            if len(self.files) == 0:
                raise RuntimeError("Built 0 samples. Check rir dir / mapping / scenes.")

            # 保存 cache（只存轻量信息）
            goals_np = torch.stack(self.goals, dim=0).cpu().numpy().astype(np.float32)
            files_np = np.array(self.files, dtype=object)

            meta = dict(
                split=split,
                use_polar_coordinates=bool(use_polar_coordinates),
                num_samples=len(self.files),
                binaural_rir_dir=self.binaural_rir_dir,
                source_sound_dir=self.source_sound_dir,
            )

            np.savez_compressed(cache_path, files=files_np, goals=goals_np, meta=np.array(meta, dtype=object))
            print(f"[AudioGoalDataset] Saved cache: {cache_path} (N={len(self.files)})")

        self.data = [None] * len(self.goals)
        self.load_source_sounds()

    def audio_length(self, sound):
        return self.source_sound_dict[sound].shape[0] // self.rir_sampling_rate

    def load_source_sounds(self):
        sound_files = os.listdir(self.source_sound_dir)
        for sound_file in sound_files:
            audio_data, sr = librosa.load(
                os.path.join(self.source_sound_dir, sound_file),
                sr=self.rir_sampling_rate
            )
            self.source_sound_dict[sound_file] = audio_data

    @staticmethod
    def _compute_goal_xy(delta_x, delta_y, angle, use_polar_coordinates):
        if angle == 0:
            x = delta_x
            y = delta_y
        elif angle == 90:
            x = delta_y
            y = -delta_x
        elif angle == 180:
            x = -delta_x
            y = -delta_y
        else:
            x = -delta_y
            y = delta_x

        if use_polar_coordinates:
            theta = np.arctan2(y, x)
            distance = np.linalg.norm([y, x])
            goal_xy = to_tensor([theta, distance])
        else:
            goal_xy = to_tensor([x, y])
        return goal_xy

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        if (self.use_cache and self.data[item] is None) or (not self.use_cache):
            rir_file, sound_file = self.files[item]
            audiogoal = self.compute_audiogoal(rir_file, sound_file)
            spectrogram = to_tensor(self.compute_spectrogram(audiogoal))
            inputs_outputs = ([spectrogram], self.goals[item])

            if self.use_cache:
                self.data[item] = inputs_outputs
        else:
            inputs_outputs = self.data[item]
        return inputs_outputs

    def compute_audiogoal(self, binaural_rir_file, sound_file):
        sampling_rate = self.rir_sampling_rate
        try:
            sampling_freq, binaural_rir = wavfile.read(binaural_rir_file)
        except ValueError:
            logging.warning(f"{binaural_rir_file} file is not readable")
            binaural_rir = np.zeros((sampling_rate, 2), dtype=np.float32)

        if len(binaural_rir) == 0:
            logging.debug(f"Empty RIR file at {binaural_rir_file}")
            binaural_rir = np.zeros((sampling_rate, 2), dtype=np.float32)

        current_source_sound = self.source_sound_dict[sound_file]
        index = random.randint(0, self.audio_length(sound_file) - 2)

        if index * sampling_rate - binaural_rir.shape[0] < 0:
            source_sound = current_source_sound[: (index + 1) * sampling_rate]
            binaural_convolved = np.array([
                fftconvolve(source_sound, binaural_rir[:, channel])
                for channel in range(binaural_rir.shape[-1])
            ])
            audiogoal = binaural_convolved[:, index * sampling_rate: (index + 1) * sampling_rate]
        else:
            source_sound = current_source_sound[
                index * sampling_rate - binaural_rir.shape[0] : (index + 1) * sampling_rate
            ]
            binaural_convolved = np.array([
                fftconvolve(source_sound, binaural_rir[:, channel], mode='valid')
                for channel in range(binaural_rir.shape[-1])
            ])
            audiogoal = binaural_convolved[:, :-1]

        return audiogoal

    @staticmethod
    def compute_spectrogram(audiogoal):
        def compute_stft(signal):
            n_fft = 512
            hop_length = 160
            win_length = 400
            stft = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
            stft = block_reduce(stft, block_size=(4, 4), func=np.mean)
            return stft

        channel1_magnitude = np.log1p(compute_stft(audiogoal[0]))
        channel2_magnitude = np.log1p(compute_stft(audiogoal[1]))
        spectrogram = np.stack([channel1_magnitude, channel2_magnitude], axis=-1)
        return spectrogram

    @staticmethod
    def make_doa_gaussian(
        DOA: float,
        num_bins: int = 360,
        base_sigma_deg: float = 0.5,
        sigma_scale_deg: float = 1.0,
    ):

        doa = DOA
        if doa < 0:
            doa += 2 * np.pi
        dist_m = np.sqrt(front**2 + right**2)
        sigma_deg = base_sigma_deg + sigma_scale_deg * dist_m
        sigma_deg = max(sigma_deg, 1e-3)
        sigma_rad = np.deg2rad(sigma_deg)

        angles = np.linspace(0.0, 2 * np.pi, num_bins, endpoint=False)  # (num_bins,)

        diff = np.angle(np.exp(1j * (angles - doa)))  # [-pi, pi)
        doa_gauss = np.exp(- (diff ** 2) / (2 * sigma_rad ** 2))
        s = doa_gauss.sum()
        if s > 0:
            doa_gauss = doa_gauss / s

        doa_gauss = (doa_gauss - doa_gauss.min()) / (doa_gauss.max() - doa_gauss.min() + 1e-8)

        return doa_gauss

    def make_r_gaussian_1d(r, num_bins=120, r_min=0.0, r_max=30.0,
                           base_sigma=0.05, sigma_scale=0.2):


        r_axis = np.linspace(r_min, r_max, num_bins).astype(np.float32)
        sigma = base_sigma + sigma_scale * r
        sigma = max(sigma, 1e-3)
        probs = np.exp(-0.5 * ((r_axis - r) / sigma)**2)
        if probs.sum() > 0:
            probs /= probs.sum()
        probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-8)
        return probs