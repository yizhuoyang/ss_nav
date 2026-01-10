import os
import pickle
from itertools import product
import logging
import copy
import random
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset
import networkx as nx
from tqdm import tqdm
from scipy.io import wavfile
from scipy.signal import fftconvolve
from skimage.measure import block_reduce
from ss_baselines.savi.config.default import get_config
from ss_baselines.common.utils import to_tensor
from soundspaces.mp3d_utils import CATEGORY_INDEX_MAPPING
from soundspaces.mp3d_utils import SCENE_SPLITS
from soundspaces.utils import load_metadata

class _DummyObject:
    def __init__(self, *a, **k): pass
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)

class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            __import__(module)
            return getattr(__import__(module, fromlist=[name]), name)
        except Exception:
            return _DummyObject

def _load_pkl_any(path):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError:
            f.seek(0)
            return _SafeUnpickler(f).load()

class AudioGoalDataset(Dataset):
    def __init__(
        self,
        scene_graphs,
        scenes,
        split,
        use_polar_coordinates=False,
        use_cache=False,
        filter_rule='',
        cache_dir="/home/Disk/yyz/sound-spaces/cache",
        cache_name=None,
        rebuild_cache=False,

        depth_dir="/home/Disk/sound-space/depth_npy/mp3d",
        return_depth=True,   # 是否在 __getitem__ 返回 depth
    ):
        self.use_cache = use_cache
        self.files = []
        self.goals = []

        self.binaural_rir_dir = 'data/binaural_rirs/mp3d'
        self.source_sound_dir = f'data/sounds/semantic_splits/{split}'
        self.observation_dir  = 'data/scene_observations/mp3d'
        self.source_sound_dict = {}
        self.rir_sampling_rate = 16000
        self.num_samples_per   = 25000

        self.depth_dir = depth_dir
        self.return_depth = return_depth

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

            self.files = self.files
            self.goals = self.goals

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
    def _parse_rir_path(rir_file: str):
        # .../mp3d/{scene}/{angle}/{r}_{s}.wav
        scene = os.path.basename(os.path.dirname(os.path.dirname(rir_file)))
        angle = int(os.path.basename(os.path.dirname(rir_file)))
        stem = os.path.splitext(os.path.basename(rir_file))[0]  # "r_s"
        r_str, s_str = stem.split("_")
        r = int(r_str)
        s = int(s_str)
        return scene, angle, r, s
    
    def _get_depth_from_pkl(self, scene: str, r: int, angle: int):
        if scene not in self._scene_pkl_cache:
            # 允许两种命名：{scene}.pkl 或直接 scene 目录下某个固定名（你可按需改）
            cand = [
                os.path.join(self.depth_pkl_dir, f"{scene}.pkl"),
                os.path.join(self.depth_pkl_dir, scene, "observations.pkl"),
            ]
            pkl_path = None
            for c in cand:
                if os.path.exists(c):
                    pkl_path = c
                    break
            if pkl_path is None:
                raise FileNotFoundError(f"Cannot find pkl for scene={scene} in {self.depth_pkl_dir}")

            self._scene_pkl_cache[scene] = _load_pkl_any(pkl_path)

        scene_dict = self._scene_pkl_cache[scene]     # dict[(node,angle)] -> obj
        key = (r, angle)
        if key not in scene_dict:
            return None

        entry = scene_dict[key]
        # entry 可能是对象（entry.depth）也可能是 dict（entry["depth"]）
        if isinstance(entry, dict):
            depth = entry[self.depth_key]
        else:
            depth = getattr(entry, self.depth_key)
        return depth
    
    def _get_depth_from_npy(self, scene: str, r: int, angle: int):
        dpath = os.path.join(self.depth_dir, scene, f"{r}_{angle}.npy")
        if not os.path.exists(dpath):
            return None
        depth = np.load(dpath, mmap_mode="r")  # (1,128,128) float16
        return depth

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
            # spectrogram = to_tensor(self.compute_stft_phase_features(audiogoal))  # (F,T,2) float32
            goal = self.goals[item]
            theta = to_tensor(make_doa_gaussian(goal[1],goal[2]))
            distance = to_tensor(make_r_gaussian_1d(goal[2]))
            sound_class = goal[0].long()

            if self.return_depth:
                scene, angle, r, s = self._parse_rir_path(rir_file)
                # depth_np = self._get_depth_from_pkl(scene, r, angle)  # (H,W,1) float32
                depth_np = self._get_depth_from_npy(scene, r, angle)
                depth = torch.from_numpy(np.array(depth_np, copy=False)).float()
                inputs_outputs = ([spectrogram, depth], [theta, distance,sound_class])
            else:
                inputs_outputs = ([spectrogram], [theta, distance,sound_class])

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
            # stft = block_reduce(stft, block_size=(4, 4), func=np.mean)
            return stft

        channel1_magnitude = np.log1p(compute_stft(audiogoal[0]))
        channel2_magnitude = np.log1p(compute_stft(audiogoal[1]))
        spectrogram = np.stack([channel1_magnitude, channel2_magnitude], axis=-1)
        return spectrogram

    @staticmethod
    def compute_stft_phase_features(audiogoal, mode="ipd", eps=1e-8):
        """
        Args:
            audiogoal: array-like, shape (2, N)  (stereo / 2-mic)
            mode:
                - "phase": return per-channel phase sin/cos  -> (F, T, 4) = [cos(phi1), sin(phi1), cos(phi2), sin(phi2)]
                - "ipd":   return inter-channel phase diff sin/cos -> (F, T, 2) = [cos(ipd), sin(ipd)]
                - "both":  return both of above -> (F, T, 6)
                - "gcc_phat_complex": return normalized cross-spectrum real/imag (PHAT) -> (F, T, 2)
        Returns:
            feat: np.ndarray, float32
        """
        def stft_complex(signal):
            n_fft = 512
            hop_length = 160
            win_length = 400
            # complex STFT: (F, T)
            return librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

        # complex STFT for each channel
        X1 = stft_complex(audiogoal[0])
        X2 = stft_complex(audiogoal[1])

        # phase in [-pi, pi]
        phi1 = np.angle(X1)
        phi2 = np.angle(X2)

        # per-channel sin/cos
        c1, s1 = np.cos(phi1), np.sin(phi1)
        c2, s2 = np.cos(phi2), np.sin(phi2)

        # IPD = phi1 - phi2, wrapped to [-pi, pi]
        ipd = np.angle(np.exp(1j * (phi1 - phi2)))
        cipd, sipd = np.cos(ipd), np.sin(ipd)

        if mode == "phase":
            feat = np.stack([c1, s1, c2, s2], axis=-1).astype(np.float32)  # (F,T,4)
            return feat

        if mode == "ipd":
            feat = np.stack([cipd, sipd], axis=-1).astype(np.float32)  # (F,T,2)
            return feat

        if mode == "both":
            feat = np.stack([c1, s1, c2, s2, cipd, sipd], axis=-1).astype(np.float32)  # (F,T,6)
            return feat

        if mode == "gcc_phat_complex":
            # PHAT normalized cross-spectrum: X1*conj(X2) / |X1*conj(X2)|
            C = X1 * np.conj(X2)
            C_phat = C / (np.abs(C) + eps)
            feat = np.stack([C_phat.real, C_phat.imag], axis=-1).astype(np.float32)  # (F,T,2)
            return feat

        raise ValueError(f"Unknown mode: {mode}")


def make_doa_gaussian(
    DOA: float,
    dist_m: float,
    num_bins: int = 360,
    base_sigma_deg: float = 0.5,
    sigma_scale_deg: float = 1.0,
):

    doa = DOA
    if doa < 0:
        doa += 2 * np.pi
    sigma_deg = base_sigma_deg + sigma_scale_deg * dist_m
    sigma_deg = max(sigma_deg, 1e-3)
    sigma_rad = np.deg2rad(sigma_deg)

    angles = np.linspace(0.0, 2 * np.pi, num_bins, endpoint=False)  # (num_bins,)

    diff = np.angle(np.exp(1j * (angles - doa.detach().cpu().numpy())))  # [-pi, pi)
    doa_gauss = np.exp(- (diff ** 2) / (2 * sigma_rad ** 2))
    s = doa_gauss.sum()
    if s > 0:
        doa_gauss = doa_gauss / s

    doa_gauss = (doa_gauss - doa_gauss.min()) / (doa_gauss.max() - doa_gauss.min() + 1e-8)

    return doa_gauss

def make_r_gaussian_1d(r, num_bins=120, r_min=0.0, r_max=30.0,
                       base_sigma=0.05, sigma_scale=0.2):

    r = r.detach().cpu().numpy()
    r_axis = np.linspace(r_min, r_max, num_bins).astype(np.float32)
    sigma = base_sigma + sigma_scale * r
    sigma = max(sigma, 1e-3)
    probs = np.exp(-0.5 * ((r_axis - r) / sigma)**2)
    if probs.sum() > 0:
        probs /= probs.sum()
    probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-8)
    return probs


if __name__ == "__main__":
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CONFIG_PATH = "/home/Disk/sound-space/ss_baselines/savi/config/semantic_audionav/savi.yaml"
    SPLIT = "train"  # "train" / "val" / "test"
    USE_CACHE = True  # 建议 inference 时 False，避免占用巨大内存
    PREDICT_LABEL = False
    PREDICT_LOCATION = True
    CKPT_PATH = "/home/Disk/yyz/sound-spaces/weights/savi/best_val.pth"  # or None

    config = get_config(config_paths=CONFIG_PATH, opts=None, run_type=None)
    meta_dir = config.TASK_CONFIG.SIMULATOR.AUDIO.METADATA_DIR

    scenes = SCENE_SPLITS[SPLIT]

    scene_graphs = {}
    for scene in scenes:
        points, graph = load_metadata(os.path.join(meta_dir, "mp3d", scene))
        scene_graphs[scene] = graph

    dataset = AudioGoalDataset(
        scene_graphs=scene_graphs,
        scenes=scenes,
        split=SPLIT,
        use_polar_coordinates=True,
        use_cache=USE_CACHE,
    )

    print(f"[INFO] dataset split={SPLIT}, len={len(dataset)}")
    x = dataset[0]
    print(x[0][0].shape,x[0][1].shape,x[0][1].min(),x[0][1].max())  # spectrogram
    # -----------------------
    # 3) 构建模型 + 可选加载权重
    # # -----------------------
    # model = AudioGoalPredictor(
    #     predict_label=PREDICT_LABEL,
    #     predict_location=PREDICT_LOCATION
    # ).to(DEVICE)
    # model.eval()
    #
    # if CKPT_PATH is not None and os.path.exists(CKPT_PATH):
    #     ckpt = torch.load(CKPT_PATH, map_location="cpu")
    #     # 你的 trainer 存的是 {"audiogoal_predictor": state_dict}
    #     if "audiogoal_predictor" in ckpt:
    #         model.load_state_dict(ckpt["audiogoal_predictor"], strict=True)
    #         print(f"[INFO] loaded ckpt: {CKPT_PATH}")
    #     else:
    #         # 万一你保存的是裸 state_dict
    #         model.load_state_dict(ckpt, strict=True)
    #         print(f"[INFO] loaded ckpt (raw state_dict): {CKPT_PATH}")
    # else:
    #     print("[WARN] ckpt not loaded (path is None or not exists).")
