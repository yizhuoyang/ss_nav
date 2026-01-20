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
        filter_rule="",
        cache_dir="/home/Disk/yyz/sound-spaces/cache",
        cache_name=None,
        rebuild_cache=False,
        depth_dir="/home/Disk/sound-space/depth_npy/mp3d",
        return_depth=True,

        # ===== 新增：distractor 配置 =====
        has_distractor_sound=False,
        distractor_prob=1.0,
        distractor_snr_db="random_0_60",  # "random_0_60" / None / float
        distractor_split=None,           # 默认用 split；你也可传 "train"/"val"/"test"
    ):
        self.use_cache = use_cache
        self.files = []
        self.goals = []

        self.binaural_rir_dir = "data/binaural_rirs/mp3d"

        # 目标声目录
        self.source_sound_dir = f"data/sounds/semantic_splits/{split}"
        # 干扰声目录（可与目标声不同 split / 不同文件夹）
        if distractor_split is None:
            distractor_split = split
        self.distractor_sound_dir = f"data/sounds/1s_all_distractor/{distractor_split}"

        self.observation_dir = "data/scene_observations/mp3d"

        # 分开缓存两套 sound dict（目标声 / 干扰声）
        self.source_sound_dict = {}
        self.distractor_sound_dict = {}

        self.rir_sampling_rate = 16000
        self.num_samples_per = 25000

        self.depth_dir = depth_dir
        self.return_depth = return_depth

        # ===== Distractor settings =====
        self.has_distractor = bool(has_distractor_sound)
        self.distractor_prob = float(distractor_prob)
        self.distractor_snr_db = distractor_snr_db

        # 记录目标/干扰可用 wav 列表
        self.source_sound_files = [f for f in os.listdir(self.source_sound_dir) if f.endswith(".wav")]
        if len(self.source_sound_files) == 0:
            raise RuntimeError(f"No sound files in {self.source_sound_dir}")

        self.distractor_sound_files = [f for f in os.listdir(self.distractor_sound_dir) if f.endswith(".wav")]
        if self.has_distractor and len(self.distractor_sound_files) == 0:
            raise RuntimeError(f"No distractor sound files in {self.distractor_sound_dir}")

        # ===== 1) 磁盘 cache 路径 =====
        os.makedirs(cache_dir, exist_ok=True)
        if cache_name is None:
            cache_name = f"mp3d_{split}_pairs{self.num_samples_per}_polar{int(use_polar_coordinates)}.npz"
        cache_path = os.path.join(cache_dir, cache_name)

        # ===== 2) 如果 cache 存在且不 rebuild：直接读取 =====
        if (not rebuild_cache) and os.path.exists(cache_path):
            print(f"[AudioGoalDataset] Load cache: {cache_path}")
            data = np.load(cache_path, allow_pickle=True)

            self.files = data["files"].tolist()
            goals_np = data["goals"].astype(np.float32)
            self.goals = [to_tensor(g) for g in goals_np]

        else:
            # ===== 3) 否则：生成并保存 cache =====
            print(f"[AudioGoalDataset] Build cache: {cache_path}")
            print("[AudioGoalDataset] source sound files:", self.source_sound_files)
            print("[AudioGoalDataset] distractor sound dir:", self.distractor_sound_dir)

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

                    sound_file = random.choice(self.source_sound_files)
                    cat = sound_file[:-4]
                    if cat not in CATEGORY_INDEX_MAPPING:
                        continue
                    index = CATEGORY_INDEX_MAPPING[cat]

                    angle = random.choice([0, 90, 180, 270])
                    rir_file = os.path.join(self.binaural_rir_dir, scene, str(angle), f"{r}_{s}.wav")
                    if not os.path.exists(rir_file):
                        continue

                    self.files.append((rir_file, sound_file))

                    delta_x = scene_graph.nodes[s]["point"][0] - scene_graph.nodes[r]["point"][0]
                    delta_y = scene_graph.nodes[s]["point"][2] - scene_graph.nodes[r]["point"][2]
                    goal_xy = self._compute_goal_xy(delta_x, delta_y, angle, use_polar_coordinates)

                    goal = to_tensor(np.zeros(3, dtype=np.float32))
                    goal[0] = float(index)
                    goal[1:] = goal_xy
                    goals.append(goal)

                    kept += 1

                self.goals += goals

            if len(self.files) == 0:
                raise RuntimeError("Built 0 samples. Check rir dir / mapping / scenes.")

            goals_np = torch.stack(self.goals, dim=0).cpu().numpy().astype(np.float32)
            files_np = np.array(self.files, dtype=object)

            meta = dict(
                split=split,
                distractor_split=distractor_split,
                use_polar_coordinates=bool(use_polar_coordinates),
                num_samples=len(self.files),
                binaural_rir_dir=self.binaural_rir_dir,
                source_sound_dir=self.source_sound_dir,
                distractor_sound_dir=self.distractor_sound_dir,
            )

            np.savez_compressed(cache_path, files=files_np, goals=goals_np, meta=np.array(meta, dtype=object))
            print(f"[AudioGoalDataset] Saved cache: {cache_path} (N={len(self.files)})")

        self.data = [None] * len(self.goals)

        # 加载两套音频
        self.load_source_sounds()
        if self.has_distractor:
            self.load_distractor_sounds()

    # ===== 音频长度（按字典选择）=====
    def audio_length(self, sound_file, which="source"):
        d = self.source_sound_dict if which == "source" else self.distractor_sound_dict
        return d[sound_file].shape[0] // self.rir_sampling_rate

    def load_source_sounds(self):
        for sound_file in self.source_sound_files:
            audio_data, _ = librosa.load(
                os.path.join(self.source_sound_dir, sound_file),
                sr=self.rir_sampling_rate
            )
            self.source_sound_dict[sound_file] = audio_data

    def load_distractor_sounds(self):
        for sound_file in self.distractor_sound_files:
            audio_data, _ = librosa.load(
                os.path.join(self.distractor_sound_dir, sound_file),
                sr=self.rir_sampling_rate
            )
            self.distractor_sound_dict[sound_file] = audio_data

    @staticmethod
    def _parse_rir_path(rir_file: str):
        scene = os.path.basename(os.path.dirname(os.path.dirname(rir_file)))
        angle = int(os.path.basename(os.path.dirname(rir_file)))
        stem = os.path.splitext(os.path.basename(rir_file))[0]
        r_str, s_str = stem.split("_")
        r = int(r_str)
        s = int(s_str)
        return scene, angle, r, s

    def _get_depth_from_npy(self, scene: str, r: int, angle: int):
        dpath = os.path.join(self.depth_dir, scene, f"{r}_{angle}.npy")
        if not os.path.exists(dpath):
            return None
        depth = np.load(dpath, mmap_mode="r")
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

            spectrogram = to_tensor(self.compute_stft_phase_features(audiogoal))  # (F,T,2)
            goal = self.goals[item]
            theta = to_tensor(make_doa_gaussian(goal[1], goal[2]))
            distance = to_tensor(make_r_gaussian_1d(goal[2]))

            if self.return_depth:
                scene, angle, r, s = self._parse_rir_path(rir_file)
                depth_np = self._get_depth_from_npy(scene, r, angle)
                depth = torch.from_numpy(np.array(depth_np, copy=False)).float()
                inputs_outputs = ([spectrogram, depth], [theta, distance])
            else:
                inputs_outputs = ([spectrogram], [theta, distance])

            if self.use_cache:
                self.data[item] = inputs_outputs
        else:
            inputs_outputs = self.data[item]
        return inputs_outputs

    def compute_audiogoal(self, binaural_rir_file, sound_file):
        sampling_rate = self.rir_sampling_rate

        # ===== 1) target RIR =====
        try:
            _, binaural_rir = wavfile.read(binaural_rir_file)
        except ValueError:
            logging.warning(f"{binaural_rir_file} file is not readable")
            binaural_rir = np.zeros((sampling_rate, 2), dtype=np.float32)

        if binaural_rir is None or len(binaural_rir) == 0:
            logging.debug(f"Empty RIR file at {binaural_rir_file}")
            binaural_rir = np.zeros((sampling_rate, 2), dtype=np.float32)

        if binaural_rir.dtype != np.float32:
            binaural_rir = binaural_rir.astype(np.float32)

        # ===== 2) target sound segment =====
        current_source_sound = self.source_sound_dict[sound_file]
        max_start_sec = max(0, self.audio_length(sound_file, which="source") - 2)
        index = random.randint(0, max_start_sec)

        if index * sampling_rate - binaural_rir.shape[0] < 0:
            source_sound = current_source_sound[: (index + 1) * sampling_rate]
            binaural_convolved = np.array([
                fftconvolve(source_sound, binaural_rir[:, ch])
                for ch in range(binaural_rir.shape[-1])
            ], dtype=np.float32)
            audiogoal = binaural_convolved[:, index * sampling_rate: (index + 1) * sampling_rate]
        else:
            source_sound = current_source_sound[
                index * sampling_rate - binaural_rir.shape[0] : (index + 1) * sampling_rate
            ]
            binaural_convolved = np.array([
                fftconvolve(source_sound, binaural_rir[:, ch], mode="valid")
                for ch in range(binaural_rir.shape[-1])
            ], dtype=np.float32)
            audiogoal = binaural_convolved[:, :-1]

        # 对齐到 (2, sampling_rate)
        if audiogoal.shape[-1] != sampling_rate:
            audiogoal = (
                audiogoal[:, :sampling_rate]
                if audiogoal.shape[-1] > sampling_rate
                else np.pad(audiogoal, ((0, 0), (0, sampling_rate - audiogoal.shape[-1])), mode="constant")
            )

        # ===== 3) (optional) add distractor (from different folder) =====
        if self.has_distractor and (random.random() < self.distractor_prob):
            scene, angle, r, s = self._parse_rir_path(binaural_rir_file)

            # 在同 scene/angle 下找一个 receiver=r 的其他 source 作为 distractor_s
            distractor_s = None
            rir_dir = os.path.join(self.binaural_rir_dir, scene, str(angle))
            if os.path.isdir(rir_dir):
                all_rirs = [fn for fn in os.listdir(rir_dir) if fn.endswith(".wav")]
                random.shuffle(all_rirs)
                for fn in all_rirs[:200]:  # 限制一下，避免太慢
                    stem = os.path.splitext(fn)[0]
                    r2, s2 = stem.split("_")
                    r2 = int(r2); s2 = int(s2)
                    if r2 == r and s2 != s:
                        distractor_s = s2
                        break

            if distractor_s is not None:
                distractor_rir_file = os.path.join(self.binaural_rir_dir, scene, str(angle), f"{r}_{distractor_s}.wav")
                try:
                    _, distractor_rir = wavfile.read(distractor_rir_file)
                except ValueError:
                    logging.warning(f"{distractor_rir_file} file is not readable")
                    distractor_rir = np.zeros((sampling_rate, 2), dtype=np.float32)

                if distractor_rir is None or len(distractor_rir) == 0:
                    distractor_rir = np.zeros((sampling_rate, 2), dtype=np.float32)

                if distractor_rir.dtype != np.float32:
                    distractor_rir = distractor_rir.astype(np.float32)

                # ===== 关键修改：distractor_sound 从 distractor_sound_dir 读 =====
                distractor_sound_file = random.choice(self.distractor_sound_files)
                distractor_sound = self.distractor_sound_dict[distractor_sound_file]

                max_start_sec_d = max(0, self.audio_length(distractor_sound_file, which="distractor") - 2)
                index_d = random.randint(0, max_start_sec_d)

                if index_d * sampling_rate - distractor_rir.shape[0] < 0:
                    d_src = distractor_sound[: (index_d + 1) * sampling_rate]
                    d_conv = np.array([
                        fftconvolve(d_src, distractor_rir[:, ch])
                        for ch in range(distractor_rir.shape[-1])
                    ], dtype=np.float32)
                    distractor_convolved = d_conv[:, index_d * sampling_rate: (index_d + 1) * sampling_rate]
                else:
                    d_src = distractor_sound[
                        index_d * sampling_rate - distractor_rir.shape[0] : (index_d + 1) * sampling_rate
                    ]
                    d_conv = np.array([
                        fftconvolve(d_src, distractor_rir[:, ch], mode="valid")
                        for ch in range(distractor_rir.shape[-1])
                    ], dtype=np.float32)
                    distractor_convolved = d_conv[:, :-1]

                if distractor_convolved.shape[-1] != sampling_rate:
                    distractor_convolved = (
                        distractor_convolved[:, :sampling_rate]
                        if distractor_convolved.shape[-1] > sampling_rate
                        else np.pad(
                            distractor_convolved,
                            ((0, 0), (0, sampling_rate - distractor_convolved.shape[-1])),
                            mode="constant",
                        )
                    )

                # ===== SNR control：随机 0~60 dB =====
                snr_db = self.distractor_snr_db
                if snr_db == "random_0_60":
                    snr_db = random.uniform(0.0, 60.0)
                    # print(snr_db)
                if snr_db is not None:
                    s_sig = audiogoal.astype(np.float32)
                    n_sig = distractor_convolved.astype(np.float32)
                    eps = 1e-12
                    Ps = float(np.mean(s_sig ** 2) + eps)
                    Pn = float(np.mean(n_sig ** 2) + eps)
                    target_Pn = Ps / (10.0 ** (float(snr_db) / 10.0))
                    alpha = np.sqrt(target_Pn / Pn)
                    distractor_convolved = (alpha * n_sig).astype(np.float32)

                audiogoal = audiogoal + distractor_convolved

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
    SPLIT = "train"  
    USE_CACHE = True  
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
        has_distractor_sound=True,
        distractor_prob=1.0,
        distractor_snr_db="random_0_60",
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
