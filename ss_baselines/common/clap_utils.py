import numpy as np
import librosa
import torch
import laion_clap
import csv
import torch
from collections import deque
import os
import torch.nn.functional as F
from collections import defaultdict

def int16_to_float32(x):
    return (x / 32767.0).astype('float32')

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype('int16')

def load_id_to_text(csv_path):
    id_to_text = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["id"])
            txt = row["text"].strip()
            id_to_text[idx] = txt
    print(f"[INFO] Loaded {len(id_to_text)} labels from {csv_path}")
    return id_to_text

def retrieve_topk(audio_embed, text_embed, k=3):
    audio_embed = torch.as_tensor(audio_embed).detach()
    text_embed = torch.as_tensor(text_embed).detach()
    if audio_embed.dim() == 1:
        audio_embed = audio_embed.unsqueeze(0)

    if audio_embed.size(-1) != text_embed.size(-1):
        raise ValueError(
            f"Dim mismatch: audio_embed dim={audio_embed.size(-1)}, "
            f"text_embed dim={text_embed.size(-1)}"
        )
    sim = audio_embed @ text_embed.t()
    k = min(k, sim.size(1))
    topk_scores, topk_ids = torch.topk(sim, k=k, dim=1)  # 两个都是 [N_audio, K]

    return topk_ids, topk_scores

def topk_ids_to_texts(topk_ids, id_to_text):
    texts_all = []
    for row in topk_ids:
        row_texts = []
        for idx in row.tolist():
            row_texts.append(id_to_text.get(int(idx), "UNKNOWN_ID"))
        texts_all.append(row_texts)
    return texts_all

def extract_object_from_text(text: str) -> str:

    s = text.strip()

    prefixes = ["sound of a ", "sound of an ", "sound of "]
    s_lower = s.lower()
    for p in prefixes:
        if s_lower.startswith(p):
            s = s[len(p):]
            s_lower = s_lower[len(p):]
            break

    tokens = s.split()
    tokens_lower = s_lower.split()

    if not tokens:
        return ""

    verb_candidates = {
        "ringing", "running", "blowing", "sizzling", "boiling",
        "playing", "spinning", "tumbling", "humming", "clicking",
        "typing", "knocking", "rustling", "pouring", "dripping",
        "bubbling", "flushing", "opening", "closing", "locking",
        "sliding", "tearing", "crumpling", "popping", "beeping",
        "striking", "squeezing", "sweeping", "spraying", "stirring",
        "grinding", "chirping"
    }

    cut_idx = len(tokens) 
    for i, w in enumerate(tokens_lower):
        if w in verb_candidates:
            cut_idx = i
            break
        if w.endswith("ing") and len(w) > 4:
            cut_idx = i
            break
        if w == "being":
            cut_idx = i
            break

    obj_tokens = tokens[:cut_idx]
    if not obj_tokens:
        obj_tokens = tokens

    obj_name = " ".join(obj_tokens)
    return obj_name


def extract_object_phrase(text: str) -> str:
    s = text.strip()
    s_lower = s.lower()
    prefixes = [
        "sound of a ",
        "sound of an ",
        "sound of the ",
        "sound of "
    ]
    for p in prefixes:
        if s_lower.startswith(p):
            s = s[len(p):]
            s_lower = s_lower[len(p):]
            break

    tokens = s_lower.split()
    orig_tokens = s.split()
    if not tokens:
        return ""
    stop_words = {
        "being", "as", "when", "while", "with",
        "on", "in", "at", "by", "to", "for",
        "from", "after", "before", "into", "onto"
    }

    end = len(tokens)
    for i, w in enumerate(tokens):
        if w in stop_words:
            end = i
            break
        if w.endswith("ing") and len(w) > 3:
            end = i
            break

    if end <= 0:
        phrase_tokens = orig_tokens
    else:
        phrase_tokens = orig_tokens[:end]

    object_phrase = " ".join(phrase_tokens).strip()
    return object_phrase

OBJECT_PATTERNS = [
    # 多词优先
    ("chest of drawers", "drawers"),
    ("tv monitor",        "tv monitor"),
    ("sofa cushion",      "cushion"),
    ("seat cushion",      "cushion"),
    ("plant pot",         "plant"),
    ("plant leaves",      "plant"),
    ("shower head",       "shower"),
    ("treadmill belt",    "treadmill"),
    ("treadmill motor",   "treadmill"),
    ("rowing machine",    "treadmill"),
    ("weight machine",    "treadmill"),
    ("weight plates",     "treadmill"),
    ("exercise bike",     "treadmill"),
    ("wooden bed",        "bed"),
    ("wooden stool",      "stool"),
    ("metal stool",       "stool"),
    ("bathtub water",      "bathtub"),
    # 单词类（基础物体）
    ("picture",   "picture"),
    ("camera",    "picture"),   # 保险：如果以后有 camera-only 句子，仍映射到 picture 类
    ("chair",     "chair"),
    ("table",     "table"),
    ("cabinet",   "cabinet"),
    ("cushion",   "cushion"),
    ("sofa",      "sofa"),
    ("bed",       "bed"),
    ("drawer",    "drawers"),
    ("plant",     "plant"),
    ("sink",      "sink"),
    ("faucet",    "sink"),      # 如果 object phrase 只有 faucet，也映射到 sink 类
    ("toilet",    "toilet"),
    ("stool",     "stool"),
    ("towel",     "towel"),
    ("counter",   "counter"),
    ("fireplace", "fireplace"),
    ("treadmill", "treadmill"),
    ("clothes",   "clothes"),
    
]

WATER_FALLBACK_OBJECTS = [
    ("bathtub",  "bathtub"),
    ("sink",     "sink"),
    ("shower",   "shower"),
    ("toilet",   "toilet"),
    ("counter",  "counter"),
    ("fireplace","fireplace"),
]


def extract_object_name(text: str) -> str:
    phrase = extract_object_phrase(text)
    phrase_lower = phrase.lower()
    text_lower = text.lower()

    for pattern, canon in OBJECT_PATTERNS:
        if pattern in phrase_lower:
            return canon

    if phrase_lower.startswith("water"):
        for kw, canon in WATER_FALLBACK_OBJECTS:
            if kw in text_lower:
                return canon
        return "water"

    for pattern, canon in OBJECT_PATTERNS:
        if pattern in text_lower:
            return canon
    return phrase_lower

def stereo_to_mono_energy_weighted(x, eps=1e-8):
    """
    x: torch.Tensor, shape (1,2,T)
    return: (1,T)
    """
    assert x.ndim == 3 and x.shape[0] == 1 and x.shape[1] == 2
    L = x[:, 0, :]  # (1,T)
    R = x[:, 1, :]  # (1,T)

    # 能量（均方）
    eL = (L * L).mean(dim=-1, keepdim=True)  # (1,1)
    eR = (R * R).mean(dim=-1, keepdim=True)  # (1,1)

    wL = eL / (eL + eR + eps)  # (1,1)
    wR = 1.0 - wL

    mono = wL * L + wR * R
    return mono

class AudioHistory5s:
    """
    Keep last window_seconds of audio.
    - Only append when intensity > 0
    - Stores chunks on CPU to save GPU mem
    """
    def __init__(self, window_seconds=5.0, sr=48000):
        self.sr = int(sr)
        self.window_samples = int(window_seconds * self.sr)
        self.buf = deque()           # each: torch (1,2,T) on CPU
        self.total_samples = 0

    def clear(self):
        self.buf.clear()
        self.total_samples = 0

    def append(self, waveform_1_2_T: torch.Tensor):
        x = waveform_1_2_T.detach().cpu()
        self.buf.append(x)
        self.total_samples += x.shape[-1]

        # trim to last window_samples
        while self.total_samples > self.window_samples and len(self.buf) > 0:
            y = self.buf.popleft()
            self.total_samples -= y.shape[-1]

    def concat(self, device="cuda"):
        if len(self.buf) == 0:
            return None
        x = torch.cat(list(self.buf), dim=-1)  # (1,2,T_total)
        return x.to(device)

# ---------- get top unique names from history ----------
def top_unique_names_from_history(
    audio_hist: AudioHistory5s,
    model_audio,
    retrieve_topk,
    text_embed,
    topk_ids_to_texts,
    id_to_text,
    extract_object_name,
    device="cuda",
    k_retrieve=5,
    keep_top=3,
):
    x = audio_hist.concat(device=device)
    if x is None:
        return []

    x_mono = stereo_to_mono_energy_weighted(x)  # (1,T_total)
    audio_embed = model_audio.get_audio_embedding_from_data(x=x_mono, use_tensor=True)

    topk_ids, topk_scores = retrieve_topk(audio_embed, text_embed, k=k_retrieve)
    topk_texts = topk_ids_to_texts(topk_ids, id_to_text)

    seen = set()
    out = []
    i = 0
    for txt in topk_texts[i]:
        name = extract_object_name(text=txt)
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
        if len(out) >= keep_top:
            break
    return out

# ---------- main periodic updater ----------
def step_update_sliding5s(
    waveform,                 # torch (1,2,T)
    audio_intensity,          # float
    step_idx,                 # int
    N_update,                 # int
    audio_hist: AudioHistory5s,
    names,                    # list[str] persistent
    model_audio,
    retrieve_topk,
    text_embed,
    topk_ids_to_texts,
    id_to_text,
    extract_object_name,
    model_yolo,
    device="cuda",
    keep_top=3,               # take top-3 from audio
    set_top=3,                # yolo uses top-3 prompts
    max_names=20,             # avoid explosion
    k_retrieve=5,
):
    """
    Behavior you requested:
    - Keep last 5s audio in audio_hist (append only if intensity > 0)
    - If intensity == 0: do NOT append, and do NOT compute embedding update
    - Every N steps (while intensity > 0): compute top3 from history
    - If new name appears (not in names): append it
    - Only if names changed: call model_yolo.set_classes(...)
    """
    did_set = False
    top_now = []

    if audio_intensity <= 0:
        return names, did_set, top_now  # skip everything

    # accumulate 5s history
    audio_hist.append(waveform)

    # periodic update
    if (step_idx % N_update) != 0:
        return names, did_set, top_now

    top_now = top_unique_names_from_history(
        audio_hist=audio_hist,
        model_audio=model_audio,
        retrieve_topk=retrieve_topk,
        text_embed=text_embed,
        topk_ids_to_texts=topk_ids_to_texts,
        id_to_text=id_to_text,
        extract_object_name=extract_object_name,
        device=device,
        k_retrieve=k_retrieve,
        keep_top=keep_top,
    )

    # update candidate list + set yolo only if changed
    existing = set(names)
    changed = False
    for n in top_now:
        if n not in existing:
            names.append(n)
            existing.add(n)
            changed = True
            if len(names) >= max_names:
                break

    if changed:
        yolo_names = names
        model_yolo.set_classes(yolo_names, model_yolo.get_text_pe(yolo_names))
        did_set = True

    return names, did_set, top_now

class Warmup5ChooseTop1ThenLock:
    """
    - Warmup first K steps: accumulate audio retrieval scores across different names
      (no YOLO during warmup)
    - After warmup: pick top1 name, set YOLO with that single class, and lock.
    """

    def __init__(
        self,
        model_audio,
        retrieve_topk,
        text_embed,
        topk_ids_to_texts,
        id_to_text,
        extract_object_name,
        model_yolo,
        device="cuda",
        warmup_steps=5,
        k_retrieve=10,
        merge="max",          # "max" or "sum"
        intensity_eps=0.0,    # if intensity<=eps: skip update
    ):
        self.model_audio = model_audio
        self.retrieve_topk = retrieve_topk
        self.text_embed = text_embed
        self.topk_ids_to_texts = topk_ids_to_texts
        self.id_to_text = id_to_text
        self.extract_object_name = extract_object_name
        self.model_yolo = model_yolo
        self.device = device

        self.warmup_steps = int(warmup_steps)
        self.k_retrieve = int(k_retrieve)
        self.merge = merge
        self.intensity_eps = float(intensity_eps)

        self.reset()

    def reset(self):
        self.step_count = 0
        self.locked = False
        self.best_name = None
        self.score_bank = {}   # name -> accumulated score (float)

    def _accumulate_from_step(self, waveform_1_2_T):
        # mono
        x_mono = stereo_to_mono_energy_weighted(waveform_1_2_T)

        # embed + retrieve
        audio_embed = self.model_audio.get_audio_embedding_from_data(x=x_mono, use_tensor=True)
        topk_ids, topk_scores = self.retrieve_topk(audio_embed, self.text_embed, k=self.k_retrieve)
        topk_texts = self.topk_ids_to_texts(topk_ids, self.id_to_text)

        # aggregate by object name for this step (dedup within step)
        per_step = {}
        i = 0
        for rank in range(topk_ids.size(1)):
            s = float(topk_scores[i, rank])
            txt = topk_texts[i][rank]
            name = self.extract_object_name(text=txt)

            if name not in per_step:
                per_step[name] = s
            else:
                # within-step merge
                per_step[name] = max(per_step[name], s) if self.merge == "max" else (per_step[name] + s)

        # merge into global score_bank
        for name, s in per_step.items():
            if name not in self.score_bank:
                self.score_bank[name] = s
            else:
                self.score_bank[name] = max(self.score_bank[name], s) if self.merge == "max" else (self.score_bank[name] + s)

    def _finalize_and_lock(self):
        if len(self.score_bank) == 0:
            self.best_name = None
            self.locked = True
            return None

        self.best_name = max(self.score_bank.items(), key=lambda x: x[1])[0]
        # set YOLO with only this class (once)
        self.model_yolo.set_classes([self.best_name], self.model_yolo.get_text_pe([self.best_name]))
        self.locked = True
        return self.best_name

    def step(self, waveform, audio_intensity=1.0):
        """
        waveform: torch (1,2,T) on GPU
        audio_intensity: float; if <= intensity_eps, skip accumulating

        Returns:
          locked (bool),
          best_name (str or None),
          score_bank (dict)  # only meaningful during warmup
        """
        if self.locked:
            return True, self.best_name, self.score_bank

        if audio_intensity <= self.intensity_eps:
            # still count step? usually you DON'T want silence to consume warmup slots.
            # Here we do NOT increment step_count when silent.
            return False, None, self.score_bank

        # warmup accumulate
        self._accumulate_from_step(waveform)

        self.step_count += 1
        if self.step_count >= self.warmup_steps:
            self._finalize_and_lock()

        return self.locked, self.best_name, self.score_bank
    
def rms_normalize_clipped(x, target_rms=0.05, min_rms=0.005, max_gain=10.0, eps=1e-8):


    """
    x: torch.Tensor (1,T)
    """
    rms = torch.sqrt((x*x).mean(dim=-1, keepdim=True) + eps)

    # 太小的 rms 直接不放大（避免把噪声拉起来）
    rms_eff = torch.clamp(rms, min=min_rms)

    gain = target_rms / rms_eff
    gain = torch.clamp(gain, max=max_gain)

    return x * gain

def float32_to_int16(x):
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)

def int16_to_float32(x):
    x = np.asarray(x, dtype=np.int16)
    return (x.astype(np.float32) / 32767.0).astype(np.float32)

@torch.no_grad()
def embed_audio_whole_file(audio, model, sr=48000, device="cuda"):

    yq = int16_to_float32(float32_to_int16(audio))
    x = torch.from_numpy(yq).float().to(device)  # (1,T)

    emb = model.get_audio_embedding_from_data(x=x, use_tensor=True)  # (1,D) or (D,)
    if emb.ndim == 2:
        emb = emb[0]
    return emb.detach().cpu()  # (D,)

SPECIAL_MAP = {
    "chest_of_drawers": ["drawers"],
    "gym_equipment": ["treadmill"],
    "seating": ["sofa", "bed"],
    "tv_monitor": ["tv"]
}

def extract_object_classes_from_path(path: str, special_map=SPECIAL_MAP):
    """
    Return list[str] because some items map to multiple object classes.
    """
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0].strip().lower()
    return special_map.get(stem, [stem])

def topk_most_similar_objects(query_emb_D, db_pt_path, k=3, special_map=SPECIAL_MAP):
    """
    Returns:
      list of tuples: (sim, object_class)
    Note:
      - If a retrieved item maps to multiple object classes, we expand them.
      - We keep unique object classes, preserving order by similarity rank.
    """
    db = torch.load(db_pt_path, map_location="cpu")
    E = db["embeddings"]                 # (N,D)
    paths = db.get("paths", None)
    names = db.get("names", None)

    q = F.normalize(query_emb_D.unsqueeze(0), dim=1)   # (1,D)
    En = F.normalize(E, dim=1)                         # (N,D)

    sims = (q @ En.T).squeeze(0)                       # (N,)
    vals, idxs = torch.topk(sims, k=min(k, sims.numel()), largest=True)

    results = []
    seen = set()

    for v, i in zip(vals.tolist(), idxs.tolist()):
        if names is not None:
            # if you saved a clean label name, use it
            src = str(names[i])
            obj_list = special_map.get(src.strip().lower(), [src.strip().lower()])
        elif paths is not None:
            # parse from filename
            obj_list = extract_object_classes_from_path(paths[i], special_map=special_map)
        else:
            obj_list = [str(int(i))]

        for obj in obj_list:
            if obj in seen:
                continue
            seen.add(obj)
            results.append((float(v), obj))
            if len(results) >= k:
                return results

    return results


class AudioTopKAccumulator:
    def __init__(self, n_steps=5, topk=3, mode="ema", alpha=0.6, final_k=3):
        self.n_steps = int(n_steps)
        self.topk = int(topk)          # 每一步取多少个候选来更新bank
        self.mode = mode
        self.alpha = float(alpha)
        self.final_k = int(final_k)    # 最终输出多少个best
        self.reset()

    def reset(self):
        self.step = 0
        self.score_bank = defaultdict(float)
        self.locked = False
        self.best_names = []           # list[str]

    def _finalize_topk(self):
        if len(self.score_bank) == 0:
            self.best_names = []
        else:
            items = sorted(self.score_bank.items(), key=lambda x: x[1], reverse=True)
            self.best_names = [name for name, _ in items[: self.final_k]]

    def update(self, topk_objs):
        """
        topk_objs: list[(sim, obj)]
        Returns: (best_names, locked)
          - best_names is [] until locked
        """
        if self.locked:
            return self.best_names, True

        self.step += 1

        for sim, obj in topk_objs[: self.topk]:
            sim = float(sim)
            obj = str(obj)
            if self.mode == "ema":
                self.score_bank[obj] = self.alpha * sim + (1.0 - self.alpha) * self.score_bank[obj]
            else:
                self.score_bank[obj] += sim

        if self.step >= self.n_steps:
            self._finalize_topk()
            self.locked = True
            return self.best_names, True

        return [], False