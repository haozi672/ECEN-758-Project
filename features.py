import os
import numpy as np
import librosa
from tqdm import tqdm
from joblib import dump, load
from config import SR, N_MFCC, FEATURES_CACHE, FORCE_REBUILD
from config import CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

def _zscore_per_clip(M: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = float(M.mean())
    sd = float(M.std())
    return (M - mu) / (sd + eps)

def logmel_stats_from_wave(
    y: np.ndarray,
    sr: int = SR,
    n_mels: int = 128,
    hop: int = 1024,
    fmin: int = 50,
    fmax: int | None = None,
    use_zscore: bool = True,
    percentiles: tuple[int, int] = (10, 90),
) -> np.ndarray:
    if fmax is None:
        fmax = sr // 2
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,hop_length=hop, fmin=fmin, fmax=fmax, power=2.0)
    L = librosa.power_to_db(S, ref=np.max).astype(np.float32)
    if use_zscore:
        L = _zscore_per_clip(L)
    p_lo, p_hi = np.percentile(L, percentiles[0], axis=1), np.percentile(L, percentiles[1], axis=1)
    feat = np.concatenate([L.mean(axis=1), L.std(axis=1), p_lo, p_hi]).astype(np.float32)
    return feat

def build_logmel_stats_matrix(
    paths: list[str],
    sr: int = SR,
    n_mels: int = 128,
    hop: int = 1024,
    fmin: int = 50,
    fmax: int | None = None,
    use_zscore: bool = True,
    percentiles: tuple[int, int] = (10, 90),
    cache_name: str | None = None,
) -> np.ndarray:
    if fmax is None:
        fmax = sr // 2
    if cache_name is None:
        cache_name = os.path.join(_CACHE_DIR, f"logmel_stats_m{n_mels}_sr{sr}_h{hop}_"
                                              f"f{fmin}-{fmax}_z{int(use_zscore)}_p{percentiles[0]}-{percentiles[1]}.joblib")
    try:
        payload = load(cache_name)
        ok = (
            payload.get("sr") == sr and payload.get("n_mels") == n_mels and
            payload.get("hop") == hop and payload.get("fmin") == fmin and
            payload.get("fmax") == fmax and payload.get("use_zscore") == use_zscore and
            tuple(payload.get("percentiles", (10, 90))) == percentiles and
            payload.get("paths") == list(paths) and
            tuple(payload.get("shape", ())) == (len(paths), 4 * n_mels)
        )
        if ok:
            print(f"Using cached log-mel stats: {cache_name}")
            return payload["X"]
        else:
            print("Rebuilding cache")
    except Exception:
        pass

    feats = []
    for p in tqdm(paths, desc="extracting logmel pooled stats"):
        y, _ = librosa.load(p, sr=sr, mono=True)
        feats.append(logmel_stats_from_wave(y, sr=sr, n_mels=n_mels, hop=hop,
                                            fmin=fmin, fmax=fmax, use_zscore=use_zscore,
                                            percentiles=percentiles))
    X = np.asarray(feats, dtype=np.float32)
    dump({
        "sr": sr, "n_mels": n_mels, "hop": hop, "fmin": fmin, "fmax": fmax,
        "use_zscore": use_zscore, "percentiles": percentiles,
        "paths": list(paths), "shape": tuple(X.shape), "X": X
    }, cache_name)
    print(f"saved logmel stats to {cache_name} (shape={X.shape})")
    return X

def extract_from_wave(y: np.ndarray, sr: int = SR, n_mfcc: int = N_MFCC) -> np.ndarray:
    M = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    feat = np.concatenate([M.mean(axis=1), M.std(axis=1)]).astype(np.float32)
    return feat

def extract_from_path(path: str, sr: int = SR, n_mfcc: int = N_MFCC) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return extract_from_wave(y, sr=sr, n_mfcc=n_mfcc)

def _make_cache_payload(X: np.ndarray, paths: list[str], sr: int, n_mfcc: int) -> dict:
    return {
        "version": 1,
        "sr": sr,
        "n_mfcc": n_mfcc,
        "paths": list(paths),
        "shape": tuple(X.shape),
        "X": X.astype(np.float32),
    }

def _cache_is_compatible(payload: dict, paths: list[str], sr: int, n_mfcc: int) -> bool:
    try:
        return (
            isinstance(payload, dict)
            and payload.get("version", 0) == 1
            and payload.get("sr") == sr
            and payload.get("n_mfcc") == n_mfcc
            and payload.get("paths") == list(paths)
            and tuple(payload.get("shape", ())) == (len(paths), 2 * n_mfcc)
            and isinstance(payload.get("X"), np.ndarray)
        )
    except Exception:
        return False

def build_feature_matrix(
    paths: list[str],
    cache_path: str | None = None,
    sr: int = SR,
    n_mfcc: int = N_MFCC,
    show_progress: bool = True,
) -> np.ndarray:
    paths = list(paths)

    if cache_path is None:
        cache_path = FEATURES_CACHE

    if cache_path and os.path.exists(cache_path) and not FORCE_REBUILD:
        try:
            payload = load(cache_path)
            if _cache_is_compatible(payload, paths, sr, n_mfcc):
                print(f"ssing cached features: {cache_path}")
                return payload["X"]
            else:
                print("existing cache is incompatible (params or file list changed).")
        except Exception as e:
            print(f"failed to read cache ({e}).")

    feats = []
    iterator = tqdm(paths, desc="Extracting MFCC features") if show_progress else paths
    for p in iterator:
        feats.append(extract_from_path(p, sr=sr, n_mfcc=n_mfcc))
    X = np.asarray(feats, dtype=np.float32)

    if cache_path:
        payload = _make_cache_payload(X, paths, sr, n_mfcc)
        dump(payload, cache_path)
        print(f"[cache] Saved features to {cache_path} (shape={X.shape})")

    return X
