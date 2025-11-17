import numpy as np
import librosa
from typing import List, Callable
from config import SR as DEFAULT_SR
_RNG = np.random.default_rng(42)

def set_aug_seed(seed: int | None):
    global _RNG
    _RNG = np.random.default_rng(seed if seed is not None else None)

def _ensure_len(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    return y[:target_len]

def aug_time_shift(y: np.ndarray, sr: int = DEFAULT_SR, max_frac: float = 0.1) -> np.ndarray:
    L = len(y)
    if L == 0: return y
    n = int(max_frac * L)
    if n <= 0: return y
    shift = int(_RNG.integers(-n, n + 1))
    return np.roll(y, shift).astype(np.float32)

def aug_time_stretch(y: np.ndarray, sr: int = DEFAULT_SR, min_rate: float = 0.9, max_rate: float = 1.1) -> np.ndarray:
    L = len(y)
    if L == 0: return y
    rate = float(_RNG.uniform(min_rate, max_rate))
    if abs(rate - 1.0) < 1e-3:  # no-op
        return y.astype(np.float32)
    if L < 16:
        return y.astype(np.float32)
    z = librosa.effects.time_stretch(y.astype(np.float32), rate=rate)
    z = _ensure_len(z, L)
    return z.astype(np.float32)

def aug_pitch_shift(y: np.ndarray, sr: int = DEFAULT_SR, semitone_range: int = 2) -> np.ndarray:
    if len(y) == 0: return y
    steps = int(_RNG.integers(-semitone_range, semitone_range + 1))
    if steps == 0:
        return y.astype(np.float32)
    z = librosa.effects.pitch_shift(y.astype(np.float32), sr=sr, n_steps=steps)
    return z.astype(np.float32)

def aug_add_noise(y: np.ndarray, sr: int = DEFAULT_SR, snr_db: float = 20.0) -> np.ndarray:
    if len(y) == 0: return y
    y = y.astype(np.float32)
    sig_pow = float(np.mean(y**2) + 1e-12)
    noise_pow = sig_pow / (10 ** (snr_db / 10))
    noise = _RNG.normal(0.0, np.sqrt(noise_pow), size=y.shape).astype(np.float32)
    return (y + noise).astype(np.float32)

def aug_gain(y: np.ndarray, sr: int = DEFAULT_SR, min_db: float = -6.0, max_db: float = 6.0) -> np.ndarray:
    if len(y) == 0: return y
    db = float(_RNG.uniform(min_db, max_db))
    gain = 10 ** (db / 20.0)
    return (y.astype(np.float32) * gain).astype(np.float32)

FUNCS: List[Callable[..., np.ndarray]] = [
    aug_time_shift,
    aug_pitch_shift,
    aug_time_stretch,
    aug_add_noise,
    aug_gain,
]
def make_aug_variants(y: np.ndarray, n_aug: int, sr: int = DEFAULT_SR) -> list[np.ndarray]:
    outs = [y.astype(np.float32)]
    if n_aug <= 0:
        return outs
    for _ in range(n_aug):
        z = y.astype(np.float32)
        k = int(_RNG.integers(1, 3))
        for f in _RNG.choice(FUNCS, size=k, replace=False):
            z = f(z, sr=sr)
        outs.append(z.astype(np.float32))
    return outs
