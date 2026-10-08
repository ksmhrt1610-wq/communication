"""指標B:声の高さ(librosa.pyin)。

- fmin 70Hz、fmax 400Hz、frame_length 1024、hop 160
- 有声フレームの F0 中央値(Hz)
- 中央値を基準にした半音値 12·log2(F0 / 中央値) の 5〜95パーセンタイル幅と標準偏差
- 音量のばらつきはスマホのノイズ抑制で歪むため指標に含めない
"""

from __future__ import annotations

import numpy as np

SR = 16000
FMIN = 70.0
FMAX = 400.0
FRAME_LENGTH = 1024
HOP = 160


def semitone_stats(f0_voiced: np.ndarray) -> dict:
    f0 = np.asarray(f0_voiced, dtype=np.float64)
    f0 = f0[np.isfinite(f0) & (f0 > 0)]
    if len(f0) == 0:
        return {"voiced_frames": 0, "f0_median_hz": None, "semitone_range_5_95": None, "semitone_std": None}
    med = float(np.median(f0))
    st = 12.0 * np.log2(f0 / med)
    return {
        "voiced_frames": int(len(f0)),
        "f0_median_hz": round(med, 1),
        "semitone_range_5_95": round(float(np.percentile(st, 95) - np.percentile(st, 5)), 2),
        "semitone_std": round(float(np.std(st)), 2),
    }


def analyze_pitch(y: np.ndarray, sr: int = SR) -> dict:
    import librosa

    f0, voiced_flag, _ = librosa.pyin(
        np.asarray(y, dtype=np.float32), fmin=FMIN, fmax=FMAX, sr=sr,
        frame_length=FRAME_LENGTH, hop_length=HOP,
    )
    return semitone_stats(f0[voiced_flag])
