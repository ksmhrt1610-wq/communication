"""指標A:間と発声(音声のみから計算)。

定義(過去の測定値と比較するため変更しないこと):
- 16kHz モノラル、RMS のフレーム長 400 サンプル、ホップ 160(10ms)
  フレームは中心揃え(両端を 200 サンプルのゼロで埋める。librosa.feature.rms の center=True と同じ)。
  フレーム i の時刻は i × 10ms。
- dB = 20·log10(RMS + 1e-9)
- 閾値 = 雑音(dB の10パーセンタイル)+ 0.35 ×(95パーセンタイル − 雑音)。閾値を超えるフレームを発声とする。
- 平滑化:150ms 未満の無音は発声に埋める(先頭・末尾の無音を除く)。その後、80ms 未満の発声は無音にする。
"""

from __future__ import annotations

import numpy as np

SR = 16000
FRAME_LENGTH = 400
HOP = 160
FRAME_SEC = HOP / SR  # 0.01 秒
THRESHOLD_RATIO = 0.35
FILL_SILENCE_SEC = 0.150
MIN_VOICED_SEC = 0.080
MIN_PAUSE_SEC = 0.15
WINDOW_SEC = 10.0


def frame_db(y: np.ndarray) -> np.ndarray:
    """中心揃えフレームの RMS を dB で返す。"""
    y = np.asarray(y, dtype=np.float64)
    pad = FRAME_LENGTH // 2
    ypad = np.pad(y, (pad, pad), mode="constant")
    frames = np.lib.stride_tricks.sliding_window_view(ypad, FRAME_LENGTH)[::HOP]
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    return 20.0 * np.log10(rms + 1e-9)


def runs(mask: np.ndarray) -> list[tuple[bool, int, int]]:
    """ブール配列を (値, 開始, 終了[排他]) の連続区間に分ける。"""
    out: list[tuple[bool, int, int]] = []
    n = len(mask)
    if n == 0:
        return out
    change = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    bounds = np.concatenate(([0], change, [n]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        out.append((bool(mask[a]), int(a), int(b)))
    return out


def smooth(voiced: np.ndarray) -> np.ndarray:
    """150ms 未満の無音(先頭・末尾以外)を埋め、その後 80ms 未満の発声を消す。"""
    v = voiced.copy()
    fill_frames = round(FILL_SILENCE_SEC / FRAME_SEC)  # 15
    min_voiced_frames = round(MIN_VOICED_SEC / FRAME_SEC)  # 8
    n = len(v)
    for val, a, b in runs(v):
        if not val and a > 0 and b < n and (b - a) < fill_frames:
            v[a:b] = True
    for val, a, b in runs(v):
        if val and (b - a) < min_voiced_frames:
            v[a:b] = False
    return v


def voiced_mask(y: np.ndarray) -> tuple[np.ndarray, dict]:
    db = frame_db(y)
    noise = float(np.percentile(db, 10))
    p95 = float(np.percentile(db, 95))
    thr = noise + THRESHOLD_RATIO * (p95 - noise)
    raw = db > thr
    return smooth(raw), {"noise_db": noise, "p95_db": p95, "threshold_db": thr}


def _r(x: float | None, nd: int = 2) -> float | None:
    return None if x is None else round(float(x), nd)


def analyze_pauses(y: np.ndarray, sr: int = SR) -> dict:
    if sr != SR:
        raise ValueError(f"サンプリング周波数は {SR}Hz である必要があります")
    duration = len(y) / sr
    mask, levels = voiced_mask(y)
    n = len(mask)
    rs = runs(mask)
    voiced_runs = [(a, b) for val, a, b in rs if val]

    result: dict = {
        "duration_sec": _r(duration),
        "levels_db": {k: _r(v, 1) for k, v in levels.items()},
    }

    # 10秒ごとの発声率(録音全体を 0 秒から 10 秒ずつ区切る)
    win = round(WINDOW_SEC / FRAME_SEC)
    windows = []
    for i in range(0, n, win):
        seg = mask[i:i + win]
        windows.append({
            "start_sec": _r(i * FRAME_SEC, 1),
            "end_sec": _r(min(i + win, n) * FRAME_SEC, 1),
            "voiced_pct": _r(100.0 * seg.mean(), 1),
        })
    result["voiced_pct_per_10s"] = windows

    if not voiced_runs:
        result.update({
            "first_onset_sec": None, "speech_start_sec": None, "speech_end_sec": None,
            "speech_span_sec": 0.0, "voiced_sec": 0.0, "voiced_ratio_pct": None,
            "voiced_ratio_of_total_pct": 0.0,
            "chunks": {"count": 0, "mean_sec": None, "max_sec": None},
            "pauses": _pause_stats([], 0.0),
            "halves": None,
            "pauses_list": [],
        })
        return result

    start_f = voiced_runs[0][0]
    end_f = voiced_runs[-1][1]
    span = (end_f - start_f) * FRAME_SEC
    voiced_sec = float(mask.sum()) * FRAME_SEC
    chunk_lens = [(b - a) * FRAME_SEC for a, b in voiced_runs]

    pauses = []
    for val, a, b in rs:
        if not val and a >= start_f and b <= end_f:
            dur = (b - a) * FRAME_SEC
            if dur >= MIN_PAUSE_SEC - 1e-9:
                pauses.append({"start_sec": _r(a * FRAME_SEC), "end_sec": _r(b * FRAME_SEC), "dur_sec": _r(dur)})

    mid = (start_f + end_f) / 2 * FRAME_SEC
    first = [p for p in pauses if (p["start_sec"] + p["end_sec"]) / 2 < mid]
    second = [p for p in pauses if (p["start_sec"] + p["end_sec"]) / 2 >= mid]

    result.update({
        "first_onset_sec": _r(start_f * FRAME_SEC),
        "speech_start_sec": _r(start_f * FRAME_SEC),
        "speech_end_sec": _r(end_f * FRAME_SEC),
        "speech_span_sec": _r(span),
        "voiced_sec": _r(voiced_sec),
        "voiced_ratio_pct": _r(100.0 * voiced_sec / span, 1) if span > 0 else None,
        "voiced_ratio_of_total_pct": _r(100.0 * voiced_sec / duration, 1) if duration > 0 else None,
        "chunks": {
            "count": len(chunk_lens),
            "mean_sec": _r(float(np.mean(chunk_lens))),
            "max_sec": _r(max(chunk_lens)),
        },
        "pauses": _pause_stats(pauses, span),
        "halves": {
            "midpoint_sec": _r(mid),
            "first": {"count": len(first), "ge_0_5": sum(p["dur_sec"] >= 0.5 for p in first)},
            "second": {"count": len(second), "ge_0_5": sum(p["dur_sec"] >= 0.5 for p in second)},
        },
        "pauses_list": pauses,
    })
    return result


def _pause_stats(pauses: list[dict], span: float) -> dict:
    durs = [p["dur_sec"] for p in pauses]
    minutes = span / 60.0
    ge05 = sum(d >= 0.5 for d in durs)
    ge10 = sum(d >= 1.0 for d in durs)

    def per_min(c: int) -> float | None:
        return _r(c / minutes) if minutes > 0 else None

    return {
        "count": len(durs),
        "median_sec": _r(float(np.median(durs))) if durs else None,
        "ge_0_5": ge05,
        "ge_1_0": ge10,
        "max_sec": _r(max(durs)) if durs else None,
        "per_min": per_min(len(durs)),
        "ge_0_5_per_min": per_min(ge05),
        "ge_1_0_per_min": per_min(ge10),
        "long_pauses": [p for p in pauses if p["dur_sec"] >= 1.0],
    }
