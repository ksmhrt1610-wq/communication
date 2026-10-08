"""文字起こし(faster-whisper、すべてローカルで実行)。"""

from __future__ import annotations

import sys

import numpy as np

# フィラーを省略させないため、フィラーを含む文を initial_prompt に入れる
INITIAL_PROMPT = "えー、あのー、えっと、今日はですね、まあ、なんか、うーん、そのー、話してみます。"
DEFAULT_MODEL = "medium"


def _resolve_device(device: str, compute_type: str) -> tuple[str, str]:
    if device == "auto":
        try:
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type


def transcribe(
    y: np.ndarray,
    model: str = DEFAULT_MODEL,
    device: str = "auto",
    compute_type: str = "auto",
) -> dict:
    """16kHz モノラルの配列を文字起こしし、セグメントと単語のタイムスタンプを返す。"""
    from faster_whisper import WhisperModel

    device, compute_type = _resolve_device(device, compute_type)
    print(f"文字起こし中(モデル {model}、{device}/{compute_type})...", file=sys.stderr)
    wm = WhisperModel(model, device=device, compute_type=compute_type)
    segments, _info = wm.transcribe(
        np.asarray(y, dtype=np.float32),
        language="ja",
        word_timestamps=True,
        vad_filter=False,
        initial_prompt=INITIAL_PROMPT,
        beam_size=5,
    )
    out = []
    for seg in segments:
        words = [
            {"start": round(w.start, 2), "end": round(w.end, 2), "word": w.word}
            for w in (seg.words or [])
        ]
        out.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": seg.text, "words": words})
    return {
        "model": model,
        "device": device,
        "compute_type": compute_type,
        "initial_prompt": INITIAL_PROMPT,
        "segments": out,
    }
