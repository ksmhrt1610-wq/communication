"""音声の読み込み(ffmpeg で 16kHz モノラルに変換)。"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

SR = 16000
SUPPORTED_EXTS = {".wav", ".m4a", ".mp3"}


def load_audio(path: str | Path, sr: int = SR) -> np.ndarray:
    """ffmpeg で任意の音声を sr Hz モノラル float32 に変換して返す。"""
    path = Path(path)
    if not path.exists():
        msg = f"音声ファイルが見つかりません: {path}"
        if path.parent.is_dir():
            near = sorted(p.name for p in path.parent.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
            if near:
                msg += "\n同じフォルダにある音声ファイル:\n" + "\n".join(f"  {n}" for n in near[:20])
        raise FileNotFoundError(msg)
    if path.suffix.lower() not in SUPPORTED_EXTS:
        print(
            f"注意: {path.suffix} は想定外の形式です(wav / m4a / mp3 を想定)。ffmpeg で読み込みを試みます。",
            file=sys.stderr,
        )
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg が見つかりません。インストールして PATH に通してください。")
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(path),
        "-ac", "1", "-ar", str(sr),
        "-f", "f32le", "-acodec", "pcm_f32le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg での変換に失敗しました: {proc.stderr.decode(errors='replace')}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()
