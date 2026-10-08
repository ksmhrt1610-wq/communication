"""CLI の通し動作(文字起こしはスタブに差し替え。ffmpeg と librosa は実際に使う)。"""

import json
import shutil
import subprocess

import numpy as np
import pytest

pytest.importorskip("librosa")
pytest.importorskip("fugashi")

from commtrain import cli, transcribe  # noqa: E402
from commtrain.pitch import analyze_pitch  # noqa: E402
from tests.test_text import SEGMENTS  # noqa: E402

SR = 16000


def tone(freq, sec, amp=0.3):
    t = np.arange(int(sec * SR)) / SR
    return amp * np.sin(2 * np.pi * freq * t)


def test_pitch_semitone_range():
    # 150 / 200 / 250 Hz を同じ長さで → 中央値 200Hz、半音値 −4.98 / 0 / +3.86
    y = np.concatenate([tone(150, 1.0), tone(200, 1.0), tone(250, 1.0)]).astype(np.float32)
    b = analyze_pitch(y)
    assert b["f0_median_hz"] == pytest.approx(200, abs=3)
    assert b["semitone_range_5_95"] == pytest.approx(4.98 + 3.86, abs=0.5)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg が必要")
def test_analyze_packet_compare_history(tmp_path, monkeypatch, capsys):
    rng = np.random.default_rng(0)
    y = np.concatenate([
        np.zeros(8000), tone(180, 3.0), np.zeros(19200), tone(220, 4.0), np.zeros(8000),
    ]) + 1e-4 * rng.standard_normal(8000 * 2 + 48000 + 19200 + 64000)
    wav = tmp_path / "rec.wav"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "f32le", "-ar", "16000", "-ac", "1", "-i", "-", str(wav)],
        input=y.astype(np.float32).tobytes(), check=True,
    )
    m4a = tmp_path / "rec.m4a"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(wav), "-ar", "44100", "-ac", "2", str(m4a)], check=True)

    def fake_transcribe(y, model="medium", device="auto", compute_type="auto"):
        return {"model": model, "device": "cpu", "compute_type": "int8",
                "initial_prompt": transcribe.INITIAL_PROMPT, "segments": SEGMENTS}

    monkeypatch.setattr(transcribe, "transcribe", fake_transcribe)
    hist = tmp_path / "history.csv"
    monkeypatch.setenv("COMMTRAIN_HISTORY", str(hist))

    assert cli.main(["analyze", str(m4a), "--task", "自己紹介", "--week", "1"]) == 0
    m = json.loads((tmp_path / "rec_metrics.json").read_text(encoding="utf-8"))
    # 0.5 + 3.0 + 1.2 + 4.0 + 0.5 秒(AAC のエンコーダ遅延で数十ms 伸びることがある)
    assert m["A"]["duration_sec"] == pytest.approx(9.2, abs=0.1)
    assert m["A"]["pauses"]["count"] == 1
    assert m["A"]["pauses"]["max_sec"] == pytest.approx(1.2, abs=0.05)
    assert m["B"]["f0_median_hz"] == pytest.approx(220, abs=5)
    report = (tmp_path / "rec_report.md").read_text(encoding="utf-8")
    assert "フィラー数は下限値" in report
    assert hist.exists()

    assert cli.main(["packet", str(m4a)]) == 0
    packet = (tmp_path / "rec_packet.md").read_text(encoding="utf-8")
    assert "課題:自己紹介" in packet and "Week 1" in packet
    assert "次の6要素を1〜5点で採点してください" in packet

    assert cli.main(["analyze", str(m4a), "--task", "自己紹介", "--week", "2"]) == 0
    capsys.readouterr()
    assert cli.main(["compare", "--week-a", "1", "--week-b", "2", "--task", "自己紹介"]) == 0
    out = capsys.readouterr().out
    assert "| 発声割合(%) |" in out and "±0" in out
    assert cli.main(["history"]) == 0
    out = capsys.readouterr().out
    assert "## Week 1" in out and "## Week 2" in out
