"""指標A:既知の長さの無音を挟んだ合成音声で、間の回数と長さを確認する。"""

import numpy as np
import pytest

from commtrain.pauses import FRAME_SEC, analyze_pauses, smooth

SR = 16000
# 25ms(400サンプル)の窓が音の端にかかったフレームは発声になるため、
# 測定される間は実際の無音より最大で約1窓分(0.025秒)短くなる。
TOL = 0.04


def make_signal(pattern, seed=0):
    """pattern: [("tone"|"silence", 秒), ...]。無音部にも弱い雑音を入れる。"""
    rng = np.random.default_rng(seed)
    parts = []
    for kind, sec in pattern:
        n = int(round(sec * SR))
        noise = 1e-4 * rng.standard_normal(n)
        if kind == "tone":
            t = np.arange(n) / SR
            parts.append(0.3 * np.sin(2 * np.pi * 200 * t) + noise)
        else:
            parts.append(noise)
    return np.concatenate(parts).astype(np.float32)


PATTERN = [
    ("silence", 0.8),
    ("tone", 2.0),
    ("silence", 0.3),   # 間(0.15秒以上・0.5秒未満)
    ("tone", 1.5),
    ("silence", 0.7),   # 0.5秒以上
    ("tone", 2.5),
    ("silence", 1.2),   # 1.0秒以上
    ("tone", 1.0),
    ("silence", 0.1),   # 150ms 未満 → 発声に埋められる
    ("tone", 1.5),
    ("silence", 2.0),   # 1.0秒以上・最長
    ("tone", 2.0),
    ("silence", 0.6),   # 末尾の無音は間に数えない
]


@pytest.fixture(scope="module")
def result():
    return analyze_pauses(make_signal(PATTERN))


def test_duration_and_onset(result):
    total = sum(s for _, s in PATTERN)
    assert result["duration_sec"] == pytest.approx(total, abs=0.01)
    assert result["first_onset_sec"] == pytest.approx(0.8, abs=TOL)
    assert result["speech_end_sec"] == pytest.approx(total - 0.6, abs=TOL)
    assert result["speech_span_sec"] == pytest.approx(total - 0.8 - 0.6, abs=2 * TOL)


def test_pause_count_and_lengths(result):
    p = result["pauses"]
    durs = [x["dur_sec"] for x in result["pauses_list"]]
    expected = [0.3, 0.7, 1.2, 2.0]
    assert p["count"] == 4
    assert durs == pytest.approx(expected, abs=TOL)
    assert p["ge_0_5"] == 3
    assert p["ge_1_0"] == 2
    assert p["max_sec"] == pytest.approx(2.0, abs=TOL)
    assert p["median_sec"] == pytest.approx((0.7 + 1.2) / 2, abs=TOL)


def test_pause_positions(result):
    starts = [x["start_sec"] for x in result["pauses"]["long_pauses"]]
    # 1.2秒の間は 0.8+2.0+0.3+1.5+0.7+2.5 = 7.8 秒、2.0秒の間は 7.8+1.2+1.0+0.1+1.5 = 11.6 秒から
    assert starts == pytest.approx([7.8, 11.6], abs=TOL)


def test_chunks_and_voiced(result):
    c = result["chunks"]
    # 0.1秒の無音が埋められ、1.0秒と1.5秒のトーンは1つのかたまりになる
    assert c["count"] == 5
    assert c["max_sec"] == pytest.approx(2.6, abs=2 * TOL)
    voiced = 2.0 + 1.5 + 2.5 + 2.6 + 2.0
    assert result["voiced_sec"] == pytest.approx(voiced, abs=0.2)
    assert result["voiced_ratio_pct"] == pytest.approx(100 * voiced / result["speech_span_sec"], abs=1.5)


def test_halves_and_per_minute(result):
    h = result["halves"]
    # 発話区間 0.8〜15.3 秒の中点は約 8.05 秒。間の中央:3.0, 4.85, 8.4, 12.6 秒
    assert h["first"]["count"] == 2
    assert h["first"]["ge_0_5"] == 1
    assert h["second"]["count"] == 2
    assert h["second"]["ge_0_5"] == 2
    minutes = result["speech_span_sec"] / 60
    assert result["pauses"]["per_min"] == pytest.approx(4 / minutes, abs=0.05)
    assert result["pauses"]["ge_0_5_per_min"] == pytest.approx(3 / minutes, abs=0.05)


def test_voiced_pct_per_10s(result):
    w = result["voiced_pct_per_10s"]
    assert [x["start_sec"] for x in w] == [0, 10]
    # 0〜10秒:発声は 0.8〜10 秒のうち間を除いた部分
    expected0 = (2.0 + 1.5 + 2.5 + (10 - 9.0)) / 10 * 100
    assert w[0]["voiced_pct"] == pytest.approx(expected0, abs=2.0)


def test_smoothing_rules():
    f = lambda s: np.array([c == "1" for c in s])  # noqa: E731
    # 先頭・末尾の無音は短くても埋めない
    assert smooth(f("0001111111111")).tolist() == f("0001111111111").tolist()
    # 内部の 14 フレーム(140ms)の無音は埋める、15 フレーム(150ms)は残す
    v = f("1" * 10 + "0" * 14 + "1" * 10)
    assert smooth(v).all()
    v = f("1" * 10 + "0" * 15 + "1" * 10)
    assert smooth(v).sum() == 20
    # 7 フレーム(70ms)の発声は消す、8 フレーム(80ms)は残す
    assert not smooth(f("0" * 20 + "1" * 7 + "0" * 20)).any()
    assert smooth(f("0" * 20 + "1" * 8 + "0" * 20)).sum() == 8
    assert FRAME_SEC == 0.01


def test_silence_only():
    r = analyze_pauses(make_signal([("silence", 2.0)]) * 0)
    assert r["pauses"]["count"] == 0
