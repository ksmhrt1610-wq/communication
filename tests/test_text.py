import pytest

pytest.importorskip("fugashi")

from commtrain.report import render_packet, render_report, transcript_md  # noqa: E402
from commtrain.text_metrics import analyze_text, find_fillers, pause_contexts  # noqa: E402


def labels(text):
    return [f[0] for f in find_fillers(text)]


def test_fillers_basic():
    assert labels("えーと、あのー、まぁ、うーん、そのー、えっと、えー") == [
        "えーと", "あのー", "まあ", "うーん", "そのー", "えっと", "えー",
    ]


def test_fillers_exclude_demonstratives():
    # 「その後」「あの人」は指示詞なので数えない。「あの、」「その、」は数える
    assert labels("あの人はその後来た。") == []
    assert labels("あの、その、来た。") == ["あの", "その"]
    # 語の途中(マーケット)は数えない
    assert labels("マーケットの話") == []


def test_un_only_at_sentence_start():
    assert labels("うん、そうです。それはうん、違う。うん。") == ["うん", "うん"]
    assert labels("うんどうは大事") == []


def seg(start, words):
    t = start
    out = []
    for w in words:
        out.append({"start": t, "end": t + 0.5, "word": w})
        t += 0.5
    return {"start": start, "end": t, "text": "".join(words), "words": out}


SEGMENTS = [
    seg(0.5, ["えーと", "、", "私", "は", "田中", "です", "。"]),
    seg(5.0, ["経済学", "を", "勉強", "して", "います", "。", "あの", "、", "趣味", "は", "?"]),
    seg(12.0, ["読書", "です"]),
]


def test_analyze_text():
    c = analyze_text(SEGMENTS, voiced_sec=8.0, span_sec=12.0)
    assert c["fillers"]["count"] == 2
    assert [f["start_sec"] for f in c["fillers"]["items"]] == [0.5, 8.0]
    s = c["sentences"]
    assert [x["text"] for x in s["items"]] == [
        "えーと、私は田中です。", "経済学を勉強しています。", "あの、趣味は?", "読書です",
    ]
    assert s["first"]["text"] == "えーと、私は田中です。"
    assert s["longest"]["text"] == "経済学を勉強しています。"
    assert s["longest"]["chars"] == 11
    assert c["articulation_rate"] == pytest.approx(c["morae"] / 8.0, abs=0.01)
    assert c["overall_rate"] == pytest.approx(c["morae"] / 12.0, abs=0.01)


def test_pause_context():
    ctx = pause_contexts(SEGMENTS, [{"start_sec": 4.0, "end_sec": 5.0, "dur_sec": 1.0}])
    assert ctx[0]["before"].endswith("田中です。")
    assert ctx[0]["after"].startswith("経済学を")


def test_transcript_has_10s_breaks():
    md = transcript_md(SEGMENTS)
    assert "**── 0:00.0〜0:10.0 ──**" in md
    assert "**── 0:10.0〜0:20.0 ──**" in md
    # 5.0秒から始まるセグメントは 10秒の境界(「?」が 10.0 秒)で2行に分かれる
    assert "[0:05.0] 経済学を勉強しています。あの、趣味は  " in md
    assert "[0:10.0] ?" in md
    assert md.index("[0:05.0]") < md.index("0:10.0〜0:20.0") < md.index("[0:10.0]")


def test_report_and_packet_render():
    from commtrain.pauses import analyze_pauses
    from tests.test_pauses import PATTERN, make_signal

    a = analyze_pauses(make_signal(PATTERN))
    a["long_pause_contexts"] = pause_contexts(SEGMENTS, a["pauses"]["long_pauses"])
    m = {
        "file": "x.m4a", "task": "自己紹介", "week": 0, "analyzed_at": "2026-01-01T00:00:00",
        "A": a,
        "B": {"voiced_frames": 10, "f0_median_hz": 120.0, "semitone_range_5_95": 9.3, "semitone_std": 2.8},
        "C": analyze_text(SEGMENTS, a["voiced_sec"], a["speech_span_sec"]),
        "transcript": {"model": "medium", "device": "cpu", "compute_type": "int8", "segments": SEGMENTS},
    }
    r = render_report(m)
    assert "フィラー数は下限値" in r
    assert "秒の間:直前『" in r
    p = render_packet(m)
    assert p.index("## 1. 録音情報") < p.index("## 2.") < p.index("## 3.") < p.index("## 4.")
    assert "| 適応と主張 | 相手を問わず同じ言い方。意見を濁す | 相手に応じて言い換え、要約してから異論を言える |" in p
