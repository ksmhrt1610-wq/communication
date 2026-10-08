from pathlib import Path

from commtrain.history import append_row, read_rows, render_compare, render_history

REPO_HISTORY = Path(__file__).resolve().parent.parent / "history.csv"


def test_baseline_rows_present():
    rows = read_rows(REPO_HISTORY)
    base = {r["task"]: r for r in rows if r["week"] == "0" and r["file"] == "(基準値)"}
    assert base["自己紹介"]["voiced_ratio_pct"] == "77"
    assert base["自己紹介"]["pauses_ge05"] == "11"
    assert base["自己紹介"]["longest_pause_sec"] == "2.43"
    assert base["経済学"]["pauses_ge10"] == "6"
    assert base["経済学"]["semitone_range"] == "10.7"
    assert base["経済学"]["articulation_rate"] == ""


def test_append_and_compare(tmp_path):
    path = tmp_path / "history.csv"
    append_row({"date": "", "week": 0, "task": "自己紹介", "file": "a", "voiced_ratio_pct": 77,
                "pauses_ge10": 2, "longest_pause_sec": 2.43}, path)
    append_row({"date": "2026-01-01", "week": 8, "task": "自己紹介", "file": "b", "voiced_ratio_pct": 82.5,
                "pauses_ge10": 0, "longest_pause_sec": 0.9, "filler_count": 3}, path)
    append_row({"date": "2026-01-01", "week": 8, "task": "経済学", "file": "c"}, path)
    rows = read_rows(path)
    assert len(rows) == 3
    assert path.read_bytes().count(b"\xef\xbb\xbf") == 1  # BOM は先頭だけ

    md = render_compare(rows, 0, 8, "自己紹介")
    assert "| 発声割合(%) | 77 | 82.5 | +5.5 |" in md
    assert "| 1.0秒以上の間(回) | 2 | 0 | -2 |" in md
    assert "| 最長の間(秒) | 2.43 | 0.9 | -1.53 |" in md
    assert "| フィラー数(下限値)(回) | — | 3 | — |" in md

    md_all = render_compare(rows, 0, 8)
    assert "経済学:Week 0 の記録がありません" in md_all

    h = render_history(rows)
    assert h.index("## Week 0") < h.index("## Week 8")
