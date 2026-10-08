"""履歴 CSV の追記・一覧・2時点比較。"""

from __future__ import annotations

import csv
import os
from pathlib import Path

from . import TASKS

COLUMNS = [
    "date", "week", "task", "file",
    "duration_sec", "span_sec",
    "voiced_ratio_pct", "pauses_ge05", "pauses_ge05_per_min", "pauses_ge10", "longest_pause_sec",
    "articulation_rate", "overall_rate", "filler_count", "mean_sentence_chars", "semitone_range",
    "note",
]

# 比較・一覧に使う数値列:(列名, 表示名, 単位)
METRICS = [
    ("duration_sec", "全長", "秒"),
    ("voiced_ratio_pct", "発声割合", "%"),
    ("pauses_ge05", "0.5秒以上の間", "回"),
    ("pauses_ge05_per_min", "1分あたりの0.5秒以上の間", "回/分"),
    ("pauses_ge10", "1.0秒以上の間", "回"),
    ("longest_pause_sec", "最長の間", "秒"),
    ("articulation_rate", "調音速度", "モーラ/秒"),
    ("overall_rate", "全体速度", "モーラ/秒"),
    ("filler_count", "フィラー数(下限値)", "回"),
    ("mean_sentence_chars", "平均文長", "字"),
    ("semitone_range", "半音幅(5〜95%)", "半音"),
]


def history_path() -> Path:
    env = os.environ.get("COMMTRAIN_HISTORY")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "history.csv"


def row_from_metrics(m: dict, date: str) -> dict:
    a, b, c = m["A"], m["B"], m["C"]
    return {
        "date": date,
        "week": m["week"],
        "task": m["task"],
        "file": m["file"],
        "duration_sec": a["duration_sec"],
        "span_sec": a["speech_span_sec"],
        "voiced_ratio_pct": a["voiced_ratio_pct"],
        "pauses_ge05": a["pauses"]["ge_0_5"],
        "pauses_ge05_per_min": a["pauses"]["ge_0_5_per_min"],
        "pauses_ge10": a["pauses"]["ge_1_0"],
        "longest_pause_sec": a["pauses"]["max_sec"],
        "articulation_rate": c["articulation_rate"],
        "overall_rate": c["overall_rate"],
        "filler_count": c["fillers"]["count"],
        "mean_sentence_chars": c["sentences"]["mean_chars"],
        "semitone_range": b["semitone_range_5_95"],
        "note": "",
    }


def append_row(row: dict, path: Path | None = None) -> Path:
    path = path or history_path()
    new = not path.exists() or path.stat().st_size == 0
    # 新規作成時は Excel で文字化けしないよう BOM を付ける。追記時は付けない。
    with open(path, "a", newline="", encoding="utf-8-sig" if new else "utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in COLUMNS})
    return path


def read_rows(path: Path | None = None) -> list[dict]:
    path = path or history_path()
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def num(v) -> float | None:
    if v is None or str(v).strip() == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def show(v) -> str:
    x = num(v)
    if x is None:
        return "—"
    return f"{x:g}" if x == int(x) else f"{x:.2f}".rstrip("0").rstrip(".")


def _task_order(t: str) -> int:
    return TASKS.index(t) if t in TASKS else len(TASKS)


def _week(r: dict) -> int:
    w = num(r.get("week"))
    return int(w) if w is not None else -1


def render_history(rows: list[dict]) -> str:
    if not rows:
        return "履歴はまだありません。"
    rows = sorted(rows, key=lambda r: (_week(r), _task_order(r["task"]), r.get("date", "")))
    short = [
        ("voiced_ratio_pct", "発声割合%"), ("pauses_ge05_per_min", "0.5秒+/分"), ("pauses_ge10", "1.0秒+"),
        ("longest_pause_sec", "最長間"), ("articulation_rate", "調音速度"), ("overall_rate", "全体速度"),
        ("filler_count", "フィラー"), ("mean_sentence_chars", "平均文長"), ("semitone_range", "半音幅"),
    ]
    out: list[str] = []
    cur = None
    for r in rows:
        wk = _week(r)
        if wk != cur:
            if cur is not None:
                out.append("")
            out += [
                f"## Week {wk}",
                "",
                "| 課題 | 日付 | ファイル | 全長 | " + " | ".join(lbl for _, lbl in short) + " |",
                "|---" * (4 + len(short)) + "|",
            ]
            cur = wk
        cells = [r["task"], r.get("date") or "—", r.get("file") or "—", show(r.get("duration_sec"))]
        cells += [show(r.get(k)) for k, _ in short]
        out.append("| " + " | ".join(cells) + " |")
    if any(r.get("note") for r in rows):
        out += ["", "注記:"]
        for r in rows:
            if r.get("note"):
                out.append(f"- Week {_week(r)} {r['task']}:{r['note']}")
    return "\n".join(out)


def latest(rows: list[dict], week: int, task: str) -> dict | None:
    """同じ週・課題の行が複数あるときは最後に追記された行を使う。"""
    match = [r for r in rows if _week(r) == week and r["task"] == task]
    return match[-1] if match else None


def diff_str(a, b) -> str:
    x, y = num(a), num(b)
    if x is None or y is None:
        return "—"
    d = round(y - x, 2)
    if d == 0:
        return "±0"
    return f"{d:+g}"


def render_compare(rows: list[dict], week_a: int, week_b: int, task: str | None = None) -> str:
    tasks = [task] if task else sorted({r["task"] for r in rows}, key=_task_order)
    out = [f"# Week {week_a} と Week {week_b} の比較", ""]
    out.append("同じ課題同士を比較しています。増減は Week %d − Week %d です。" % (week_b, week_a))
    out.append("同じ週・課題に複数の記録がある場合は、最後に追記された記録を使います。")
    any_table = False
    missing = []
    for t in tasks:
        ra, rb = latest(rows, week_a, t), latest(rows, week_b, t)
        if not ra or not rb:
            missing.append((t, ra is not None, rb is not None))
            continue
        any_table = True
        out += [
            "",
            f"## {t}",
            "",
            f"| 指標 | Week {week_a} | Week {week_b} | 増減 |",
            "|---|---|---|---|",
            f"| ファイル | {ra.get('file') or '—'} | {rb.get('file') or '—'} | |",
        ]
        for key, label, unit in METRICS:
            out.append(f"| {label}({unit}) | {show(ra.get(key))} | {show(rb.get(key))} | {diff_str(ra.get(key), rb.get(key))} |")
        notes = [f"Week {w}:{r['note']}" for w, r in ((week_a, ra), (week_b, rb)) if r.get("note")]
        if notes:
            out += ["", *[f"- 注記 {n}" for n in notes]]
    if missing:
        out += ["", "## 比較できなかった課題", ""]
        for t, has_a, has_b in missing:
            lack = [f"Week {w}" for w, ok in ((week_a, has_a), (week_b, has_b)) if not ok]
            out.append(f"- {t}:{'・'.join(lack)} の記録がありません")
    if not any_table and not missing:
        out += ["", "記録がありません。"]
    return "\n".join(out)
