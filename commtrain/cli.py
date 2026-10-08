"""commtrain コマンド。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from . import TASKS, __version__


def _out_paths(audio: Path) -> dict[str, Path]:
    stem = audio.with_suffix("")
    return {
        "report": Path(f"{stem}_report.md"),
        "metrics": Path(f"{stem}_metrics.json"),
        "packet": Path(f"{stem}_packet.md"),
    }


def run_analysis(audio: Path, task: str, week: int | None, model: str, device: str, compute_type: str) -> dict:
    from .audio import load_audio
    from .pauses import analyze_pauses
    from .pitch import analyze_pitch
    from .text_metrics import analyze_text, pause_contexts
    from .transcribe import transcribe

    print(f"読み込み中:{audio}", file=sys.stderr)
    y = load_audio(audio)
    print("指標A(間と発声)を計算中...", file=sys.stderr)
    a = analyze_pauses(y)
    print("指標B(声の高さ)を計算中...", file=sys.stderr)
    b = analyze_pitch(y)
    tr = transcribe(y, model=model, device=device, compute_type=compute_type)
    print("指標C(文字起こし)を計算中...", file=sys.stderr)
    c = analyze_text(tr["segments"], a["voiced_sec"], a["speech_span_sec"])
    a["long_pause_contexts"] = pause_contexts(tr["segments"], a["pauses"]["long_pauses"])
    return {
        "tool_version": __version__,
        "analyzed_at": datetime.now().isoformat(timespec="seconds"),
        "file": audio.name,
        "task": task,
        "week": week,
        "A": a,
        "B": b,
        "C": c,
        "transcript": tr,
    }


def _write_outputs(audio: Path, m: dict) -> dict[str, Path]:
    from .report import render_report

    paths = _out_paths(audio)
    paths["metrics"].write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["report"].write_text(render_report(m), encoding="utf-8")
    return paths


def cmd_analyze(args) -> int:
    from .history import append_row, row_from_metrics

    audio = Path(args.audio).expanduser()
    m = run_analysis(audio, args.task, args.week, args.model, args.device, args.compute_type)
    paths = _write_outputs(audio, m)
    print(f"レポート:{paths['report']}")
    print(f"指標:{paths['metrics']}")
    if not args.no_history:
        hp = append_row(row_from_metrics(m, datetime.now().strftime("%Y-%m-%d")))
        print(f"履歴に追記:{hp}")
    return 0


def cmd_packet(args) -> int:
    from .report import render_packet

    audio = Path(args.audio).expanduser()
    paths = _out_paths(audio)
    if paths["metrics"].exists() and not args.reanalyze:
        m = json.loads(paths["metrics"].read_text(encoding="utf-8"))
        if args.task:
            m["task"] = args.task
        if args.week is not None:
            m["week"] = args.week
    else:
        print("指標ファイルがないため分析します(履歴には追記しません)。", file=sys.stderr)
        m = run_analysis(audio, args.task or "その他", args.week, args.model, args.device, args.compute_type)
        _write_outputs(audio, m)
    text = render_packet(m)
    if args.output == "-":
        sys.stdout.write(text)
    else:
        out = Path(args.output) if args.output else paths["packet"]
        out.write_text(text, encoding="utf-8")
        print(f"採点パケット:{out}")
    return 0


def cmd_compare(args) -> int:
    from .history import read_rows, render_compare

    text = render_compare(read_rows(), args.week_a, args.week_b, args.task)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"比較表:{args.output}")
    else:
        print(text)
    return 0


def cmd_history(args) -> int:
    from .history import history_path, read_rows, render_history

    rows = read_rows()
    if args.task:
        rows = [r for r in rows if r["task"] == args.task]
    print(f"履歴ファイル:{history_path()}\n")
    print(render_history(rows))
    return 0


def _week(v: str) -> int:
    w = int(v)
    if not 0 <= w <= 8:
        raise argparse.ArgumentTypeError("週は 0〜8 で指定してください")
    return w


def _add_model_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="medium", help="faster-whisper のモデル(既定 medium。large-v3 など)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="既定 auto(GPU があれば使う)")
    p.add_argument("--compute-type", default="auto", help="既定 auto(GPU は float16、CPU は int8)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="commtrain", description="コミュニケーション訓練用の録音分析ツール(ローカル処理)")
    p.add_argument("--version", action="version", version=f"commtrain {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="録音を分析してレポートと指標を出力し、履歴に追記する")
    a.add_argument("audio", help="音声ファイル(wav / m4a / mp3)")
    a.add_argument("--task", required=True, choices=TASKS)
    a.add_argument("--week", required=True, type=_week, help="0〜8")
    a.add_argument("--no-history", action="store_true", help="履歴CSVに追記しない")
    _add_model_opts(a)
    a.set_defaults(func=cmd_analyze)

    k = sub.add_parser("packet", help="Claude に貼る採点パケットを Markdown で出力する")
    k.add_argument("audio", help="音声ファイル(analyze 済みなら <名前>_metrics.json を再利用)")
    k.add_argument("--task", choices=TASKS, help="課題名(未分析のとき、または上書きしたいとき)")
    k.add_argument("--week", type=_week, help="週(未分析のとき、または上書きしたいとき)")
    k.add_argument("-o", "--output", help="出力先(既定 <名前>_packet.md。'-' で標準出力)")
    k.add_argument("--reanalyze", action="store_true", help="既存の指標ファイルを使わず分析し直す")
    _add_model_opts(k)
    k.set_defaults(func=cmd_packet)

    c = sub.add_parser("compare", help="2時点の指標を同じ課題同士で比較する")
    c.add_argument("--week-a", required=True, type=_week)
    c.add_argument("--week-b", required=True, type=_week)
    c.add_argument("--task", choices=TASKS)
    c.add_argument("-o", "--output", help="Markdown の出力先(省略時は標準出力)")
    c.set_defaults(func=cmd_compare)

    h = sub.add_parser("history", help="履歴CSVを週・課題別に一覧表示する")
    h.add_argument("--task", choices=TASKS)
    h.set_defaults(func=cmd_history)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"エラー:{e}", file=sys.stderr)
        return 1
