"""レポート(report.md)と採点パケットの生成。数値と事実だけを書く。"""

from __future__ import annotations

from .text_metrics import FILLER_LABELS

FILLER_NOTE = "フィラー数は下限値です(Whisper はフィラーを省略して文字起こしすることが多いため、実際の数はこれ以上です)。"

SCORING_REQUEST = """\
以下はコミュニケーション訓練の録音の文字起こしと測定値です。次の6要素を1〜5点で採点してください。
各要素について、点数・根拠となる発言の引用(タイムスタンプつき)・次に直す点を1つ書いてください。お世辞は不要です。
文字起こしの誤変換と思われる箇所は減点の根拠にせず、「誤変換の可能性」と明記してください。
録音1本からは判断できない要素は「判断不可」とし、無理に点をつけないでください。

| 要素 | 1点の状態 | 5点の状態 |
| 落ち着き | 早口でフィラーが多く、声が揺れる | 一呼吸置いて話し始め、速度が一定 |
| 傾聴・注意 | すぐ自分の話に移る | 内容と感情を言い換え、追加質問で深める |
| 会話の調整 | 割り込むか黙るかの両極端 | 話題をつなぎ、量を調整し、自分から閉じられる |
| 論理構成 | 思いつく順に話し、結論が見えない | 冒頭で結論、理由2〜3個に根拠を添える |
| 表現力 | 平板で抽象語が多い | 強弱と間を使い、具体的な場面で語る |
| 適応と主張 | 相手を問わず同じ言い方。意見を濁す | 相手に応じて言い換え、要約してから異論を言える |"""


def fmt(v, unit: str = "", nd: int | None = None) -> str:
    if v is None or v == "":
        return "—"
    if isinstance(v, float) and nd is not None:
        v = f"{v:.{nd}f}"
    return f"{v}{unit}"


def ts(t: float) -> str:
    m, s = divmod(float(t), 60)
    return f"{int(m)}:{s:04.1f}"


def week_label(week) -> str:
    return "未指定" if week is None else f"Week {week}"


# ---------------------------------------------------------------- 表


def table(rows: list[tuple[str, str]], head=("指標", "値")) -> str:
    out = [f"| {head[0]} | {head[1]} |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(out)


def rows_a(a: dict) -> list[tuple[str, str]]:
    p = a["pauses"]
    c = a["chunks"]
    return [
        ("全長", fmt(a["duration_sec"], " 秒")),
        ("最初の発声までの秒数", fmt(a["first_onset_sec"], " 秒")),
        ("発話区間(最初の発声〜最後の発声)", fmt(a["speech_span_sec"], " 秒")),
        ("発声時間", fmt(a["voiced_sec"], " 秒")),
        ("発声割合(発声時間 ÷ 発話区間)", fmt(a["voiced_ratio_pct"], " %")),
        ("発声割合(発声時間 ÷ 全長、参考)", fmt(a["voiced_ratio_of_total_pct"], " %")),
        ("発声のかたまりの数", fmt(c["count"], " 個")),
        ("発声のかたまりの平均", fmt(c["mean_sec"], " 秒")),
        ("発声のかたまりの最長", fmt(c["max_sec"], " 秒")),
        ("間の回数(0.15秒以上)", fmt(p["count"], " 回")),
        ("間の中央値", fmt(p["median_sec"], " 秒")),
        ("0.5秒以上の間", fmt(p["ge_0_5"], " 回")),
        ("1.0秒以上の間", fmt(p["ge_1_0"], " 回")),
        ("最長の間", fmt(p["max_sec"], " 秒")),
        ("1分あたりの間の回数", fmt(p["per_min"], " 回/分")),
        ("1分あたりの0.5秒以上の間", fmt(p["ge_0_5_per_min"], " 回/分")),
        ("1分あたりの1.0秒以上の間", fmt(p["ge_1_0_per_min"], " 回/分")),
    ]


def rows_b(b: dict) -> list[tuple[str, str]]:
    return [
        ("F0 中央値(有声フレーム)", fmt(b["f0_median_hz"], " Hz")),
        ("半音幅(5〜95パーセンタイル)", fmt(b["semitone_range_5_95"], " 半音")),
        ("半音の標準偏差", fmt(b["semitone_std"], " 半音")),
        ("有声フレーム数", fmt(b["voiced_frames"])),
    ]


def rows_c(c: dict) -> list[tuple[str, str]]:
    s = c["sentences"]
    longest = s["longest"]
    return [
        ("モーラ数", fmt(c["morae"])),
        ("調音速度(モーラ数 ÷ 発声時間)", fmt(c["articulation_rate"], " モーラ/秒")),
        ("全体速度(モーラ数 ÷ 発話区間)", fmt(c["overall_rate"], " モーラ/秒")),
        ("フィラー数(下限値)", fmt(c["fillers"]["count"], " 回")),
        ("文の数", fmt(s["count"])),
        ("平均文字数", fmt(s["mean_chars"], " 字")),
        ("最長の文の文字数", fmt(longest["chars"] if longest else None, " 字")),
    ]


def halves_table(a: dict) -> str:
    h = a.get("halves")
    if not h:
        return "(発声が検出されませんでした)"
    return "\n".join([
        f"発話区間の中点:{h['midpoint_sec']} 秒(間の中央の時刻で前半・後半に振り分け)",
        "",
        "| | 間の回数 | 0.5秒以上の間 |",
        "|---|---|---|",
        f"| 前半 | {h['first']['count']} 回 | {h['first']['ge_0_5']} 回 |",
        f"| 後半 | {h['second']['count']} 回 | {h['second']['ge_0_5']} 回 |",
    ])


def windows_table(a: dict) -> str:
    out = ["| 区間 | 発声率 |", "|---|---|"]
    for w in a["voiced_pct_per_10s"]:
        out.append(f"| {w['start_sec']:g}〜{w['end_sec']:g} 秒 | {w['voiced_pct']} % |")
    return "\n".join(out)


def filler_breakdown(c: dict) -> str:
    by = c["fillers"]["by_label"]
    used = [(k, by.get(k, 0)) for k in FILLER_LABELS if by.get(k, 0)]
    if not used:
        return "(検出なし)"
    return "、".join(f"{k} {v}回" for k, v in used)


# ---------------------------------------------------------------- 文字起こし


def transcript_md(segments: list[dict]) -> str:
    """タイムスタンプ付き文字起こし。10秒ごとに区切りを入れる。"""
    lines: list[str] = []
    cur_bin = -1
    buf: list[str] = []
    buf_start: float | None = None

    def flush():
        nonlocal buf, buf_start
        text = "".join(buf).strip()
        if text:
            lines.append(f"[{ts(buf_start)}] {text}  ")
        buf, buf_start = [], None

    for seg in segments:
        pieces = seg.get("words") or [{"start": seg["start"], "end": seg["end"], "word": seg["text"]}]
        for w in pieces:
            b = max(int(w["start"] // 10), cur_bin)
            if b != cur_bin:
                flush()
                if lines:
                    lines.append("")
                lines.append(f"**── {ts(b * 10)}〜{ts(b * 10 + 10)} ──**  ")
                cur_bin = b
            if buf_start is None:
                buf_start = w["start"]
            buf.append(w["word"])
        flush()
    return "\n".join(lines) if lines else "(文字起こし結果なし)"


def long_pause_lines(a: dict) -> list[str]:
    items = a.get("long_pause_contexts") or a["pauses"]["long_pauses"]
    out = []
    for p in items:
        before = p.get("before", "")
        after = p.get("after", "")
        out.append(f"- {p['start_sec']:.1f}秒 {p['dur_sec']:.2f}秒の間:直前『{before}』/直後『{after}』")
    return out or ["- (1.0秒以上の間はありません)"]


def filler_lines(c: dict) -> list[str]:
    out = []
    for f in c["fillers"]["items"]:
        out.append(f"- {f['start_sec']:.1f}秒「{f['surface']}」:直前『{f['before']}』/直後『{f['after']}』")
    return out or ["- (検出されたフィラーはありません)"]


# ---------------------------------------------------------------- レポート


def render_report(m: dict) -> str:
    a, b, c = m["A"], m["B"], m["C"]
    s = c["sentences"]
    first = s["first"]["text"] if s["first"] else "—"
    longest = s["longest"]
    tr = m["transcript"]
    parts = [
        f"# 録音分析レポート:{m['file']}",
        "",
        f"- 課題:{m['task']}",
        f"- 週:{week_label(m['week'])}",
        f"- 分析日時:{m['analyzed_at']}",
        f"- 録音の長さ:{a['duration_sec']} 秒",
        f"- 文字起こしモデル:faster-whisper {tr['model']}({tr['device']}/{tr['compute_type']})",
        "",
        "このレポートは測定値と事実のみを記載しています。",
        "",
        "## A. 間と発声",
        "",
        table(rows_a(a)),
        "",
        f"閾値:{a['levels_db']['threshold_db']} dB(雑音 {a['levels_db']['noise_db']} dB、95パーセンタイル {a['levels_db']['p95_db']} dB)",
        "",
        "### 1.0秒以上の間の位置と長さ",
        "",
        *[f"- {p['start_sec']:.1f}秒から {p['dur_sec']:.2f}秒" for p in a["pauses"]["long_pauses"]],
        *([] if a["pauses"]["long_pauses"] else ["- (なし)"]),
        "",
        "### 前半/後半",
        "",
        halves_table(a),
        "",
        "### 10秒ごとの発声率",
        "",
        windows_table(a),
        "",
        "## B. 声の高さ",
        "",
        table(rows_b(b)),
        "",
        "## C. 文字起こしから計算した指標",
        "",
        table(rows_c(c)),
        "",
        f"> {FILLER_NOTE}",
        "",
        f"- フィラーの内訳:{filler_breakdown(c)}",
        f"- 最長の文({longest['chars']}字、{longest['start_sec']:.1f}秒〜):「{longest['text']}」" if longest else "- 最長の文:—",
        *([f"- 読みを取得できずモーラ数に含めていない語:{'、'.join(c['unread_tokens'])}"] if c["unread_tokens"] else []),
        "",
        "### 冒頭の一文",
        "",
        f"「{first}」",
        "",
        "## 1.0秒以上の間と前後の発言",
        "",
        *long_pause_lines(a),
        "",
        "## フィラーの位置と前後の発言",
        "",
        f"> {FILLER_NOTE}",
        "",
        *filler_lines(c),
        "",
        "## タイムスタンプ付き文字起こし",
        "",
        transcript_md(tr["segments"]),
        "",
    ]
    return "\n".join(parts)


def render_packet(m: dict) -> str:
    a, b, c = m["A"], m["B"], m["C"]
    p = a["pauses"]
    s = c["sentences"]
    first = s["first"]["text"] if s["first"] else "—"
    long_p = "、".join(f"{x['start_sec']:.1f}秒({x['dur_sec']:.2f}秒)" for x in p["long_pauses"]) or "なし"
    parts = [
        "# コミュニケーション訓練 採点パケット",
        "",
        "## 1. 録音情報",
        "",
        f"- 課題:{m['task']}",
        f"- 週:{week_label(m['week'])}",
        f"- 録音の長さ:{a['duration_sec']} 秒",
        "",
        "## 2. 測定値の要約",
        "",
        "### A. 間と発声",
        "",
        f"- 最初の発声まで {fmt(a['first_onset_sec'], ' 秒')}、発話区間 {fmt(a['speech_span_sec'], ' 秒')}、"
        f"発声割合 {fmt(a['voiced_ratio_pct'], ' %')}",
        f"- 発声のかたまり {fmt(a['chunks']['count'], ' 個')}(平均 {fmt(a['chunks']['mean_sec'], ' 秒')}、"
        f"最長 {fmt(a['chunks']['max_sec'], ' 秒')})",
        f"- 間(0.15秒以上){fmt(p['count'], ' 回')}(中央値 {fmt(p['median_sec'], ' 秒')})、"
        f"0.5秒以上 {fmt(p['ge_0_5'], ' 回')}、1.0秒以上 {fmt(p['ge_1_0'], ' 回')}、最長 {fmt(p['max_sec'], ' 秒')}",
        f"- 1分あたり:間 {fmt(p['per_min'], ' 回')}、0.5秒以上 {fmt(p['ge_0_5_per_min'], ' 回')}",
        f"- 1.0秒以上の間の位置:{long_p}",
    ]
    h = a.get("halves")
    if h:
        parts.append(
            f"- 前半/後半の間:前半 {h['first']['count']} 回(0.5秒以上 {h['first']['ge_0_5']} 回)、"
            f"後半 {h['second']['count']} 回(0.5秒以上 {h['second']['ge_0_5']} 回)"
        )
    parts += [
        "",
        "### B. 声の高さ",
        "",
        f"- F0 中央値 {fmt(b['f0_median_hz'], ' Hz')}、半音幅(5〜95%){fmt(b['semitone_range_5_95'], ' 半音')}、"
        f"半音の標準偏差 {fmt(b['semitone_std'], ' 半音')}",
        "",
        "### C. 文字起こしから",
        "",
        f"- 調音速度 {fmt(c['articulation_rate'], ' モーラ/秒')}、全体速度 {fmt(c['overall_rate'], ' モーラ/秒')}",
        f"- フィラー {c['fillers']['count']} 回(下限値。Whisper はフィラーを省略しがち)。内訳:{filler_breakdown(c)}",
        f"- 文の数 {s['count']}、平均文字数 {fmt(s['mean_chars'], ' 字')}、"
        f"最長の文 {fmt(s['longest']['chars'] if s['longest'] else None, ' 字')}",
        f"- 冒頭の一文:「{first}」",
        "",
        "## 3. タイムスタンプ付き文字起こし",
        "",
        transcript_md(m["transcript"]["segments"]),
        "",
        "## 4. 採点依頼",
        "",
        "---",
        "",
        SCORING_REQUEST,
        "",
        "---",
        "",
    ]
    return "\n".join(parts)
