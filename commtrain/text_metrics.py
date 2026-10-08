"""指標C:文字起こしから計算する指標(モーラ数・話速・フィラー・文)。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

# ---------------------------------------------------------------- モーラ

# 前の字と合わせて1モーラになる小書き文字
SMALL_KANA = set("ャュョァィゥェォヮ")
PUNCT = set("、。,.，．!?！？…・「」『』()()[]【】〈〉《》“”\"' 　\n\t")
SENTENCE_END = set("。?？")


def kata(s: str) -> str:
    """ひらがなをカタカナに変換する(長さは変わらない)。"""
    return "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in s)


def hira(s: str) -> str:
    """カタカナをひらがなに変換する(長さは変わらない)。"""
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


def is_kana(s: str) -> bool:
    return bool(s) and all(("ぁ" <= c <= "ゖ") or ("ァ" <= c <= "ヺ") or c == "ー" for c in s)


def count_morae(reading: str) -> int:
    """読み(カタカナ/ひらがな)のモーラ数を数える。

    - 小書きのャュョァィゥェォ(とヮ)は前の字と合わせて1モーラ(=単独では数えない)
    - ッ・ー・ンは1モーラ
    - かな以外の文字は数えない
    """
    n = 0
    for c in kata(reading):
        if c in SMALL_KANA:
            continue
        if ("ァ" <= c <= "ヺ") or c == "ー":
            n += 1
    return n


@lru_cache(maxsize=1)
def _tagger():
    import fugashi

    return fugashi.Tagger()


def _token_reading(word) -> str | None:
    feat = word.feature
    for attr in ("pron", "kana"):
        r = getattr(feat, attr, None)
        if r and r != "*":
            return r
    if is_kana(word.surface):
        return word.surface
    return None


def tokenize(text: str) -> list[tuple[int, int, str, str | None]]:
    """(開始, 終了, 表層, 読み) のリスト。オフセットは text 内の文字位置。"""
    out = []
    pos = 0
    for w in _tagger()(text):
        pos += len(w.white_space)
        start = text.find(w.surface, pos) if w.surface else pos
        if start < 0:
            start = pos
        end = start + len(w.surface)
        out.append((start, end, w.surface, _token_reading(w)))
        pos = end
    return out


def text_morae(text: str) -> tuple[int, list[str]]:
    """文章のモーラ数と、読みが取れなかった語(数字・英字など)を返す。"""
    total = 0
    unread: list[str] = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        for _s, _e, surface, reading in tokenize(line):
            if reading:
                total += count_morae(reading)
            elif any(c not in PUNCT for c in surface):
                unread.append(surface)
    return total, unread


# ---------------------------------------------------------------- 文字列と時刻


@dataclass
class CharStream:
    """文字起こし全体を1本の文字列にし、各文字に時刻を対応させたもの。

    セグメントの境界には "\\n" を入れる。
    """

    text: str
    starts: list[float]
    ends: list[float]

    def mid(self, i: int) -> float:
        return (self.starts[i] + self.ends[i]) / 2


def build_stream(segments: list[dict]) -> CharStream:
    chars: list[str] = []
    starts: list[float] = []
    ends: list[float] = []
    for si, seg in enumerate(segments):
        seg_chars: list[tuple[str, float, float]] = []
        words = seg.get("words") or []
        if words:
            for w in words:
                for c in w["word"]:
                    seg_chars.append((c, w["start"], w["end"]))
        else:
            text = seg.get("text", "")
            n = max(len(text), 1)
            step = (seg["end"] - seg["start"]) / n
            for k, c in enumerate(text):
                seg_chars.append((c, seg["start"] + k * step, seg["start"] + (k + 1) * step))
        while seg_chars and seg_chars[0][0].isspace():
            seg_chars.pop(0)
        while seg_chars and seg_chars[-1][0].isspace():
            seg_chars.pop()
        if not seg_chars:
            continue
        if chars:
            chars.append("\n")
            starts.append(seg_chars[0][1])
            ends.append(seg_chars[0][1])
        for c, s, e in seg_chars:
            chars.append(c)
            starts.append(s)
            ends.append(e)
    return CharStream("".join(chars), starts, ends)


# ---------------------------------------------------------------- フィラー

# (表示名, 正規表現, 直後に区切りが必要か, 文頭のみか)。長いものから順に照合する。
# 照合はひらがなに揃えた文字列に対して行う。
FILLERS: list[tuple[str, str, bool, bool]] = [
    ("えーと", r"えー+っ?と|ええと", False, False),
    ("えっと", r"えっと", False, False),
    ("えー", r"えー+", False, False),
    ("あのー", r"あのー+", False, False),
    ("あの", r"あの", True, False),
    ("そのー", r"そのー+", False, False),
    ("その", r"その", True, False),
    ("まあ", r"ま[あぁー]+", False, False),
    ("なんか", r"なんか", True, False),
    ("うーん", r"うー+ん", False, False),
    ("うん", r"うん", True, True),
]
FILLER_LABELS = [f[0] for f in FILLERS]
_FILLER_RE = [(label, re.compile(p), rb, ss) for label, p, rb, ss in FILLERS]
BOUNDARY = set("、。,.，．!?！？…・ 　\n")


def _boundaries(text: str) -> set[int]:
    """形態素の境界になる文字位置の集合。"""
    b = {0, len(text)}
    offset = 0
    for line in text.split("\n"):
        b.add(offset)
        b.add(offset + len(line))
        if line:
            for s, e, _surf, _r in tokenize(line):
                b.add(offset + s)
                b.add(offset + e)
        offset += len(line) + 1
    return b


def _filler_at(h: str, i: int):
    for label, rx, rb, ss in _FILLER_RE:
        m = rx.match(h, i)
        if m:
            yield label, m.end(), rb, ss


def _is_sentence_start(text: str, i: int) -> bool:
    j = i - 1
    while j >= 0 and text[j] in " 　":
        j -= 1
    return j < 0 or text[j] in "。?？!！\n"


def find_fillers(text: str) -> list[tuple[str, int, int]]:
    """フィラーを (表示名, 開始, 終了) で返す。

    規則:
    - 照合はひらがなに揃えて行い、長い形を優先する(「えーと」は「えー」より先)。
    - 形態素の境界から始まり、境界で終わるものだけを数える(「マーケット」の「まー」等を除く)。
    - 「あの」「その」「なんか」「うん」は、直後が読点・句点・空白・文末・別のフィラーのときだけ数える
      (「その後」「あの人」等の指示詞を除く)。
    - 「うん」は文頭のみ。
    """
    h = hira(text)
    bounds = _boundaries(text)
    found: list[tuple[str, int, int]] = []
    i = 0
    n = len(h)
    while i < n:
        hit = None
        if i in bounds:
            for label, end, rb, ss in _filler_at(h, i):
                if end not in bounds:
                    continue
                if rb and not (end >= n or h[end] in BOUNDARY or any(True for _ in _filler_at(h, end))):
                    continue
                if ss and not _is_sentence_start(text, i):
                    continue
                hit = (label, i, end)
                break
        if hit:
            found.append(hit)
            i = hit[2]
        else:
            i += 1
    return found


# ---------------------------------------------------------------- 文


def char_len(s: str) -> int:
    """文の文字数(空白と句読点・記号を除く)。"""
    return sum(1 for c in s if c not in PUNCT)


def split_sentences(stream: CharStream) -> list[dict]:
    """句点・疑問符・セグメント境界で文に分ける。"""
    out = []
    buf_start = 0
    text = stream.text
    for i, c in enumerate(text + "\n"):
        if c in SENTENCE_END or c == "\n":
            end = i + 1 if c in SENTENCE_END else i
            raw = text[buf_start:end]
            lead = len(raw) - len(raw.lstrip())
            s = raw.strip()
            if s and char_len(s) > 0:
                k = buf_start + lead
                out.append({"text": s, "chars": char_len(s), "start_sec": round(stream.starts[k], 2)})
            buf_start = i + 1
    return out


# ---------------------------------------------------------------- まとめ


def context(stream: CharStream, i: int, j: int, width: int = 15) -> tuple[str, str]:
    before = stream.text[max(0, i - width):i].replace("\n", " ").strip()
    after = stream.text[j:j + width].replace("\n", " ").strip()
    return before, after


def split_index_at(stream: CharStream, t: float) -> int:
    """時刻 t より前に発話された文字数(=t での分割位置)。"""
    for k in range(len(stream.text)):
        if stream.text[k] != "\n" and stream.mid(k) >= t:
            return k
    return len(stream.text)


def analyze_text(segments: list[dict], voiced_sec: float | None, span_sec: float | None) -> dict:
    stream = build_stream(segments)
    text = stream.text
    morae, unread = text_morae(text)

    fillers = []
    for label, i, j in find_fillers(text):
        before, after = context(stream, i, j)
        fillers.append({
            "label": label, "surface": text[i:j], "start_sec": round(stream.starts[i], 2),
            "before": before, "after": after,
        })
    counts = {label: 0 for label in FILLER_LABELS}
    for f in fillers:
        counts[f["label"]] += 1

    sentences = split_sentences(stream)
    longest = max(sentences, key=lambda s: s["chars"]) if sentences else None

    def rate(den):
        return round(morae / den, 2) if den else None

    return {
        "text": text,
        "morae": morae,
        "unread_tokens": unread,
        "articulation_rate": rate(voiced_sec),
        "overall_rate": rate(span_sec),
        "fillers": {"count": len(fillers), "by_label": counts, "items": fillers},
        "sentences": {
            "count": len(sentences),
            "mean_chars": round(sum(s["chars"] for s in sentences) / len(sentences), 1) if sentences else None,
            "longest": longest,
            "first": sentences[0] if sentences else None,
            "items": sentences,
        },
    }


def pause_contexts(segments: list[dict], pauses: list[dict]) -> list[dict]:
    stream = build_stream(segments)
    out = []
    for p in pauses:
        k = split_index_at(stream, (p["start_sec"] + p["end_sec"]) / 2)
        before, after = context(stream, k, k)
        out.append({**p, "before": before, "after": after})
    return out
