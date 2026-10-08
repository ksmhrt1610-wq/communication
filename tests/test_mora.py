import pytest

from commtrain.text_metrics import count_morae


@pytest.mark.parametrize(
    "reading, expected",
    [
        ("きょうは", 3),
        ("がっこう", 4),
        ("コーヒー", 4),
        ("キョウハ", 3),
        ("しんぶん", 4),
        ("ちゃっきょ", 3),   # ちゃ・っ・きょ
        ("ファイル", 3),
        ("ジェットコースター", 8),  # ジェ・ッ・ト・コ・ー・ス・タ・ー
        ("ん", 1),
        ("ー", 1),
        ("", 0),
        ("、。", 0),
    ],
)
def test_count_morae(reading, expected):
    assert count_morae(reading) == expected


def test_text_morae_uses_reading():
    pytest.importorskip("fugashi")
    from commtrain.text_metrics import text_morae

    # 今日(キョー2)は(ワ1)学校(ガッコー4)で(デ1)コーヒー(4)を(オ1)飲ん(ノン2)だ(ダ1)= 16
    n, unread = text_morae("今日は学校でコーヒーを飲んだ。")
    assert n == 16
    assert unread == []
    n, unread = text_morae("2020年")
    assert unread == ["2020"]
