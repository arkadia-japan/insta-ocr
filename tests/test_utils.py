from app.utils import normalize_text


def test_normalize_text_removes_spaces_between_japanese_characters():
    assert normalize_text("女 に 追われる 男") == "女に追われる男"


def test_normalize_text_keeps_ascii_word_spacing():
    assert normalize_text("follow me now") == "follow me now"
