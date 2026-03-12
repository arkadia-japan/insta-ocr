from app.text_corrections import apply_text_corrections


def test_apply_text_corrections_replaces_known_ocr_errors():
    corrections = {
        "特微": "特徴",
        "短文いNE": "短文LINE",
    }

    text = "女に追われる男の特微\n短文いNEの男"

    assert apply_text_corrections(text, corrections) == "女に追われる男の特徴\n短文LINEの男"


def test_apply_text_corrections_does_not_duplicate_existing_target_text():
    corrections = {
        "※気に入ったらフォローして": "※気に入ったらフォローしてね!",
    }

    text = "※気に入ったらフォローしてね!"

    assert apply_text_corrections(text, corrections) == "※気に入ったらフォローしてね!"


def test_apply_text_corrections_handles_recent_real_world_misreads():
    corrections = {
        "月然体": "自然体",
        "自分の時間ある男": "自分の時間がある男",
        "フォロ一": "フォロー",
    }

    text = "月然体でいれる男\n自分の時間ある男\n※気にスっナらフォロ一してね |"

    assert apply_text_corrections(text, corrections) == (
        "自然体でいれる男\n"
        "自分の時間がある男\n"
        "※気にスっナらフォローしてね |"
    )
