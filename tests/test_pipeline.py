from types import SimpleNamespace

from app.pipeline import ProcessingOptions, _should_reuse_previous_result


def test_should_reuse_previous_result_when_frame_changes_are_small():
    sample = SimpleNamespace(scene_delta=0.03, text_delta=0.01)
    options = ProcessingOptions()

    assert _should_reuse_previous_result(sample, ("同じテキスト", 0.9), options) is True


def test_should_not_reuse_previous_result_when_text_change_is_large():
    sample = SimpleNamespace(scene_delta=0.03, text_delta=0.05)
    options = ProcessingOptions()

    assert _should_reuse_previous_result(sample, ("別テキスト", 0.9), options) is False
