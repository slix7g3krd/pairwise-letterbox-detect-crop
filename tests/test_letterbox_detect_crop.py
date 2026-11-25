import pytest
import media.letterbox_detect_crop as mod

CASES = [
    ("brightness_hysteresis", (1,)),
    ("detect_full_frame", (1,)),
    ("track_per_frame", (1,)),
    ("even_dimension", (1,)),
    ("crop_in_display_space", (1,)),
    ("review_result", (1,)),
    ("load", (1,)),
    ("detect", (1,)),
    ("crop", (1,)),
    ("export", (1,)),
    ("validate_letterbox", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
