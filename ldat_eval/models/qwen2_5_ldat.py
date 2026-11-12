"""Qwen2.5-VL learned DAT; shares the task prompt/scoring wrapper with Qwen3.5."""
from .qwen3_5_ldat import Qwen3_5_LDAT


class Qwen2_5_LDAT(Qwen3_5_LDAT):
    model_family = "qwen2_5_vl"

    def __init__(self, *args, enable_thinking=False, **kwargs):
        from .qwen3_5_ldat import _flag
        if _flag(enable_thinking):
            raise ValueError("Qwen2.5-VL has no Qwen3.5 thinking mode")
        super().__init__(*args, enable_thinking=False, **kwargs)
