"""Validated single-sequence decoding for the DAT autoregressive loop."""
from dataclasses import asdict, dataclass, fields
import math

import torch
from transformers import (
    MinPLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


@dataclass(frozen=True)
class DecodeOptions:
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0

    @classmethod
    def from_kwargs(cls, kwargs=None):
        values = dict(kwargs or {})
        # Length is controlled by the runner; stop strings by the lmms adapter.
        values.pop("max_new_tokens", None)
        values.pop("until", None)
        for key in ("num_beams", "num_return_sequences"):
            if values.pop(key, 1) != 1:
                raise ValueError(f"DAT supports only {key}=1")
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unsupported DAT generation options: {sorted(unknown)}")
        sample = values.get("do_sample", False)
        if isinstance(sample, str) and sample.lower() in {"true", "false"}:
            sample = sample.lower() == "true"
        if not isinstance(sample, bool):
            raise ValueError("do_sample must be a boolean")
        values["do_sample"] = sample
        for key in ("temperature", "top_p", "min_p", "presence_penalty", "repetition_penalty"):
            if key in values:
                values[key] = float(values[key])
                if not math.isfinite(values[key]):
                    raise ValueError(f"{key} must be finite")
        k = values.get("top_k", 0)
        if isinstance(k, bool) or int(k) != k or k < 0:
            raise ValueError("top_k must be a non-negative integer")
        values["top_k"] = int(k)
        result = cls(**values)
        if result.temperature < 0 or (result.do_sample and result.temperature == 0):
            raise ValueError("temperature must be positive for sampling (zero is allowed for greedy)")
        if not 0 < result.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if not 0 <= result.min_p <= 1:
            raise ValueError("min_p must be in [0, 1]")
        if result.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        return result

    def to_dict(self):
        return asdict(self)

    def process(self, logits, generated, prompt_ids=()):
        scores = logits.float().clone()[None]
        if self.repetition_penalty != 1:
            history = torch.as_tensor(list(prompt_ids) + list(generated), device=logits.device,
                                      dtype=torch.long)[None]
            scores = RepetitionPenaltyLogitsProcessor(self.repetition_penalty)(history, scores)
        if self.presence_penalty and generated:
            seen = torch.tensor(generated, dtype=torch.long, device=logits.device).unique()
            scores[0, seen] -= self.presence_penalty
        if self.do_sample:
            if self.temperature != 1:
                scores = TemperatureLogitsWarper(self.temperature)(None, scores)
            if self.top_k:
                scores = TopKLogitsWarper(self.top_k)(None, scores)
            if self.top_p < 1:
                scores = TopPLogitsWarper(self.top_p)(None, scores)
            if self.min_p:
                scores = MinPLogitsWarper(self.min_p)(None, scores)
        return scores[0]

    def choose(self, logits, generated, prompt_ids=()):
        if not self.do_sample and self.presence_penalty == 0 and self.repetition_penalty == 1:
            return int(logits.argmax())
        scores = self.process(logits, generated, prompt_ids)
        if not self.do_sample:
            return int(scores.argmax())
        return int(torch.multinomial(torch.softmax(scores, dim=-1), 1))
