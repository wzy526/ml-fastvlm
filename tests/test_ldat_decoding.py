"""Exercise real DAT decoding with controlled logits, without GPU checkpoints."""
from types import SimpleNamespace

import pytest
import torch

from ldat_eval.infer import LDatRunner


def decoder(rows, *, thinking=False):
    outputs = [torch.tensor(row, dtype=torch.float32)[None, None] for row in rows]
    state = {"next": 1}

    def forward(**kwargs):
        idx = min(state["next"], len(outputs) - 1)
        state["next"] += 1
        return SimpleNamespace(logits=outputs[idx].clone(), past_key_values=None)

    runner = LDatRunner.__new__(LDatRunner)
    runner.dev = torch.device("cpu")
    runner.eos = {4}
    runner.think_end = 3 if thinking else None
    runner.enable_thinking = thinking
    runner.n_think_closed = runner.n_think_capped = 0
    runner.backbone = SimpleNamespace(forward=forward)
    runner.tok = SimpleNamespace(decode=lambda ids, **kw: " ".join(map(str, ids)))
    initial = SimpleNamespace(logits=outputs[0].clone(), past_key_values=None)
    return runner, initial


def decode(rows, limit=3, **kwargs):
    runner, initial = decoder(rows)
    answer = runner._decode(initial, torch.zeros(3, 1, 1), 1, limit,
                            generation_kwargs=kwargs)
    return answer, runner.last_generation


def test_greedy_preserves_tokens_eos_and_rng():
    torch.manual_seed(42)
    before = torch.get_rng_state().clone()
    answer, meta = decode([[9, 0, 0, 0, -9], [0, 9, 0, 0, -9], [0, 0, 0, 0, 9]],
                          do_sample="False", temperature=0, top_p=1)
    assert answer == "0 1"
    assert meta["finish_reason"] == "eos"
    assert meta["generated_token_ids"] == [0, 1, 4]
    assert meta["output_tokens"] == 3
    assert torch.equal(before, torch.get_rng_state())


def test_presence_penalty_reaches_autoregressive_loop():
    rows = [[2, 1.9, -10, -10, -10]]
    assert decode(rows, limit=3)[0] == "0 0 0"
    answer, meta = decode(rows, limit=3, presence_penalty=1.5)
    assert answer == "0 1 0"  # Presence penalizes once, not once per occurrence.
    assert meta["finish_reason"] == "length"
    assert meta["output_tokens"] == 3


def test_sampling_is_seeded_and_not_argmax():
    rows = [[0, 0, -100, -100, -100]]
    torch.manual_seed(7)
    a, meta = decode(rows, limit=32, do_sample=True, temperature=0.7, top_k=2, top_p=1)
    torch.manual_seed(7)
    b, _ = decode(rows, limit=32, do_sample=True, temperature=0.7, top_k=2, top_p=1)
    assert a == b
    assert set(a.split()) == {"0", "1"}
    assert meta["output_tokens"] == 32


def test_limit_does_not_consume_an_extra_sampling_draw():
    torch.manual_seed(13)
    decode([[0, 0, -100, -100, -100]], limit=1, do_sample=True)
    actual = torch.get_rng_state()
    torch.manual_seed(13)
    torch.multinomial(torch.tensor([0.5, 0.5, 0., 0., 0.]), 1)
    assert torch.equal(actual, torch.get_rng_state())


def test_temperature_top_k_and_top_p_filter_real_logits():
    from ldat_eval.decoding import DecodeOptions
    logits = torch.tensor([4., 2., 0., -10.])
    original = logits.clone()
    opts = DecodeOptions.from_kwargs(dict(do_sample=True, temperature=2, top_k=2, top_p=1))
    scores = opts.process(logits, [], [])
    torch.testing.assert_close(scores[:2], torch.tensor([2., 1.]))
    assert torch.isneginf(scores[2:]).all()
    opts = DecodeOptions.from_kwargs(dict(do_sample=True, top_p=0.7))
    scores = opts.process(torch.log(torch.tensor([0.6, 0.25, 0.15])), [], [])
    assert torch.isfinite(scores[:2]).all() and torch.isneginf(scores[2])
    torch.testing.assert_close(logits, original)


def test_presence_excludes_prompt_and_repetition_includes_it():
    from ldat_eval.decoding import DecodeOptions
    logits = torch.tensor([4., -4., 3.])
    opts = DecodeOptions.from_kwargs(dict(presence_penalty=0.5, repetition_penalty=2))
    scores = opts.process(logits, [1, 1], [0])
    torch.testing.assert_close(scores, torch.tensor([2., -8.5, 3.]))
    torch.testing.assert_close(logits, torch.tensor([4., -4., 3.]))


@pytest.mark.parametrize("kwargs", [
    {"do_sample": True, "temperature": 0}, {"top_p": 0}, {"top_p": 1.1},
    {"top_k": -1}, {"top_k": 1.5}, {"temperature": float("nan")},
    {"presence_penalty": float("inf")}, {"repetition_penalty": 0},
    {"num_beams": 2}, {"num_return_sequences": 2}, {"do_sample": "maybe"},
    {"unknown_option": 1},
])
def test_invalid_or_unsupported_generation_options_fail_visibly(kwargs):
    with pytest.raises(ValueError):
        decode([[1, 0, 0, 0, -1]], **kwargs)


def test_immediate_eos_is_one_generated_token():
    answer, meta = decode([[0, 0, 0, 0, 9]])
    assert answer == "" and meta["output_tokens"] == 1
    assert meta["finish_reason"] == "eos"


def test_nonpositive_length_rejected():
    with pytest.raises(ValueError, match="max_new_tokens"):
        decode([[1, 0, 0, 0, -1]], limit=0)


def test_adapter_applies_options_and_persists_raw_trace(tmp_path, monkeypatch):
    import json
    from PIL import Image
    from lmms_eval.api.instance import Instance, unwrap_generation_output
    from lmms_eval.api.model import lmms
    from ldat_eval.models.qwen3_5_ldat import Qwen3_5_LDAT

    runner, initial = decoder([[2, 1.9, -10, -10, -10]])
    runner.last_n_tokens = 0
    runner.prepare = lambda image, question: {"n_tokens": 1, "ids": torch.tensor([0])}
    runner._prefill = lambda pr: (initial, torch.zeros(3, 1, 1), 1)
    model = Qwen3_5_LDAT.__new__(Qwen3_5_LDAT)
    lmms.__init__(model)
    model.runner = runner
    model._tokenizer = SimpleNamespace(eos_token_id=4, decode=lambda x: "<eos>", encode=lambda s: [0])
    model.batch_size_per_gpu = 1
    model._printed_tok = True
    model._warned_multi = False
    image = Image.new("RGB", (2, 2))
    model.task_dict = {"vstar": {"test": [{"image": image}, {"image": None}]}}
    opts = dict(max_new_tokens=3, do_sample=False, presence_penalty=1.5, until=[])
    requests = [Instance("generate_until", ("question", opts, lambda d: d["image"], i, "vstar", "test"),
                         i, metadata={"task": "vstar", "doc_id": i, "repeats": 1}) for i in (0, 1)]
    requests.append(Instance("generate_until", requests[0].arguments, 2,
                             metadata={"task": "vstar", "doc_id": 0, "repeats": 1,
                                       "__padding_only__": True}))
    monkeypatch.setenv("LDAT_DECODE_TRACE_DIR", str(tmp_path))
    results = model.generate_until(requests)
    text, count = unwrap_generation_output(results[0])
    assert text == "0 1 0"  # The current adapter silently drops presence_penalty.
    assert count.input_tokens == 1 and count.output_tokens == 3
    assert unwrap_generation_output(results[1])[1].output_tokens == 0
    rows = [json.loads(line) for line in (tmp_path / "rank0.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and sum(r["padding_only"] for r in rows) == 1
    rows = [r for r in rows if not r["padding_only"]]
    assert {r["doc_id"] for r in rows} == {0, 1}
    record = next(r for r in rows if r["doc_id"] == 0)
    assert record["raw_response"] == "0 1 0" and record["finish_reason"] == "length"
    assert record["generation_kwargs"]["presence_penalty"] == 1.5
    missing = next(r for r in rows if r["doc_id"] == 1)
    assert missing["finish_reason"] == "missing_image" and missing["output_tokens"] == 0


def test_real_qwen35_hybrid_cache_matches_native_greedy_and_supports_sampling():
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    torch.manual_seed(17)
    config = Qwen3_5Config(
        text_config=dict(vocab_size=32, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=16, partial_rotary_factor=1., linear_num_key_heads=1,
                         linear_num_value_heads=1, linear_key_head_dim=16, linear_value_head_dim=16,
                         layer_types=["linear_attention", "full_attention"],
                         rope_parameters=dict(rope_type="default", rope_theta=10000.,
                                              partial_rotary_factor=1., mrope_section=[2, 3, 3])),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
                           out_hidden_size=32, patch_size=16, spatial_merge_size=2,
                           temporal_patch_size=2, num_position_embeddings=16),
        image_token_id=29, video_token_id=30, vision_start_token_id=28,
    )
    model = Qwen3_5ForConditionalGeneration(config).eval()
    ids = torch.tensor([[1, 2, 3]])
    positions = torch.arange(3).view(1, 1, 3).expand(3, 1, 3)
    runner, _ = decoder([[0, 0, 0, 0, 0]])
    runner.eos = set()
    runner.backbone = SimpleNamespace(forward=model.forward)
    with torch.inference_mode():
        expected = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                  max_new_tokens=6, do_sample=False, pad_token_id=0)
        initial = model(input_ids=ids, position_ids=positions, use_cache=True)
        runner._decode(initial, positions, 3, 6, prompt_ids=ids[0].tolist())
        assert runner.last_generation["generated_token_ids"] == expected[0, 3:].tolist()
        records = []
        for _ in range(2):
            torch.manual_seed(1234)
            initial = model(input_ids=ids, position_ids=positions, use_cache=True)
            runner._decode(initial, positions, 3, 6, prompt_ids=ids[0].tolist(),
                           generation_kwargs=dict(do_sample=True, temperature=0.7, top_p=0.8,
                                                  top_k=20, presence_penalty=1.5))
            records.append(runner.last_generation["generated_token_ids"])
        assert records[0] == records[1]
        assert records[0] != expected[0, 3:].tolist()
        assert len(records[0]) == 6
