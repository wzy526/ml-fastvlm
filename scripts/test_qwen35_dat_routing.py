#!/usr/bin/env python3
"""CPU/stdlib regression checks for Qwen3.5 DAT's next-token query routing.

Run: python scripts/test_qwen35_dat_routing.py
The production functions are extracted by AST to avoid importing CUDA attention
backends. Tensor parser checks run when torch is installed; all other checks use
only the standard library.
"""

import ast
import copy
import itertools
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "llava/model/language_model/modeling_qwen3_5_dat.py"
)
MODEL_AST = ast.parse(MODEL_PATH.read_text(), filename=str(MODEL_PATH))
FUNCTION_NAMES = {
    "_find_im_start_backward",
    "_dat_answer_query_range",
    "_dat_question_ranges",
    "compute_image_range_list",
}


def load_production_functions():
    selected = [
        copy.deepcopy(node)
        for node in MODEL_AST.body
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTION_NAMES
        or isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "IM_START_TOKEN_ID"
                for t in node.targets)
    ]
    assert {n.name for n in selected if isinstance(n, ast.FunctionDef)} == FUNCTION_NAMES
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(module, str(MODEL_PATH), "exec"), namespace)
    return namespace


PRODUCTION = load_production_functions()
answer_queries = PRODUCTION["_dat_answer_query_range"]
question_queries = PRODUCTION["_dat_question_ranges"]


class QueryRoutingTests(unittest.TestCase):
    def test_inclusive_labels_shift_to_queries(self):
        cases = [
            ([5, 5, 2], 10, (4, 5)),  # one valid label
            ([5, 8, 2], 10, (4, 8)),
            ([9, 9, 2], 10, (8, 9)),  # final non-padding label
            ([0, 0, 0], 10, (0, 0)),  # label 0 is absent from shifted CE
            ([0, 2, 0], 10, (0, 2)),
            ([0, 0, 0], 0, (0, 0)),
        ]
        for answer, length, expected in cases:
            with self.subTest(answer=answer, length=length):
                self.assertEqual(answer_queries(answer, length), expected)

    def test_all_short_label_masks_match_shifted_ce(self):
        # The oracle is the loss's label shift, independent of range endpoints.
        # Exhaust all short masks, including label 0, padding, and multiple turns.
        length = 8
        for valid in itertools.product((False, True), repeat=length):
            ranges = []
            for supervised, group in itertools.groupby(
                enumerate(valid), key=lambda item: item[1]
            ):
                indices = [index for index, _ in group]
                if supervised:
                    ranges.append([indices[0], indices[-1], 0])
            actual = [
                query for answer in ranges
                for query in range(*answer_queries(answer, length))
            ]
            expected = [index - 1 for index in range(1, length) if valid[index]]
            self.assertEqual(actual, expected, valid)
            questions = [
                query for interval in question_queries(0, ranges, length)
                for query in range(*interval)
            ]
            self.assertEqual(len(questions), len(set(questions)), valid)
            self.assertTrue(set(questions).isdisjoint(actual), valid)
            self.assertTrue(all(not valid[query + 1] for query in questions), valid)

    def test_multiturn_question_segments_exclude_answer_queries(self):
        # Optional metadata remains a text span, not an HD query interval.
        answers = [[9, 10, 6, 5, 9], [15, 16, 12, 11, 15]]
        self.assertEqual(question_queries(5, answers, 19), [(5, 8), (10, 14)])
        self.assertEqual([answer_queries(a, 19) for a in answers], [(8, 10), (14, 16)])

    def test_empty_answer_does_not_move_question_start_before_image(self):
        self.assertEqual(question_queries(5, [[0, 0, 0], [9, 9, 6]], 10), [(5, 8)])
        self.assertEqual(question_queries(5, [], 10), [])

    def test_prefill_keeps_assistant_prefix_interval(self):
        prefill = [9, -1, 6, 5, 7]
        self.assertEqual(answer_queries(prefill, 9), (7, 9))
        self.assertEqual(question_queries(5, [prefill], 9), [(5, 7)])
        # Both paths route the last prompt row to question-conditioned HD.
        # Their earlier assistant-prefix rows need not have identical routing.
        training = [9, 11, 6, 5, 9]
        self.assertIn(8, range(*answer_queries(training, 12)))
        self.assertIn(8, range(*answer_queries(prefill, 9)))


class CachedDecodeTests(unittest.TestCase):
    def test_production_forward_returns_to_base_before_sampling(self):
        attention = next(
            node for node in MODEL_AST.body
            if isinstance(node, ast.ClassDef) and node.name == "Qwen3_5AttentionDAT"
        )
        forward = next(
            node for node in attention.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )

        class BaseAttention:
            def forward(self, hidden_states, **kwargs):
                return hidden_states, kwargs

        class Cache:
            def get_seq_length(self, layer_idx):
                self.requested_layer = layer_idx
                return 7

        # Compile the actual method in a class so zero-argument super() retains
        # its class cell. An opaque hidden state cannot run the tensor path.
        stub = ast.ClassDef(
            name="RoutingAttention", bases=[ast.Name(id="BaseAttention", ctx=ast.Load())],
            keywords=[], body=[copy.deepcopy(forward)], decorator_list=[],
        )
        module = ast.Module(body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            stub,
        ], type_ignores=[])
        namespace = {"BaseAttention": BaseAttention}
        exec(compile(ast.fix_missing_locations(module), str(MODEL_PATH), "exec"), namespace)
        instance = namespace["RoutingAttention"]()
        instance.layer_idx = 3
        hidden, positions, cache = object(), object(), Cache()
        result, kwargs = instance.forward(
            hidden, positions, past_key_values=cache,
            image_hd_features=[object()],
            image_range_list=[[[(1, 5, 2, 2)], [9, -1, 6]]],
        )
        self.assertIs(result, hidden)
        self.assertIs(kwargs["position_embeddings"], positions)
        self.assertIs(kwargs["past_key_values"], cache)
        self.assertEqual(cache.requested_layer, 3)


@unittest.skipUnless(torch is not None, "torch is not installed; CPU tensor parser checks skipped")
class TensorRangeParserTests(unittest.TestCase):
    def setUp(self):
        self.image_token = 999
        im_start = PRODUCTION["IM_START_TOKEN_ID"]
        self.ids = torch.tensor([[
            0, 999, 999, 999, 999, 0, im_start, 10, 11,
            100, 101, 0, im_start, 10, 11, 102, 103, 0, 0,
        ]])
        self.parse = PRODUCTION["compute_image_range_list"]

    def test_single_label_and_text_span_are_preserved(self):
        labels = torch.full_like(self.ids, -100)
        labels[0, 9] = 100
        ranges = self.parse(self.ids, labels, self.image_token)[0]
        self.assertEqual(ranges, [[(1, 5, 2, 2)], [9, 9, 6, 5, 9]])
        self.assertEqual(answer_queries(ranges[1], self.ids.shape[1]), (8, 9))

    def test_multiturn_parser_matches_shifted_loss_mask(self):
        labels = torch.full_like(self.ids, -100)
        labels[0, 9:11] = self.ids[0, 9:11]
        labels[0, 15:17] = self.ids[0, 15:17]
        ranges = self.parse(self.ids, labels, self.image_token)[0]
        self.assertEqual(ranges[1:], [[9, 10, 6, 5, 9], [15, 16, 12, 11, 15]])
        actual = [q for a in ranges[1:] for q in range(*answer_queries(a, self.ids.shape[1]))]
        expected = torch.where(labels[0, 1:] != -100)[0].tolist()
        self.assertEqual(actual, expected)

    def test_label_zero_is_ignored_without_mutating_labels(self):
        labels = torch.full_like(self.ids, -100)
        labels[0, 0] = 123
        before = labels.clone()
        self.assertEqual(self.parse(self.ids, labels, self.image_token), [[[(1, 5, 2, 2)]]])
        self.assertTrue(torch.equal(labels, before))

    def test_prefill_metadata_is_unchanged(self):
        ranges = self.parse(self.ids[:, :9], None, self.image_token)[0]
        self.assertEqual(ranges, [[(1, 5, 2, 2)], [9, -1, 6, 5, 7]])
        self.assertEqual(answer_queries(ranges[1], 9), (7, 9))


if __name__ == "__main__":
    unittest.main(verbosity=2)
