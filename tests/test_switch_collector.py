from types import SimpleNamespace
import unittest

import torch
from torch import nn

from moe_cache_lab.switch_collector import (
    SwitchRouterObservation,
    SwitchTraceCollector,
    _SwitchRouterCapture,
    _discover_switch_sparse_mlps,
    _events_from_switch_observations,
)
from moe_cache_lab.trace_v2 import RoutingStageProfile, RoutingTraceV2


class _FakeRouter(nn.Module):
    def __init__(self, num_experts: int = 2) -> None:
        super().__init__()
        self.num_experts = num_experts

    def forward(self, hidden: torch.Tensor):
        token_count = hidden.shape[0]
        weights = torch.full((token_count, 1), 0.75)
        mask = torch.zeros((token_count, 1, self.num_experts), dtype=torch.long)
        mask[:, 0, 1] = 1
        return weights, mask, weights.clone()


class _FakeExpert(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class _FakeSparseMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = _FakeRouter()
        self.experts = nn.ModuleDict(
            {"expert_0": _FakeExpert(), "expert_1": _FakeExpert()}
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        weights, mask, routing_weights = self.router(hidden)
        del weights
        output = torch.zeros_like(hidden)
        for expert_id, expert in enumerate(self.experts.values()):
            selected = mask[:, 0, expert_id].bool()
            if bool(selected.any()):
                output[selected] = expert(hidden[selected]) * routing_weights[selected]
        return output


class _FakeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _FakeSparseMLP()


class _FakeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.is_sparse = True
        self.layer = nn.ModuleList([_FakeLayer()])


class _FakeStack(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.ModuleList([_FakeBlock()])

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        use_cache=False,
        return_dict=True,
    ):
        del attention_mask, use_cache, return_dict
        hidden = torch.zeros((input_ids.shape[1], 3))
        self.block[0].layer[-1].mlp(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class _FakeSwitchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _FakeStack()
        self.decoder = _FakeStack()
        self.config = SimpleNamespace(decoder_start_token_id=0, eos_token_id=1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        past_key_values=None,
        use_cache=True,
        return_dict=True,
    ):
        del attention_mask, past_key_values, use_cache, return_dict
        if encoder_outputs is None:
            encoder_hidden = self.encoder(input_ids=input_ids).last_hidden_state
        else:
            encoder_hidden = (
                encoder_outputs.last_hidden_state
                if hasattr(encoder_outputs, "last_hidden_state")
                else encoder_outputs[0]
            )
        decoder_hidden = torch.zeros((decoder_input_ids.shape[1], 3))
        self.decoder.block[0].layer[-1].mlp(decoder_hidden)
        next_id = 5 if int(decoder_input_ids[0, -1]) == 0 else 1
        logits = torch.zeros((1, decoder_input_ids.shape[1], 8))
        logits[0, -1, next_id] = 1.0
        return SimpleNamespace(
            logits=logits,
            past_key_values=("fake",),
            encoder_last_hidden_state=encoder_hidden,
        )


class _FakeTokenizer:
    eos_token_id = 1

    def __call__(self, _: str, return_tensors: str):
        if return_tensors != "pt":
            raise AssertionError("test tokenizer requires PyTorch tensors")
        return {
            "input_ids": torch.tensor([[10, 11]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }

    def decode(self, token_ids, skip_special_tokens: bool):
        del skip_special_tokens
        return " ".join(str(value) for value in token_ids)


def _observation(mask: torch.Tensor, weight: float = 0.9) -> SwitchRouterObservation:
    tokens = mask.shape[0]
    weights = torch.full((tokens, 1), weight)
    return SwitchRouterObservation(
        routing_stage="encoder",
        layer=1,
        output_shapes=(tuple(weights.shape), tuple(mask.shape), tuple(weights.shape)),
        post_capacity_assignment=mask,
        routing_weight=weights,
    )


class SwitchCollectorTests(unittest.TestCase):
    def test_hook_registration_preserves_stage_layer_and_callback_order(self) -> None:
        model = _FakeSwitchModel()
        with _SwitchRouterCapture(model) as capture:
            model(
                input_ids=torch.tensor([[10, 11]]),
                decoder_input_ids=torch.tensor([[0]]),
            )
            observations = capture.take()

        self.assertEqual(
            [(item.routing_stage, item.layer) for item in observations],
            [("encoder", 0), ("decoder", 0)],
        )
        self.assertEqual(
            [(stage, layer) for stage, layer, _ in _discover_switch_sparse_mlps(model)],
            [("encoder", 0), ("decoder", 0)],
        )

    def test_assigned_mask_uses_actual_expert_and_native_weight(self) -> None:
        mask = torch.tensor([[[0, 1]]])
        events = _events_from_switch_observations(
            (_observation(mask, 0.7),),
            {"encoder": torch.tensor([[22]])},
            {"encoder": "source"},
            {"encoder": 0},
            2,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].assignment_state, "assigned")
        self.assertEqual(events[0].selected_experts, (1,))
        self.assertAlmostEqual(events[0].selected_probabilities[0], 0.7)

    def test_zero_post_capacity_mask_does_not_fabricate_preference(self) -> None:
        events = _events_from_switch_observations(
            (_observation(torch.zeros((1, 1, 2), dtype=torch.long), 0.99),),
            {"encoder": torch.tensor([[22]])},
            {"encoder": "source"},
            {"encoder": 0},
            2,
        )

        event = events[0]
        self.assertEqual(event.assignment_state, "unassigned")
        self.assertEqual(event.selected_experts, ())
        self.assertEqual(event.selected_probabilities, ())
        self.assertEqual(event.unassigned_reason, "capacity")
        self.assertEqual(event.expert_requests, ())

    def test_malformed_mask_shape_cardinality_and_values_fail(self) -> None:
        malformed = (
            torch.zeros((1, 2), dtype=torch.long),
            torch.tensor([[[1, 1]]]),
            torch.tensor([[[0, 2]]]),
        )
        for mask in malformed:
            with self.subTest(mask=mask.tolist()), self.assertRaises(RuntimeError):
                _events_from_switch_observations(
                    (_observation(mask),),
                    {"encoder": torch.tensor([[22]])},
                    {"encoder": "source"},
                    {"encoder": 0},
                    2,
                )

    def test_callback_identity_is_validated_without_sorting(self) -> None:
        first = _observation(torch.tensor([[[0, 1]]]))
        second = SwitchRouterObservation(
            "encoder",
            0,
            first.output_shapes,
            first.post_capacity_assignment,
            first.routing_weight,
        )
        with self.assertRaisesRegex(RuntimeError, "callback order"):
            _events_from_switch_observations(
                (first, second),
                {"encoder": torch.tensor([[22]])},
                {"encoder": "source"},
                {"encoder": 0},
                2,
                expected_callback_identity=(("encoder", 0), ("encoder", 1)),
            )

    def test_bounded_driver_maps_phases_and_omits_terminal_unfed_candidate(self) -> None:
        collector = object.__new__(SwitchTraceCollector)
        collector.model = _FakeSwitchModel()
        collector.tokenizer = _FakeTokenizer()
        collector.num_experts = 2
        collector._sparse_mlps = _discover_switch_sparse_mlps(collector.model)
        collector.sparse_layers = (("encoder", (0,)), ("decoder", (0,)))
        collector.model_id = "google/switch-base-8"
        collector.resolved_revision = "92fe2d22b024d9937146fe097ba3d3a7ba146e1b"

        outcome = collector.collect_and_verify("public synthetic prompt")

        self.assertEqual(outcome.candidate_token_ids, (5, 1))
        self.assertEqual(outcome.fed_decoder_generated_token_ids, (5,))
        self.assertEqual(outcome.terminal_eos_token_id, 1)
        self.assertFalse(outcome.horizon_exhausted)
        self.assertEqual(
            [
                (event.routing_stage, event.phase, event.token_position, event.layer)
                for event in outcome.trace.events
            ],
            [
                ("encoder", "source", 0, 0),
                ("encoder", "source", 1, 0),
                ("decoder", "decoder_prompt", 0, 0),
                ("decoder", "decoder_generated", 1, 0),
            ],
        )
        self.assertNotIn(2, [event.token_position for event in outcome.trace.events])
        self.assertIsInstance(outcome.trace, RoutingTraceV2)
        self.assertEqual(
            outcome.trace.routing_stages,
            (
                RoutingStageProfile("encoder", 2, 1, True),
                RoutingStageProfile("decoder", 2, 1, True),
            ),
        )
        self.assertTrue(all(check.nonzero_assignments == check.expert_input_rows for check in outcome.dispatch_checks))


if __name__ == "__main__":
    unittest.main()
