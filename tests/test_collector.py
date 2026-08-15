import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn
from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeTopKGating

from moe_cache_lab.collector import GraniteTraceCollector, _RouterHookCapture, _events_from_router_logits


class _SparseMoe(nn.Module):
    def __init__(self, router: nn.Module) -> None:
        super().__init__()
        self.router = router


class _Layer(nn.Module):
    def __init__(self, router: nn.Module) -> None:
        super().__init__()
        self.block_sparse_moe = _SparseMoe(router)


class _TinyGraniteShape(nn.Module):
    def __init__(self, router: nn.Module) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer(router)])


class GraniteRouterCompatibilityTests(unittest.TestCase):
    def test_collector_resolves_and_pins_one_official_config_revision_without_download(self) -> None:
        config = SimpleNamespace(
            _commit_hash="0123456789abcdef",
            num_local_experts=4,
            num_experts_per_tok=2,
        )
        model = Mock()
        model.config = config
        tokenizer = Mock()
        with (
            patch("moe_cache_lab.collector.AutoConfig.from_pretrained", return_value=config) as config_load,
            patch("moe_cache_lab.collector.AutoTokenizer.from_pretrained", return_value=tokenizer) as tokenizer_load,
            patch("moe_cache_lab.collector.AutoModelForCausalLM.from_pretrained", return_value=model) as model_load,
        ):
            collector = GraniteTraceCollector("example/moe", revision="release-tag")

        config_load.assert_called_once_with("example/moe", revision="release-tag")
        tokenizer_load.assert_called_once_with(
            "example/moe", revision="0123456789abcdef", config=config
        )
        model_load.assert_called_once_with(
            "example/moe",
            revision="0123456789abcdef",
            config=config,
            torch_dtype=torch.float32,
        )
        self.assertEqual(collector.resolved_revision, "0123456789abcdef")

    def test_official_router_dispatch_matches_extraction_and_hook_is_observational(self) -> None:
        router = GraniteMoeTopKGating(input_size=3, num_experts=4, top_k=2)
        with torch.no_grad():
            router.layer.weight.copy_(torch.tensor([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.5, 0.5, -0.5],
            ]))
        hidden = torch.tensor([[3.0, 1.0, 0.0], [0.0, 2.0, 4.0]])
        baseline = router(hidden)

        model = _TinyGraniteShape(router)
        with _RouterHookCapture(model) as capture:
            observed = router(hidden)
            captured_logits = capture.take()

        for expected, actual in zip(baseline, observed):
            if isinstance(expected, torch.Tensor):
                torch.testing.assert_close(actual, expected)
            else:
                self.assertEqual(actual, expected)

        logits = baseline[-1]
        top_k_logits, top_k_indices = logits.topk(2, dim=1)
        top_k_probabilities = torch.softmax(top_k_logits, dim=1)
        events = _events_from_router_logits(captured_logits, torch.tensor([[10, 11]]), "prompt", 0, 2)
        self.assertEqual([item.selected_experts for item in events], [tuple(row.tolist()) for row in top_k_indices])
        for item, expected in zip(events, top_k_probabilities):
            torch.testing.assert_close(torch.tensor(item.selected_probabilities), expected)

        index_sorted_experts, batch_index, batch_gates, expert_size, _ = baseline
        flattened_experts = top_k_indices.flatten()
        torch.testing.assert_close(batch_index, index_sorted_experts.div(2, rounding_mode="trunc"))
        torch.testing.assert_close(batch_gates, top_k_probabilities.flatten()[index_sorted_experts])
        expected_grouped = torch.repeat_interleave(torch.arange(4), torch.tensor(expert_size))
        torch.testing.assert_close(flattened_experts[index_sorted_experts], expected_grouped)
