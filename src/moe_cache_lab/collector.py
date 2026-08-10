"""Transformers-based collection of measured Granite MoE router selections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .trace import RoutingEvent, RoutingTrace

DEFAULT_MODEL = "ibm-granite/granite-3.1-1b-a400m-instruct"
STAGE1_MAX_DECODE_INPUT_STEPS = 16


@dataclass(frozen=True)
class ModelInspection:
    model_id: str
    model_type: str | None
    architectures: tuple[str, ...]
    num_experts: int | None
    experts_per_token: int | None
    router_logits_supported: bool


@dataclass(frozen=True)
class Stage1CollectionOutcome:
    """One EOS-aware Stage 1 prompt result and its immutable routing trace."""

    trace: RoutingTrace
    max_decode_input_steps_requested: int
    actual_decode_input_steps_routed: int
    routed_non_eos_token_ids: tuple[int, ...]
    emitted_non_eos_token_ids: tuple[int, ...]
    emitted_non_eos_text: str
    normalized_eos_token_ids: tuple[int, ...]
    terminal_eos_token_id: int | None
    terminal_eos_candidate_position: int | None
    terminal_eos_text: str | None
    eos_emitted: bool
    horizon_exhausted: bool

    def __post_init__(self) -> None:
        if not isinstance(self.trace, RoutingTrace):
            raise TypeError("Stage 1 outcome trace must be a RoutingTrace")
        if self.max_decode_input_steps_requested != STAGE1_MAX_DECODE_INPUT_STEPS:
            raise ValueError("Stage 1 requires exactly 16 maximum decode-input steps")
        if (
            isinstance(self.actual_decode_input_steps_routed, bool)
            or not isinstance(self.actual_decode_input_steps_routed, int)
            or not 0 <= self.actual_decode_input_steps_routed <= STAGE1_MAX_DECODE_INPUT_STEPS
        ):
            raise ValueError("actual routed steps must be an integer from zero through sixteen")
        if self.actual_decode_input_steps_routed != len(self.routed_non_eos_token_ids):
            raise ValueError("actual routed steps must match routed non-EOS token IDs")
        if self.routed_non_eos_token_ids != self.emitted_non_eos_token_ids:
            raise ValueError("routed and emitted non-EOS token IDs must match exactly")
        if (
            not self.normalized_eos_token_ids
            or self.normalized_eos_token_ids != tuple(sorted(set(self.normalized_eos_token_ids)))
            or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0
                for token in (*self.normalized_eos_token_ids, *self.routed_non_eos_token_ids)
            )
        ):
            raise ValueError("Stage 1 requires at least one normalized EOS token ID")
        if any(token in self.normalized_eos_token_ids for token in self.routed_non_eos_token_ids):
            raise ValueError("terminal EOS must never appear in routed non-EOS token IDs")
        if self.eos_emitted == self.horizon_exhausted:
            raise ValueError("exactly one of eos_emitted and horizon_exhausted must be true")
        if self.eos_emitted:
            if (
                self.terminal_eos_token_id not in self.normalized_eos_token_ids
                or self.terminal_eos_candidate_position != self.actual_decode_input_steps_routed
                or self.terminal_eos_text is None
            ):
                raise ValueError("terminal EOS metadata is inconsistent")
        elif (
            self.terminal_eos_token_id is not None
            or self.terminal_eos_candidate_position is not None
            or self.terminal_eos_text is not None
            or self.actual_decode_input_steps_routed != STAGE1_MAX_DECODE_INPUT_STEPS
        ):
            raise ValueError("horizon exhaustion metadata is inconsistent")
        if self.trace.generated_text != self.emitted_non_eos_text:
            raise ValueError("trace generated text must match emitted non-EOS text")
        prompt_positions = {
            event.token_position for event in self.trace.events if event.phase == "prompt"
        }
        generated_positions = sorted({
            event.token_position for event in self.trace.events if event.phase == "generated"
        })
        expected_positions = list(range(
            len(prompt_positions),
            len(prompt_positions) + self.actual_decode_input_steps_routed,
        ))
        if generated_positions != expected_positions:
            raise ValueError("trace generated positions must match actual routed steps")
        for step, position in enumerate(generated_positions):
            token_ids = {
                event.token_id for event in self.trace.events
                if event.phase == "generated" and event.token_position == position
            }
            if token_ids != {self.routed_non_eos_token_ids[step]}:
                raise ValueError("trace generated token IDs must match routed non-EOS tokens")


def inspect_model(model_id: str = DEFAULT_MODEL, revision: str | None = None) -> ModelInspection:
    """Download config only and determine whether it exposes MoE fields."""
    config = AutoConfig.from_pretrained(model_id, revision=revision)
    experts = _config_value(config, "num_local_experts", "num_experts")
    per_token = _config_value(config, "num_experts_per_tok", "num_experts_per_token")
    return ModelInspection(
        model_id=model_id, model_type=getattr(config, "model_type", None),
        architectures=tuple(getattr(config, "architectures", None) or ()),
        num_experts=experts, experts_per_token=per_token,
        router_logits_supported=bool(getattr(config, "output_router_logits", False) or experts is not None),
    )


def collect_trace(
    prompt: str,
    model_id: str = DEFAULT_MODEL,
    max_new_tokens: int = 0,
    *,
    revision: str | None = None,
) -> RoutingTrace:
    """Load once for a single prompt and collect a measured routing trace."""
    return GraniteTraceCollector(model_id, revision=revision).collect(prompt, max_new_tokens)


class GraniteTraceCollector:
    """Reusable CPU-float32 collector that keeps one model/tokenizer loaded."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        revision: str | None = None,
        *,
        local_files_only: bool = False,
    ) -> None:
        self.model_id = model_id
        self.requested_revision = revision
        offline = {"local_files_only": True} if local_files_only else {}
        config = AutoConfig.from_pretrained(model_id, revision=revision, **offline)
        resolved_revision = getattr(config, "_commit_hash", None)
        if not isinstance(resolved_revision, str) or not resolved_revision.strip():
            raise RuntimeError(f"could not resolve an immutable Hugging Face revision for {model_id}")
        self.resolved_revision = resolved_revision
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=self.resolved_revision,
            config=config,
            **offline,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=self.resolved_revision,
            config=config,
            torch_dtype=torch.float32,
            **offline,
        )
        self.model.to(device="cpu", dtype=torch.float32)
        self.model.eval()
        self.num_experts = _config_value(self.model.config, "num_local_experts", "num_experts")
        self.experts_per_token = _config_value(
            self.model.config, "num_experts_per_tok", "num_experts_per_token"
        )
        if self.num_experts is None or self.experts_per_token is None:
            raise RuntimeError(f"{model_id} config does not expose expected MoE fields")

    def collect(self, prompt: str, max_new_tokens: int = 0) -> RoutingTrace:
        """Collect one independent prompt context with deterministic greedy decode.

        Generation uses standard autoregressive forward calls so router outputs
        remain observable without modifying Transformers internals.
        """
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to("cpu")
        encoded = {name: value.to("cpu") for name, value in encoded.items()}
        capture = _RouterHookCapture(self.model)
        events: list[RoutingEvent] = []
        with capture, torch.inference_mode():
            prompt_outputs = self.model(**encoded, use_cache=True)
            events.extend(_events_from_router_logits(
                capture.take(), input_ids, "prompt", 0, self.experts_per_token
            ))
            past = prompt_outputs.past_key_values
            next_token = prompt_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids: list[int] = []
            for offset in range(max_new_tokens):
                generated_ids.append(int(next_token[0, 0]))
                generated_outputs = self.model(input_ids=next_token, past_key_values=past, use_cache=True)
                events.extend(_events_from_router_logits(
                    capture.take(), next_token, "generated", input_ids.shape[1] + offset,
                    self.experts_per_token,
                ))
                past = generated_outputs.past_key_values
                next_token = generated_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True) if generated_ids else ""
        import transformers
        return RoutingTrace(
            model_id=self.model_id, num_experts=self.num_experts,
            experts_per_token=self.experts_per_token, events=tuple(events), source_text=prompt,
            generated_text=generated_text,
            capture_method="PyTorch forward hooks on Granite MoE router modules; greedy autoregressive forward calls for generated tokens",
            transformers_version=transformers.__version__,
            model_revision=self.resolved_revision,
        )

    def collect_stage1(
        self,
        prompt: str,
        max_decode_input_steps: int = STAGE1_MAX_DECODE_INPUT_STEPS,
    ) -> Stage1CollectionOutcome:
        """Collect one prompt with the frozen candidate-before-feed EOS protocol."""

        if max_decode_input_steps != STAGE1_MAX_DECODE_INPUT_STEPS:
            raise ValueError("Stage 1 max_decode_input_steps is frozen at 16")
        eos_token_ids = _normalize_eos_token_ids(self.tokenizer)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to("cpu")
        encoded = {name: value.to("cpu") for name, value in encoded.items()}
        capture = _RouterHookCapture(self.model)
        events: list[RoutingEvent] = []
        routed_ids: list[int] = []
        terminal_eos_token_id: int | None = None
        terminal_eos_candidate_position: int | None = None
        terminal_eos_text: str | None = None
        with capture, torch.inference_mode():
            prompt_outputs = self.model(**encoded, use_cache=True)
            events.extend(_events_from_router_logits(
                capture.take(), input_ids, "prompt", 0, self.experts_per_token
            ))
            past = prompt_outputs.past_key_values
            candidate = prompt_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            while len(routed_ids) < max_decode_input_steps:
                candidate_id = int(candidate[0, 0])
                if candidate_id in eos_token_ids:
                    terminal_eos_token_id = candidate_id
                    terminal_eos_candidate_position = len(routed_ids)
                    terminal_eos_text = self.tokenizer.decode(
                        [candidate_id], skip_special_tokens=False
                    )
                    break
                routed_ids.append(candidate_id)
                generated_outputs = self.model(
                    input_ids=candidate,
                    past_key_values=past,
                    use_cache=True,
                )
                events.extend(_events_from_router_logits(
                    capture.take(),
                    candidate,
                    "generated",
                    input_ids.shape[1] + len(routed_ids) - 1,
                    self.experts_per_token,
                ))
                past = generated_outputs.past_key_values
                candidate = generated_outputs.logits[:, -1, :].argmax(
                    dim=-1, keepdim=True
                )
        eos_emitted = terminal_eos_token_id is not None
        horizon_exhausted = len(routed_ids) == max_decode_input_steps and not eos_emitted
        emitted_text = self.tokenizer.decode(routed_ids, skip_special_tokens=True)
        import transformers
        trace = RoutingTrace(
            model_id=self.model_id,
            num_experts=self.num_experts,
            experts_per_token=self.experts_per_token,
            events=tuple(events),
            source_text=prompt,
            generated_text=emitted_text,
            capture_method="PyTorch forward hooks on Granite MoE router modules; greedy autoregressive forward calls for generated tokens",
            transformers_version=transformers.__version__,
            model_revision=self.resolved_revision,
        )
        return Stage1CollectionOutcome(
            trace=trace,
            max_decode_input_steps_requested=max_decode_input_steps,
            actual_decode_input_steps_routed=len(routed_ids),
            routed_non_eos_token_ids=tuple(routed_ids),
            emitted_non_eos_token_ids=tuple(routed_ids),
            emitted_non_eos_text=emitted_text,
            normalized_eos_token_ids=eos_token_ids,
            terminal_eos_token_id=terminal_eos_token_id,
            terminal_eos_candidate_position=terminal_eos_candidate_position,
            terminal_eos_text=terminal_eos_text,
            eos_emitted=eos_emitted,
            horizon_exhausted=horizon_exhausted,
        )


def _events_from_router_logits(
    router_logits: Any, token_ids: torch.Tensor, phase: str, position_offset: int, top_k: int,
) -> list[RoutingEvent]:
    if router_logits is None:
        raise RuntimeError("model did not return router_logits; this Transformers/model combination cannot be measured safely")
    events: list[RoutingEvent] = []
    batch_size, sequence_length = token_ids.shape
    if batch_size != 1:
        raise ValueError("V0.1 collector currently supports batch size 1")
    layers = router_logits.items() if isinstance(router_logits, dict) else enumerate(router_logits)
    for layer, layer_logits in layers:
        if layer_logits is None:
            continue
        if not isinstance(layer_logits, torch.Tensor):
            raise RuntimeError(f"unexpected router logits type at layer {layer}: {type(layer_logits)!r}")
        logits = layer_logits.detach().float().cpu()
        if logits.ndim == 2:
            if logits.shape[0] != batch_size * sequence_length:
                raise RuntimeError(f"router logits shape {tuple(logits.shape)} does not match {sequence_length} tokens")
            logits = logits.reshape(batch_size, sequence_length, logits.shape[-1])
        elif logits.ndim != 3:
            raise RuntimeError(f"unexpected router logits shape {tuple(logits.shape)}")
        selected_logits, indices = logits.topk(k=min(top_k, logits.shape[-1]), dim=-1)
        # Granite's router normalizes only the selected top-k logits. Match the
        # actual router gate weights instead of reporting a different softmax.
        values = torch.softmax(selected_logits, dim=-1)
        for token_index in range(sequence_length):
            events.append(RoutingEvent(
                phase=phase, token_position=position_offset + token_index, layer=layer,
                token_id=int(token_ids[0, token_index]),
                selected_experts=tuple(int(item) for item in indices[0, token_index]),
                selected_probabilities=tuple(float(item) for item in values[0, token_index]),
            ))
    if not events:
        raise RuntimeError("router_logits contained no routed layers")
    return events


def _config_value(config: Any, *names: str) -> int | None:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    return None


def _normalize_eos_token_ids(tokenizer: Any) -> tuple[int, ...]:
    values: list[int] = []
    for name in ("eos_token_id", "eos_token_ids"):
        raw = getattr(tokenizer, name, None)
        if raw is None:
            continue
        candidates = raw if isinstance(raw, (list, tuple, set, frozenset)) else (raw,)
        for candidate in candidates:
            if isinstance(candidate, bool) or not isinstance(candidate, int) or candidate < 0:
                raise ValueError("tokenizer EOS token IDs must be non-negative integers")
            values.append(candidate)
    normalized = tuple(sorted(set(values)))
    if not normalized:
        raise ValueError("tokenizer does not expose an EOS token ID")
    return normalized


class _RouterHookCapture:
    """Observe raw router logits without modifying Transformers or model weights.

    Transformers 5.12's GraniteMoe implementation exposes an
    ``output_router_logits`` argument but does not populate the public output.
    The router module itself returns its exact logits as its final return value,
    so a temporary PyTorch hook is the smallest reliable observation point.
    """

    def __init__(self, model: Any) -> None:
        self._handles: list[Any] = []
        self._logits_by_layer: dict[int, torch.Tensor] = {}
        try:
            for name, module in model.named_modules():
                if not name.endswith(".block_sparse_moe.router"):
                    continue
                layer = _layer_index(name)
                self._handles.append(module.register_forward_hook(self._hook(layer)))
        except BaseException as original_error:
            cleanup_errors: list[Exception] = []
            for handle in self._handles:
                try:
                    handle.remove()
                except Exception as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            self._handles.clear()
            if cleanup_errors:
                raise RuntimeError(
                    "router hook registration failed and partial-handle cleanup also failed: "
                    + "; ".join(repr(error) for error in cleanup_errors)
                ) from original_error
            raise
        if not self._handles:
            raise RuntimeError("could not locate Granite MoE router modules for observation")

    def _hook(self, layer: int):
        def hook(_: Any, __: tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, tuple) or not isinstance(output[-1], torch.Tensor):
                raise RuntimeError("Granite router hook received an unexpected output")
            self._logits_by_layer[layer] = output[-1].detach()
        hook._moe_cache_lab_router_hook = True  # type: ignore[attr-defined]
        return hook

    def take(self) -> dict[int, torch.Tensor]:
        if not self._logits_by_layer:
            raise RuntimeError("router hooks captured no logits")
        result = dict(self._logits_by_layer)
        self._logits_by_layer.clear()
        return result

    def __enter__(self) -> "_RouterHookCapture":
        return self

    def __exit__(self, *_: Any) -> None:
        first_error: Exception | None = None
        try:
            for handle in self._handles:
                try:
                    handle.remove()
                except Exception as error:
                    if first_error is None:
                        first_error = error
        finally:
            self._handles.clear()
            self._logits_by_layer.clear()
        if first_error is not None:
            raise first_error


def _layer_index(module_name: str) -> int:
    parts = module_name.split(".")
    try:
        return int(parts[parts.index("layers") + 1])
    except (ValueError, IndexError) as error:
        raise RuntimeError(f"cannot determine layer index for router {module_name}") from error
