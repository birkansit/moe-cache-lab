"""Native Transformers Switch routing observation for canonical trace v2.

This module is intentionally Switch-specific.  It observes the exact
post-capacity assignment returned by ``SwitchTransformersTop1Router`` and never
recomputes dispatch from pre-capacity logits or probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.models.switch_transformers.modeling_switch_transformers import (
    SwitchTransformersForConditionalGeneration,
)

from .switch_dependencies import DEFAULT_SWITCH_MODEL, SWITCH_MODEL_REVISION
from .trace_v2 import (
    RoutingEventV2,
    RoutingExpertKeyV2,
    RoutingStageProfile,
    RoutingTraceV2,
)


SWITCH_CAPTURE_METHOD = (
    "PyTorch forward hooks on native SwitchTransformersTop1Router outputs; "
    "actual experts derived from the returned post-capacity assignment tensor"
)


@dataclass(frozen=True)
class SwitchRouterObservation:
    """One native router callback in authoritative execution order."""

    routing_stage: str
    layer: int
    output_shapes: tuple[tuple[int, ...], ...]
    post_capacity_assignment: torch.Tensor
    routing_weight: torch.Tensor


@dataclass(frozen=True)
class SwitchDispatchCheck:
    """Independent row-count reconciliation for one sparse layer."""

    routing_stage: str
    layer: int
    nonzero_assignments: int
    expert_input_rows: int

    def __post_init__(self) -> None:
        if self.nonzero_assignments != self.expert_input_rows:
            raise ValueError("post-capacity assignments do not match expert input rows")


@dataclass(frozen=True)
class SwitchCollectionOutcome:
    """One bounded baseline-verified Switch trace-v2 collection."""

    trace: RoutingTraceV2
    source_token_ids: tuple[int, ...]
    decoder_prompt_token_ids: tuple[int, ...]
    fed_decoder_generated_token_ids: tuple[int, ...]
    candidate_token_ids: tuple[int, ...]
    terminal_eos_token_id: int | None
    horizon_exhausted: bool
    max_fed_back_non_eos_steps: int
    logits_exact: bool
    max_logit_abs_diff: float
    registered_sparse_layers: tuple[tuple[str, tuple[int, ...]], ...]
    router_output_shapes: tuple[tuple[tuple[int, ...], ...], ...]
    dispatch_checks: tuple[SwitchDispatchCheck, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.trace, RoutingTraceV2):
            raise TypeError("Switch outcome trace must be RoutingTraceV2")
        if not self.logits_exact or self.max_logit_abs_diff != 0.0:
            raise ValueError("Switch observation must be exactly non-interfering")
        if self.max_fed_back_non_eos_steps < 0:
            raise ValueError("maximum fed-back step count must be non-negative")
        if len(self.fed_decoder_generated_token_ids) > self.max_fed_back_non_eos_steps:
            raise ValueError("fed-back decoder tokens exceed the bounded horizon")


@dataclass(frozen=True)
class _GenerationRun:
    events: tuple[RoutingEventV2, ...]
    source_token_ids: tuple[int, ...]
    decoder_prompt_token_ids: tuple[int, ...]
    fed_decoder_generated_token_ids: tuple[int, ...]
    candidate_token_ids: tuple[int, ...]
    candidate_logits: tuple[torch.Tensor, ...]
    terminal_eos_token_id: int | None
    horizon_exhausted: bool
    generated_text: str
    router_output_shapes: tuple[tuple[tuple[int, ...], ...], ...]


class SwitchTraceCollector:
    """Pinned CPU-float32 collector for ``google/switch-base-8`` only."""

    def __init__(self, *, local_files_only: bool = False) -> None:
        import transformers

        if transformers.__version__ != "5.12.0":
            raise RuntimeError("Switch collection requires Transformers exactly 5.12.0")
        if torch.__version__.split("+", 1)[0] != "2.12.0":
            raise RuntimeError("Switch collection requires Torch exactly 2.12.0")

        offline = {"local_files_only": True} if local_files_only else {}
        config = AutoConfig.from_pretrained(
            DEFAULT_SWITCH_MODEL,
            revision=SWITCH_MODEL_REVISION,
            **offline,
        )
        resolved = getattr(config, "_commit_hash", None)
        if resolved != SWITCH_MODEL_REVISION:
            raise RuntimeError(
                "Switch config did not resolve to the required immutable revision"
            )
        if getattr(config, "model_type", None) != "switch_transformers":
            raise RuntimeError("pinned model did not resolve to SwitchTransformers")
        self.resolved_revision = resolved
        self.tokenizer = AutoTokenizer.from_pretrained(
            DEFAULT_SWITCH_MODEL,
            revision=resolved,
            **offline,
        )
        self.model = SwitchTransformersForConditionalGeneration.from_pretrained(
            DEFAULT_SWITCH_MODEL,
            revision=resolved,
            config=config,
            torch_dtype=torch.float32,
            **offline,
        )
        self.model.to(device="cpu", dtype=torch.float32)
        self.model.eval()
        _validate_cpu_float32_model(self.model)

        self.model_id = DEFAULT_SWITCH_MODEL
        self.num_experts = _positive_config_int(config, "num_experts")
        self._sparse_mlps = _discover_switch_sparse_mlps(self.model)
        self.sparse_layers = tuple(
            (stage, tuple(layer for found_stage, layer, _ in self._sparse_mlps if found_stage == stage))
            for stage in ("encoder", "decoder")
        )
        if any(not layers for _, layers in self.sparse_layers):
            raise RuntimeError("Switch collector requires routed encoder and decoder stages")

    def collect_and_verify(
        self,
        prompt: str,
        *,
        max_fed_back_non_eos_steps: int = 4,
    ) -> SwitchCollectionOutcome:
        """Collect one trace and prove hooks do not change greedy model outputs."""
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("Switch collection prompt must be a non-empty string")
        if (
            isinstance(max_fed_back_non_eos_steps, bool)
            or not isinstance(max_fed_back_non_eos_steps, int)
            or max_fed_back_non_eos_steps < 0
        ):
            raise ValueError("maximum fed-back step count must be non-negative")

        encoded = {
            name: value.detach().clone().to("cpu")
            for name, value in self.tokenizer(prompt, return_tensors="pt").items()
        }
        baseline = self._run_generation(
            encoded,
            max_fed_back_non_eos_steps,
            observe=False,
        )
        observed = self._run_generation(
            encoded,
            max_fed_back_non_eos_steps,
            observe=True,
        )
        max_abs_diff = _require_identical_runs(baseline, observed)
        dispatch_checks = _verify_initial_dispatch(
            self.model,
            self._sparse_mlps,
            encoded,
            _decoder_start_token_id(self.model.config),
        )

        import transformers

        trace = RoutingTraceV2(
            model_id=self.model_id,
            routing_stages=(
                RoutingStageProfile("encoder", self.num_experts, 1, True),
                RoutingStageProfile("decoder", self.num_experts, 1, True),
            ),
            events=observed.events,
            source_text=prompt,
            generated_text=observed.generated_text,
            capture_method=SWITCH_CAPTURE_METHOD,
            transformers_version=transformers.__version__,
            model_revision=self.resolved_revision,
        )
        return SwitchCollectionOutcome(
            trace=trace,
            source_token_ids=observed.source_token_ids,
            decoder_prompt_token_ids=observed.decoder_prompt_token_ids,
            fed_decoder_generated_token_ids=(
                observed.fed_decoder_generated_token_ids
            ),
            candidate_token_ids=observed.candidate_token_ids,
            terminal_eos_token_id=observed.terminal_eos_token_id,
            horizon_exhausted=observed.horizon_exhausted,
            max_fed_back_non_eos_steps=max_fed_back_non_eos_steps,
            logits_exact=True,
            max_logit_abs_diff=max_abs_diff,
            registered_sparse_layers=self.sparse_layers,
            router_output_shapes=observed.router_output_shapes,
            dispatch_checks=dispatch_checks,
        )

    def expert_parameter_payload_bytes(self) -> dict[RoutingExpertKeyV2, int]:
        """Return deterministic parameter payload bytes for every routed expert."""
        result: dict[RoutingExpertKeyV2, int] = {}
        for stage, layer, mlp in self._sparse_mlps:
            experts = getattr(mlp, "experts", None)
            if experts is None or not hasattr(experts, "items"):
                raise RuntimeError("Switch sparse MLP does not expose expert modules")
            for name, expert in experts.items():
                expert_id = _expert_index(name)
                size_bytes = sum(
                    parameter.numel() * parameter.element_size()
                    for parameter in expert.parameters()
                )
                if size_bytes <= 0:
                    raise RuntimeError("Switch expert has no parameter payload")
                key = (stage, layer, expert_id)
                if key in result:
                    raise RuntimeError("duplicate stage-qualified Switch expert")
                result[key] = size_bytes
        return dict(sorted(result.items()))

    def _run_generation(
        self,
        encoded: Mapping[str, torch.Tensor],
        max_steps: int,
        *,
        observe: bool,
    ) -> _GenerationRun:
        input_ids = encoded.get("input_ids")
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Switch collector supports one batch-size-one source")
        source_ids = tuple(int(value) for value in input_ids[0].tolist())
        decoder_start = _decoder_start_token_id(self.model.config)
        decoder_input = torch.tensor([[decoder_start]], dtype=torch.long)
        eos_ids = _normalize_eos_ids(self.model.config, self.tokenizer)
        events: list[RoutingEventV2] = []
        candidates: list[int] = []
        candidate_logits: list[torch.Tensor] = []
        fed_ids: list[int] = []
        output_shapes: list[tuple[tuple[int, ...], ...]] = []
        terminal_eos: int | None = None
        horizon_exhausted = False

        capture = _SwitchRouterCapture(self.model) if observe else None
        context = capture if capture is not None else _NullContext()
        with context, torch.inference_mode():
            encoder_outputs = self.model.encoder(
                **{name: value.detach().clone() for name, value in encoded.items()},
                use_cache=False,
                return_dict=True,
            )
            if capture is not None:
                observations = capture.take()
                events.extend(
                    _events_from_switch_observations(
                        observations,
                        {"encoder": input_ids},
                        {"encoder": "source"},
                        {"encoder": 0},
                        self.num_experts,
                        expected_callback_identity=tuple(
                            (stage, layer)
                            for stage, layer, _ in self._sparse_mlps
                            if stage == "encoder"
                        ),
                    )
                )
                output_shapes.extend(item.output_shapes for item in observations)
            initial = self.model(
                attention_mask=encoded.get("attention_mask"),
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input,
                use_cache=True,
                return_dict=True,
            )
            if capture is not None:
                observations = capture.take()
                events.extend(
                    _events_from_switch_observations(
                        observations,
                        {"decoder": decoder_input},
                        {"decoder": "decoder_prompt"},
                        {"decoder": 0},
                        self.num_experts,
                        expected_callback_identity=tuple(
                            (stage, layer)
                            for stage, layer, _ in self._sparse_mlps
                            if stage == "decoder"
                        ),
                    )
                )
                output_shapes.extend(item.output_shapes for item in observations)

            past = initial.past_key_values
            logits = initial.logits[:, -1, :].detach().float().cpu()
            candidate = int(logits.argmax(dim=-1)[0])
            candidates.append(candidate)
            candidate_logits.append(logits)

            while True:
                if candidate in eos_ids:
                    terminal_eos = candidate
                    break
                if len(fed_ids) >= max_steps:
                    horizon_exhausted = True
                    break
                fed_ids.append(candidate)
                decoder_token = torch.tensor([[candidate]], dtype=torch.long)
                subsequent = self.model(
                    attention_mask=encoded.get("attention_mask"),
                    encoder_outputs=encoder_outputs,
                    decoder_input_ids=decoder_token,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                if capture is not None:
                    observations = capture.take()
                    events.extend(
                        _events_from_switch_observations(
                            observations,
                            {"decoder": decoder_token},
                            {"decoder": "decoder_generated"},
                            {"decoder": len(fed_ids)},
                            self.num_experts,
                            expected_callback_identity=tuple(
                                (stage, layer)
                                for stage, layer, _ in self._sparse_mlps
                                if stage == "decoder"
                            ),
                        )
                    )
                    output_shapes.extend(item.output_shapes for item in observations)
                past = subsequent.past_key_values
                logits = subsequent.logits[:, -1, :].detach().float().cpu()
                candidate = int(logits.argmax(dim=-1)[0])
                candidates.append(candidate)
                candidate_logits.append(logits)

        generated_text = self.tokenizer.decode(fed_ids, skip_special_tokens=True)
        return _GenerationRun(
            events=tuple(events),
            source_token_ids=source_ids,
            decoder_prompt_token_ids=(decoder_start,),
            fed_decoder_generated_token_ids=tuple(fed_ids),
            candidate_token_ids=tuple(candidates),
            candidate_logits=tuple(candidate_logits),
            terminal_eos_token_id=terminal_eos,
            horizon_exhausted=horizon_exhausted,
            generated_text=generated_text,
            router_output_shapes=tuple(output_shapes),
        )


class _SwitchRouterCapture:
    """Temporary observer for native post-capacity Switch router outputs."""

    def __init__(self, model: Any) -> None:
        self._handles: list[Any] = []
        self._observations: list[SwitchRouterObservation] = []
        sparse_mlps = _discover_switch_sparse_mlps(model)
        try:
            for stage, layer, mlp in sparse_mlps:
                self._handles.append(
                    mlp.router.register_forward_hook(self._hook(stage, layer))
                )
        except BaseException as error:
            _remove_handles(self._handles, error)
            raise
        if not self._handles:
            raise RuntimeError("could not locate Switch sparse routers")

    def _hook(self, stage: str, layer: int):
        def hook(_: Any, __: tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, tuple) or len(output) != 3:
                raise RuntimeError("Switch router returned an unexpected output tuple")
            if any(not isinstance(value, torch.Tensor) for value in output):
                raise RuntimeError("Switch router output must contain three tensors")
            observation = SwitchRouterObservation(
                routing_stage=stage,
                layer=layer,
                output_shapes=tuple(tuple(value.shape) for value in output),
                post_capacity_assignment=output[1].detach().cpu(),
                routing_weight=output[2].detach().float().cpu(),
            )
            self._observations.append(observation)

        hook._moe_cache_lab_switch_router_hook = (stage, layer)  # type: ignore[attr-defined]
        return hook

    def take(self) -> tuple[SwitchRouterObservation, ...]:
        if not self._observations:
            raise RuntimeError("Switch router hooks captured no observations")
        result = tuple(self._observations)
        self._observations.clear()
        return result

    def __enter__(self) -> "_SwitchRouterCapture":
        return self

    def __exit__(self, *_: Any) -> None:
        try:
            _remove_handles(self._handles)
        finally:
            self._observations.clear()


class _NullContext:
    def __enter__(self) -> "_NullContext":
        return self

    def __exit__(self, *_: Any) -> None:
        return None


def _events_from_switch_observations(
    observations: tuple[SwitchRouterObservation, ...],
    token_ids_by_stage: Mapping[str, torch.Tensor],
    phase_by_stage: Mapping[str, str],
    position_offset_by_stage: Mapping[str, int],
    num_experts: int,
    *,
    expected_callback_identity: tuple[tuple[str, int], ...] | None = None,
) -> list[RoutingEventV2]:
    """Convert native callback order to v2 without sorting or repair."""
    identities = tuple((item.routing_stage, item.layer) for item in observations)
    if expected_callback_identity is not None and identities != expected_callback_identity:
        raise RuntimeError(
            "Switch router callback order/coverage did not match registered sparse layers"
        )
    events: list[RoutingEventV2] = []
    for observation in observations:
        stage = observation.routing_stage
        token_ids = token_ids_by_stage.get(stage)
        phase = phase_by_stage.get(stage)
        offset = position_offset_by_stage.get(stage)
        if token_ids is None or phase is None or offset is None:
            raise RuntimeError("Switch observation has no declared token/phase mapping")
        if token_ids.ndim != 2 or token_ids.shape[0] != 1:
            raise ValueError("Switch event conversion requires batch size one")
        mask = observation.post_capacity_assignment
        weights = observation.routing_weight
        token_count = token_ids.shape[1]
        if tuple(mask.shape) != (token_count, 1, num_experts):
            raise RuntimeError(
                "unexpected Switch post-capacity assignment shape: "
                f"{tuple(mask.shape)}"
            )
        if tuple(weights.shape) != (token_count, 1):
            raise RuntimeError(
                f"unexpected Switch routing-weight shape: {tuple(weights.shape)}"
            )
        if not bool(torch.all((mask == 0) | (mask == 1))):
            raise RuntimeError("Switch post-capacity assignment must be binary")
        if not bool(torch.isfinite(weights).all()) or not bool(
            ((weights >= 0.0) & (weights <= 1.0)).all()
        ):
            raise RuntimeError("Switch routing weights must be finite probabilities")

        for token_index in range(token_count):
            assigned = torch.nonzero(mask[token_index, 0], as_tuple=False).flatten()
            if assigned.numel() > 1:
                raise RuntimeError(
                    "Switch post-capacity assignment selected more than one expert"
                )
            common = {
                "routing_stage": stage,
                "phase": phase,
                "token_position": offset + token_index,
                "layer": observation.layer,
                "token_id": int(token_ids[0, token_index]),
            }
            if assigned.numel() == 0:
                events.append(
                    RoutingEventV2(
                        **common,
                        assignment_state="unassigned",
                        selected_experts=(),
                        selected_probabilities=(),
                        unassigned_reason="capacity",
                    )
                )
            else:
                events.append(
                    RoutingEventV2(
                        **common,
                        assignment_state="assigned",
                        selected_experts=(int(assigned[0]),),
                        selected_probabilities=(float(weights[token_index, 0]),),
                    )
                )
    if not events:
        raise RuntimeError("Switch observations produced no routing events")
    return events


def _discover_switch_sparse_mlps(model: Any) -> tuple[tuple[str, int, Any], ...]:
    discovered: list[tuple[str, int, Any]] = []
    seen_modules: set[int] = set()
    for stage in ("encoder", "decoder"):
        stack = getattr(model, stage, None)
        blocks = getattr(stack, "block", None)
        if blocks is None:
            raise RuntimeError(f"Switch model does not expose {stage}.block")
        for layer, block in enumerate(blocks):
            if not bool(getattr(block, "is_sparse", False)):
                continue
            layers = getattr(block, "layer", None)
            if layers is None or not layers:
                raise RuntimeError("Switch sparse block has no feed-forward layer")
            mlp = getattr(layers[-1], "mlp", None)
            if mlp is None or not hasattr(mlp, "router") or not hasattr(mlp, "experts"):
                raise RuntimeError("Switch sparse block does not expose router and experts")
            if id(mlp) in seen_modules:
                raise RuntimeError("Switch sparse MLP is reused across stage/layer identity")
            seen_modules.add(id(mlp))
            discovered.append((stage, layer, mlp))
    if not discovered:
        raise RuntimeError("Switch model exposes no sparse MLP layers")
    return tuple(discovered)


def _verify_initial_dispatch(
    model: Any,
    sparse_mlps: tuple[tuple[str, int, Any], ...],
    encoded: Mapping[str, torch.Tensor],
    decoder_start_token_id: int,
) -> tuple[SwitchDispatchCheck, ...]:
    rows: dict[tuple[str, int], int] = {
        (stage, layer): 0 for stage, layer, _ in sparse_mlps
    }
    handles: list[Any] = []
    capture = _SwitchRouterCapture(model)
    try:
        for stage, layer, mlp in sparse_mlps:
            for _, expert in mlp.experts.items():
                def pre_hook(
                    _: Any,
                    inputs: tuple[Any, ...],
                    key: tuple[str, int] = (stage, layer),
                ) -> None:
                    if not inputs or not isinstance(inputs[0], torch.Tensor):
                        raise RuntimeError("Switch expert hook received unexpected input")
                    rows[key] += int(inputs[0].shape[0])

                handles.append(expert.register_forward_pre_hook(pre_hook))
        with capture, torch.inference_mode():
            model(
                **{name: value.detach().clone() for name, value in encoded.items()},
                decoder_input_ids=torch.tensor([[decoder_start_token_id]], dtype=torch.long),
                use_cache=False,
                return_dict=True,
            )
            observations = capture.take()
    finally:
        _remove_handles(handles)

    expected = tuple((stage, layer) for stage, layer, _ in sparse_mlps)
    actual = tuple((item.routing_stage, item.layer) for item in observations)
    if actual != expected:
        raise RuntimeError("dispatch verifier observed unexpected router callback order")
    checks = []
    for observation in observations:
        mask = observation.post_capacity_assignment
        assigned = int(torch.count_nonzero(mask).item())
        checks.append(
            SwitchDispatchCheck(
                observation.routing_stage,
                observation.layer,
                assigned,
                rows[(observation.routing_stage, observation.layer)],
            )
        )
    return tuple(checks)


def _require_identical_runs(baseline: _GenerationRun, observed: _GenerationRun) -> float:
    semantic_fields = (
        "source_token_ids",
        "decoder_prompt_token_ids",
        "fed_decoder_generated_token_ids",
        "candidate_token_ids",
        "terminal_eos_token_id",
        "horizon_exhausted",
        "generated_text",
    )
    for field in semantic_fields:
        if getattr(baseline, field) != getattr(observed, field):
            raise RuntimeError(f"Switch router observation changed {field}")
    if len(baseline.candidate_logits) != len(observed.candidate_logits):
        raise RuntimeError("Switch router observation changed forward-call count")
    max_abs_diff = 0.0
    for expected, actual in zip(baseline.candidate_logits, observed.candidate_logits):
        if expected.shape != actual.shape:
            raise RuntimeError("Switch router observation changed logits shape")
        if not torch.equal(expected, actual):
            max_abs_diff = max(
                max_abs_diff,
                float(torch.max(torch.abs(expected - actual)).item()),
            )
    if max_abs_diff != 0.0:
        raise RuntimeError(
            "Switch router observation changed model logits "
            f"(max abs diff {max_abs_diff})"
        )
    return max_abs_diff


def _normalize_eos_ids(config: Any, tokenizer: Any) -> tuple[int, ...]:
    values: list[int] = []
    for owner in (config, tokenizer):
        raw = getattr(owner, "eos_token_id", None)
        if raw is None:
            continue
        candidates = raw if isinstance(raw, (list, tuple)) else (raw,)
        for candidate in candidates:
            if isinstance(candidate, bool) or not isinstance(candidate, int) or candidate < 0:
                raise ValueError("Switch EOS token IDs must be non-negative integers")
            values.append(candidate)
    normalized = tuple(sorted(set(values)))
    if not normalized:
        raise ValueError("Switch model/tokenizer exposes no EOS token ID")
    return normalized


def _decoder_start_token_id(config: Any) -> int:
    value = getattr(config, "decoder_start_token_id", None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Switch config exposes no valid decoder-start token ID")
    return value


def _positive_config_int(config: Any, name: str) -> int:
    value = getattr(config, name, None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"Switch config {name} must be a positive integer")
    return value


def _expert_index(name: str) -> int:
    if not isinstance(name, str) or not name.startswith("expert_"):
        raise RuntimeError(f"unexpected Switch expert module name: {name!r}")
    try:
        value = int(name.removeprefix("expert_"))
    except ValueError as error:
        raise RuntimeError(f"unexpected Switch expert module name: {name!r}") from error
    if value < 0:
        raise RuntimeError("Switch expert index must be non-negative")
    return value


def _validate_cpu_float32_model(model: Any) -> None:
    parameters = tuple(model.parameters())
    if not parameters:
        raise RuntimeError("Switch model has no parameters")
    if any(parameter.device.type != "cpu" for parameter in parameters):
        raise RuntimeError("Switch model must be entirely on CPU")
    if any(parameter.dtype != torch.float32 for parameter in parameters):
        raise RuntimeError("Switch model must be entirely float32")
    if any(parameter.is_meta for parameter in parameters):
        raise RuntimeError("Switch model cannot contain meta parameters")
    if model.training:
        raise RuntimeError("Switch model must be in evaluation mode")


def _remove_handles(handles: list[Any], original_error: BaseException | None = None) -> None:
    cleanup_errors: list[Exception] = []
    for handle in handles:
        try:
            handle.remove()
        except Exception as error:
            cleanup_errors.append(error)
    handles.clear()
    if cleanup_errors:
        message = "Switch hook cleanup failed: " + "; ".join(
            repr(error) for error in cleanup_errors
        )
        if original_error is not None:
            raise RuntimeError(message) from original_error
        raise RuntimeError(message)


__all__ = [
    "SWITCH_CAPTURE_METHOD",
    "SwitchCollectionOutcome",
    "SwitchDispatchCheck",
    "SwitchRouterObservation",
    "SwitchTraceCollector",
]
