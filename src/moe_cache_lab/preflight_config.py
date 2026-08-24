"""Strict versioned configuration for offline pre-flight analysis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .hardware_cost import (
    DEFAULT_TRANSFER_OPERATION_PLAN,
    HardwareTransferProfile,
    TransferOperationPlan,
)
from .trace import ExpertKey

PREFLIGHT_CONFIG_FORMAT = "moe-cache-lab.preflight-config"
PREFLIGHT_CONFIG_VERSION = 2
PREFLIGHT_CONFIG_LEGACY_VERSION = 1
PREFLIGHT_CONFIG_STAGE_QUALIFIED_VERSION = 3
StageQualifiedExpertKey = tuple[str, int, int]
_POLICY_ORDER = ("lru", "lfu")
_ROUTING_STAGE_ORDER = ("encoder", "decoder")

_TOP_LEVEL_FIELDS_V1 = frozenset({
    "format",
    "format_version",
    "expert_sizes",
    "capacities_bytes",
    "policies",
    "hardware_profiles",
})
_TOP_LEVEL_FIELDS_V2 = _TOP_LEVEL_FIELDS_V1 | {"transfer_operation_plans"}
_EXPERT_SIZE_FIELDS = frozenset({"layer_id", "expert_id", "size_bytes"})
_STAGE_QUALIFIED_EXPERT_SIZE_FIELDS = _EXPERT_SIZE_FIELDS | {"routing_stage"}
_HARDWARE_PROFILE_FIELDS_V1 = frozenset({
    "name",
    "h2d_payload_bandwidth_bytes_per_second",
    "setup_latency_ns_per_loaded_expert",
})
_HARDWARE_PROFILE_FIELDS_V2 = frozenset({
    "name",
    "h2d_payload_bandwidth_bytes_per_second",
    "setup_latency_ns_per_transfer_operation",
})
_TRANSFER_OPERATION_PLAN_FIELDS = frozenset({
    "name",
    "operations_per_logical_load",
})


def _normalize_shared_config_fields(
    capacities_bytes: tuple[int, ...],
    policies_input: tuple[str, ...],
    hardware_profiles: tuple[HardwareTransferProfile, ...],
    transfer_operation_plans: tuple[TransferOperationPlan, ...],
) -> tuple[
    tuple[int, ...],
    tuple[str, ...],
    tuple[HardwareTransferProfile, ...],
    tuple[TransferOperationPlan, ...],
]:
    """Validate and normalize the semantics shared by config v1, v2, and v3."""
    capacities = tuple(capacities_bytes)
    if not capacities:
        raise ValueError("preflight config requires at least one cache capacity")
    if any(
        isinstance(capacity, bool)
        or not isinstance(capacity, int)
        or capacity <= 0
        for capacity in capacities
    ):
        raise ValueError("cache capacities must be positive integers")

    policies = tuple(policies_input)
    if not policies:
        raise ValueError("preflight config requires at least one cache policy")
    unsupported = sorted(
        {str(policy) for policy in policies if policy not in _POLICY_ORDER}
    )
    if unsupported:
        raise ValueError(
            "unsupported preflight cache policy: " + ", ".join(unsupported)
        )

    profiles = tuple(hardware_profiles)
    if not profiles:
        raise ValueError("preflight config requires at least one hardware profile")
    if any(not isinstance(profile, HardwareTransferProfile) for profile in profiles):
        raise TypeError(
            "hardware_profiles must contain HardwareTransferProfile objects"
        )
    names = [profile.name for profile in profiles]
    duplicate_names = sorted(name for name in set(names) if names.count(name) > 1)
    if duplicate_names:
        raise ValueError(
            "hardware profile names must be unique within a preflight config: "
            + ", ".join(duplicate_names)
        )

    plans = tuple(transfer_operation_plans)
    if not plans:
        raise ValueError(
            "preflight config requires at least one transfer operation plan"
        )
    if any(not isinstance(plan, TransferOperationPlan) for plan in plans):
        raise TypeError(
            "transfer_operation_plans must contain TransferOperationPlan objects"
        )
    plan_names = [plan.name for plan in plans]
    duplicate_plan_names = sorted(
        name for name in set(plan_names) if plan_names.count(name) > 1
    )
    if duplicate_plan_names:
        raise ValueError(
            "transfer operation plan names must be unique within a preflight config: "
            + ", ".join(duplicate_plan_names)
        )

    selected_policies = set(policies)
    return (
        tuple(sorted(set(capacities))),
        tuple(policy for policy in _POLICY_ORDER if policy in selected_policies),
        tuple(sorted(profiles, key=lambda profile: profile.name)),
        tuple(sorted(plans, key=lambda plan: plan.name)),
    )


@dataclass(frozen=True, order=True)
class ExpertSizeRecord:
    """One caller-supplied byte size for a layer-qualified expert object."""

    layer_id: int
    expert_id: int
    size_bytes: int

    def __post_init__(self) -> None:
        for label, value in (
            ("layer_id", self.layer_id),
            ("expert_id", self.expert_id),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
        ):
            raise ValueError("size_bytes must be a positive integer")

    @property
    def key(self) -> ExpertKey:
        return self.layer_id, self.expert_id


@dataclass(frozen=True)
class StageQualifiedExpertSizeRecord:
    """One caller-supplied byte size for a stage-qualified expert object."""

    routing_stage: str
    layer_id: int
    expert_id: int
    size_bytes: int

    def __post_init__(self) -> None:
        if self.routing_stage not in _ROUTING_STAGE_ORDER:
            raise ValueError("routing_stage must be 'encoder' or 'decoder'")
        for label, value in (
            ("layer_id", self.layer_id),
            ("expert_id", self.expert_id),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
        ):
            raise ValueError("size_bytes must be a positive integer")

    @property
    def key(self) -> StageQualifiedExpertKey:
        return self.routing_stage, self.layer_id, self.expert_id


@dataclass(frozen=True)
class PreflightConfig:
    """Immutable normalized caller-supplied assumptions for pre-flight analysis."""

    expert_sizes: tuple[ExpertSizeRecord, ...]
    capacities_bytes: tuple[int, ...]
    policies: tuple[str, ...]
    hardware_profiles: tuple[HardwareTransferProfile, ...]
    transfer_operation_plans: tuple[TransferOperationPlan, ...] = (
        DEFAULT_TRANSFER_OPERATION_PLAN,
    )
    format_version: int = PREFLIGHT_CONFIG_LEGACY_VERSION

    def __post_init__(self) -> None:
        expert_sizes = tuple(self.expert_sizes)
        if not expert_sizes:
            raise ValueError("preflight config requires at least one expert size")
        if any(not isinstance(item, ExpertSizeRecord) for item in expert_sizes):
            raise TypeError("expert_sizes must contain ExpertSizeRecord objects")
        identities = [item.key for item in expert_sizes]
        duplicate_identities = sorted(
            identity for identity in set(identities) if identities.count(identity) > 1
        )
        if duplicate_identities:
            raise ValueError(
                "duplicate expert-size identity in preflight config: "
                + ", ".join(str(identity) for identity in duplicate_identities)
            )

        capacities = tuple(self.capacities_bytes)
        if not capacities:
            raise ValueError("preflight config requires at least one cache capacity")
        if any(
            isinstance(capacity, bool)
            or not isinstance(capacity, int)
            or capacity <= 0
            for capacity in capacities
        ):
            raise ValueError("cache capacities must be positive integers")

        policies = tuple(self.policies)
        if not policies:
            raise ValueError("preflight config requires at least one cache policy")
        unsupported = sorted(
            {str(policy) for policy in policies if policy not in _POLICY_ORDER}
        )
        if unsupported:
            raise ValueError(
                "unsupported preflight cache policy: " + ", ".join(unsupported)
            )

        profiles = tuple(self.hardware_profiles)
        if not profiles:
            raise ValueError("preflight config requires at least one hardware profile")
        if any(not isinstance(profile, HardwareTransferProfile) for profile in profiles):
            raise TypeError(
                "hardware_profiles must contain HardwareTransferProfile objects"
            )
        names = [profile.name for profile in profiles]
        duplicate_names = sorted(
            name for name in set(names) if names.count(name) > 1
        )
        if duplicate_names:
            raise ValueError(
                "hardware profile names must be unique within a preflight config: "
                + ", ".join(duplicate_names)
            )

        plans = tuple(self.transfer_operation_plans)
        if not plans:
            raise ValueError(
                "preflight config requires at least one transfer operation plan"
            )
        if any(not isinstance(plan, TransferOperationPlan) for plan in plans):
            raise TypeError(
                "transfer_operation_plans must contain TransferOperationPlan objects"
            )
        plan_names = [plan.name for plan in plans]
        duplicate_plan_names = sorted(
            name for name in set(plan_names) if plan_names.count(name) > 1
        )
        if duplicate_plan_names:
            raise ValueError(
                "transfer operation plan names must be unique within a preflight "
                "config: " + ", ".join(duplicate_plan_names)
            )

        if (
            isinstance(self.format_version, bool)
            or not isinstance(self.format_version, int)
            or self.format_version not in (
                PREFLIGHT_CONFIG_LEGACY_VERSION,
                PREFLIGHT_CONFIG_VERSION,
            )
        ):
            raise ValueError("unsupported preflight config format_version")
        if (
            self.format_version == PREFLIGHT_CONFIG_LEGACY_VERSION
            and plans != (DEFAULT_TRANSFER_OPERATION_PLAN,)
        ):
            raise ValueError(
                "preflight config format_version 1 supports only the default "
                "one-operation-per-logical-load plan"
            )

        object.__setattr__(self, "expert_sizes", tuple(sorted(expert_sizes)))
        object.__setattr__(self, "capacities_bytes", tuple(sorted(set(capacities))))
        selected_policies = set(policies)
        object.__setattr__(
            self,
            "policies",
            tuple(policy for policy in _POLICY_ORDER if policy in selected_policies),
        )
        object.__setattr__(
            self,
            "hardware_profiles",
            tuple(sorted(profiles, key=lambda profile: profile.name)),
        )
        object.__setattr__(
            self,
            "transfer_operation_plans",
            tuple(sorted(plans, key=lambda plan: plan.name)),
        )

    def expert_size_map(self) -> dict[ExpertKey, int]:
        """Return a fresh deterministic mapping for the byte-cache core API."""
        return {item.key: item.size_bytes for item in self.expert_sizes}


@dataclass(frozen=True)
class PreflightConfigV3:
    """Immutable stage-qualified assumptions, without enabling execution."""

    expert_sizes: tuple[StageQualifiedExpertSizeRecord, ...]
    capacities_bytes: tuple[int, ...]
    policies: tuple[str, ...]
    hardware_profiles: tuple[HardwareTransferProfile, ...]
    transfer_operation_plans: tuple[TransferOperationPlan, ...]
    format_version: int = PREFLIGHT_CONFIG_STAGE_QUALIFIED_VERSION

    def __post_init__(self) -> None:
        expert_sizes = tuple(self.expert_sizes)
        if any(
            not isinstance(item, StageQualifiedExpertSizeRecord)
            for item in expert_sizes
        ):
            raise TypeError(
                "expert_sizes must contain StageQualifiedExpertSizeRecord objects"
            )
        identities = [item.key for item in expert_sizes]
        duplicate_identities = sorted(
            identity for identity in set(identities) if identities.count(identity) > 1
        )
        if duplicate_identities:
            raise ValueError(
                "duplicate expert-size identity in preflight config: "
                + ", ".join(str(identity) for identity in duplicate_identities)
            )

        capacities, policies, profiles, plans = _normalize_shared_config_fields(
            self.capacities_bytes,
            self.policies,
            self.hardware_profiles,
            self.transfer_operation_plans,
        )

        if (
            isinstance(self.format_version, bool)
            or not isinstance(self.format_version, int)
            or self.format_version != PREFLIGHT_CONFIG_STAGE_QUALIFIED_VERSION
        ):
            raise ValueError("unsupported preflight config format_version")

        stage_order = {
            stage: index for index, stage in enumerate(_ROUTING_STAGE_ORDER)
        }
        object.__setattr__(
            self,
            "expert_sizes",
            tuple(sorted(
                expert_sizes,
                key=lambda item: (
                    stage_order[item.routing_stage],
                    item.layer_id,
                    item.expert_id,
                ),
            )),
        )
        object.__setattr__(self, "capacities_bytes", capacities)
        object.__setattr__(self, "policies", policies)
        object.__setattr__(self, "hardware_profiles", profiles)
        object.__setattr__(self, "transfer_operation_plans", plans)

    def expert_size_map(self) -> dict[StageQualifiedExpertKey, int]:
        """Return a fresh stage-qualified mapping without flattening identities."""
        return {item.key: item.size_bytes for item in self.expert_sizes}


VersionedPreflightConfig = PreflightConfig | PreflightConfigV3


def read_preflight_config(path: str | Path) -> VersionedPreflightConfig:
    """Read and strictly validate one JSON pre-flight configuration."""
    try:
        with Path(path).open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid preflight config JSON at line {exc.lineno} column {exc.colno}"
        ) from exc
    return parse_preflight_config_data(payload)


def parse_preflight_config_data(payload: Any) -> VersionedPreflightConfig:
    """Validate JSON-compatible data and return canonical immutable assumptions."""
    if not isinstance(payload, Mapping):
        raise ValueError("preflight config must be a top-level JSON object")
    if "format" not in payload or "format_version" not in payload:
        _require_exact_fields(payload, _TOP_LEVEL_FIELDS_V1, "preflight config")
    if payload["format"] != PREFLIGHT_CONFIG_FORMAT:
        raise ValueError("unsupported preflight config format")
    format_version = payload["format_version"]
    if (
        isinstance(format_version, bool)
        or not isinstance(format_version, int)
        or format_version not in (
            PREFLIGHT_CONFIG_LEGACY_VERSION,
            PREFLIGHT_CONFIG_VERSION,
            PREFLIGHT_CONFIG_STAGE_QUALIFIED_VERSION,
        )
    ):
        raise ValueError("unsupported preflight config format_version")
    expected_fields = (
        _TOP_LEVEL_FIELDS_V1
        if format_version == PREFLIGHT_CONFIG_LEGACY_VERSION
        else _TOP_LEVEL_FIELDS_V2
    )
    _require_exact_fields(payload, expected_fields, "preflight config")

    expert_payload = _require_array(payload["expert_sizes"], "expert_sizes")
    capacities_payload = _require_array(
        payload["capacities_bytes"], "capacities_bytes"
    )
    policies_payload = _require_array(payload["policies"], "policies")
    profile_payload = _require_array(
        payload["hardware_profiles"], "hardware_profiles"
    )

    expert_sizes = tuple(
        (
            _parse_stage_qualified_expert_size_record(item, index)
            if format_version == PREFLIGHT_CONFIG_STAGE_QUALIFIED_VERSION
            else _parse_expert_size_record(item, index)
        )
        for index, item in enumerate(expert_payload)
    )
    profiles = tuple(
        _parse_hardware_profile(item, index, format_version)
        for index, item in enumerate(profile_payload)
    )
    plans = (
        (DEFAULT_TRANSFER_OPERATION_PLAN,)
        if format_version == PREFLIGHT_CONFIG_LEGACY_VERSION
        else tuple(
            _parse_transfer_operation_plan(item, index)
            for index, item in enumerate(
                _require_array(
                    payload["transfer_operation_plans"],
                    "transfer_operation_plans",
                )
            )
        )
    )
    config_type = (
        PreflightConfigV3
        if format_version == PREFLIGHT_CONFIG_STAGE_QUALIFIED_VERSION
        else PreflightConfig
    )
    return config_type(
        expert_sizes=expert_sizes,
        capacities_bytes=tuple(capacities_payload),
        policies=tuple(policies_payload),
        hardware_profiles=profiles,
        transfer_operation_plans=plans,
        format_version=format_version,
    )


def preflight_config_data(config: VersionedPreflightConfig) -> dict[str, Any]:
    """Return canonical JSON-compatible config assumptions."""
    if not isinstance(config, (PreflightConfig, PreflightConfigV3)):
        raise TypeError("preflight_config_data requires a PreflightConfig")
    stage_qualified = isinstance(config, PreflightConfigV3)
    payload: dict[str, Any] = {
        "format": PREFLIGHT_CONFIG_FORMAT,
        "format_version": config.format_version,
        "expert_sizes": [
            {
                **(
                    {"routing_stage": item.routing_stage}
                    if stage_qualified
                    else {}
                ),
                "layer_id": item.layer_id,
                "expert_id": item.expert_id,
                "size_bytes": item.size_bytes,
            }
            for item in config.expert_sizes
        ],
        "capacities_bytes": list(config.capacities_bytes),
        "policies": list(config.policies),
        "hardware_profiles": [
            {
                "name": profile.name,
                "h2d_payload_bandwidth_bytes_per_second": (
                    profile.h2d_payload_bandwidth_bytes_per_second
                ),
                (
                    "setup_latency_ns_per_loaded_expert"
                    if config.format_version == PREFLIGHT_CONFIG_LEGACY_VERSION
                    else "setup_latency_ns_per_transfer_operation"
                ): profile.setup_latency_ns_per_transfer_operation,
            }
            for profile in config.hardware_profiles
        ],
    }
    if config.format_version in (
        PREFLIGHT_CONFIG_VERSION,
        PREFLIGHT_CONFIG_STAGE_QUALIFIED_VERSION,
    ):
        payload["transfer_operation_plans"] = [
            {
                "name": plan.name,
                "operations_per_logical_load": plan.operations_per_logical_load,
            }
            for plan in config.transfer_operation_plans
        ]
    return payload


def _parse_expert_size_record(payload: Any, index: int) -> ExpertSizeRecord:
    if not isinstance(payload, Mapping):
        raise ValueError(f"expert_sizes[{index}] must be a JSON object")
    _require_exact_fields(
        payload, _EXPERT_SIZE_FIELDS, f"expert_sizes[{index}]"
    )
    return ExpertSizeRecord(
        layer_id=payload["layer_id"],
        expert_id=payload["expert_id"],
        size_bytes=payload["size_bytes"],
    )


def _parse_stage_qualified_expert_size_record(
    payload: Any,
    index: int,
) -> StageQualifiedExpertSizeRecord:
    if not isinstance(payload, Mapping):
        raise ValueError(f"expert_sizes[{index}] must be a JSON object")
    _require_exact_fields(
        payload,
        _STAGE_QUALIFIED_EXPERT_SIZE_FIELDS,
        f"expert_sizes[{index}]",
    )
    return StageQualifiedExpertSizeRecord(
        routing_stage=payload["routing_stage"],
        layer_id=payload["layer_id"],
        expert_id=payload["expert_id"],
        size_bytes=payload["size_bytes"],
    )


def _parse_hardware_profile(
    payload: Any,
    index: int,
    format_version: int,
) -> HardwareTransferProfile:
    if not isinstance(payload, Mapping):
        raise ValueError(f"hardware_profiles[{index}] must be a JSON object")
    _require_exact_fields(
        payload,
        (
            _HARDWARE_PROFILE_FIELDS_V1
            if format_version == PREFLIGHT_CONFIG_LEGACY_VERSION
            else _HARDWARE_PROFILE_FIELDS_V2
        ),
        f"hardware_profiles[{index}]",
    )
    return HardwareTransferProfile(
        name=payload["name"],
        h2d_payload_bandwidth_bytes_per_second=(
            payload["h2d_payload_bandwidth_bytes_per_second"]
        ),
        setup_latency_ns_per_loaded_expert=(
            payload[
                "setup_latency_ns_per_loaded_expert"
                if format_version == PREFLIGHT_CONFIG_LEGACY_VERSION
                else "setup_latency_ns_per_transfer_operation"
            ]
        ),
    )


def _parse_transfer_operation_plan(
    payload: Any,
    index: int,
) -> TransferOperationPlan:
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"transfer_operation_plans[{index}] must be a JSON object"
        )
    _require_exact_fields(
        payload,
        _TRANSFER_OPERATION_PLAN_FIELDS,
        f"transfer_operation_plans[{index}]",
    )
    return TransferOperationPlan(
        name=payload["name"],
        operations_per_logical_load=payload["operations_per_logical_load"],
    )


def _require_array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _require_exact_fields(
    payload: Mapping[str, Any],
    expected: frozenset[str],
    label: str,
) -> None:
    keys = set(payload)
    missing = sorted(expected.difference(keys))
    unknown = sorted(keys.difference(expected))
    if missing:
        raise ValueError(f"{label} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {', '.join(unknown)}")
