"""Strict versioned configuration for offline pre-flight analysis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .hardware_cost import HardwareTransferProfile
from .trace import ExpertKey

PREFLIGHT_CONFIG_FORMAT = "moe-cache-lab.preflight-config"
PREFLIGHT_CONFIG_VERSION = 1
_POLICY_ORDER = ("lru", "lfu")

_TOP_LEVEL_FIELDS = frozenset({
    "format",
    "format_version",
    "expert_sizes",
    "capacities_bytes",
    "policies",
    "hardware_profiles",
})
_EXPERT_SIZE_FIELDS = frozenset({"layer_id", "expert_id", "size_bytes"})
_HARDWARE_PROFILE_FIELDS = frozenset({
    "name",
    "h2d_payload_bandwidth_bytes_per_second",
    "setup_latency_ns_per_loaded_expert",
})


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
class PreflightConfig:
    """Immutable normalized caller-supplied assumptions for pre-flight analysis."""

    expert_sizes: tuple[ExpertSizeRecord, ...]
    capacities_bytes: tuple[int, ...]
    policies: tuple[str, ...]
    hardware_profiles: tuple[HardwareTransferProfile, ...]

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

    def expert_size_map(self) -> dict[ExpertKey, int]:
        """Return a fresh deterministic mapping for the byte-cache core API."""
        return {item.key: item.size_bytes for item in self.expert_sizes}


def read_preflight_config(path: str | Path) -> PreflightConfig:
    """Read and strictly validate one JSON pre-flight configuration."""
    try:
        with Path(path).open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid preflight config JSON at line {exc.lineno} column {exc.colno}"
        ) from exc
    return parse_preflight_config_data(payload)


def parse_preflight_config_data(payload: Any) -> PreflightConfig:
    """Validate JSON-compatible data and return canonical immutable assumptions."""
    if not isinstance(payload, Mapping):
        raise ValueError("preflight config must be a top-level JSON object")
    _require_exact_fields(payload, _TOP_LEVEL_FIELDS, "preflight config")
    if payload["format"] != PREFLIGHT_CONFIG_FORMAT:
        raise ValueError("unsupported preflight config format")
    format_version = payload["format_version"]
    if (
        isinstance(format_version, bool)
        or not isinstance(format_version, int)
        or format_version != PREFLIGHT_CONFIG_VERSION
    ):
        raise ValueError("unsupported preflight config format_version")

    expert_payload = _require_array(payload["expert_sizes"], "expert_sizes")
    capacities_payload = _require_array(
        payload["capacities_bytes"], "capacities_bytes"
    )
    policies_payload = _require_array(payload["policies"], "policies")
    profile_payload = _require_array(
        payload["hardware_profiles"], "hardware_profiles"
    )

    expert_sizes = tuple(
        _parse_expert_size_record(item, index)
        for index, item in enumerate(expert_payload)
    )
    profiles = tuple(
        _parse_hardware_profile(item, index)
        for index, item in enumerate(profile_payload)
    )
    return PreflightConfig(
        expert_sizes=expert_sizes,
        capacities_bytes=tuple(capacities_payload),
        policies=tuple(policies_payload),
        hardware_profiles=profiles,
    )


def preflight_config_data(config: PreflightConfig) -> dict[str, Any]:
    """Return canonical JSON-compatible config assumptions."""
    if not isinstance(config, PreflightConfig):
        raise TypeError("preflight_config_data requires a PreflightConfig")
    return {
        "format": PREFLIGHT_CONFIG_FORMAT,
        "format_version": PREFLIGHT_CONFIG_VERSION,
        "expert_sizes": [
            {
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
                "setup_latency_ns_per_loaded_expert": (
                    profile.setup_latency_ns_per_loaded_expert
                ),
            }
            for profile in config.hardware_profiles
        ],
    }


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


def _parse_hardware_profile(payload: Any, index: int) -> HardwareTransferProfile:
    if not isinstance(payload, Mapping):
        raise ValueError(f"hardware_profiles[{index}] must be a JSON object")
    _require_exact_fields(
        payload, _HARDWARE_PROFILE_FIELDS, f"hardware_profiles[{index}]"
    )
    return HardwareTransferProfile(
        name=payload["name"],
        h2d_payload_bandwidth_bytes_per_second=(
            payload["h2d_payload_bandwidth_bytes_per_second"]
        ),
        setup_latency_ns_per_loaded_expert=(
            payload["setup_latency_ns_per_loaded_expert"]
        ),
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
