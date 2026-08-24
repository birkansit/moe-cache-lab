"""Emit one deterministic synthetic canonical routing trace v2.

This standalone example deliberately uses only the Python standard library.
Validation and downstream interpretation belong to the installed moe-cache-lab
CLI rather than this producer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _records() -> tuple[dict[str, object], ...]:
    return (
        {
            "capture_method": "synthetic-external-event-source",
            "created_at": "2026-01-01T00:00:00+00:00",
            "format": "moe-cache-lab.routing-jsonl",
            "format_version": 2,
            "generated_text": None,
            "model_id": "synthetic/external-producer-no-model",
            "model_revision": "synthetic-contract-v1",
            "record_type": "metadata",
            "routing_stages": [
                {
                    "allows_unassigned": True,
                    "assigned_experts_per_token": 1,
                    "num_experts": 4,
                    "routing_stage": "encoder",
                },
                {
                    "allows_unassigned": True,
                    "assigned_experts_per_token": 1,
                    "num_experts": 4,
                    "routing_stage": "decoder",
                },
            ],
            "source_text": None,
            "transformers_version": None,
        },
        {
            "assignment_state": "assigned",
            "layer": 0,
            "phase": "source",
            "record_type": "routing_selection",
            "routing_stage": "encoder",
            "selected_experts": [1],
            "selected_probabilities": [0.8],
            "token_id": 10,
            "token_position": 0,
        },
        {
            "assignment_state": "unassigned",
            "layer": 0,
            "phase": "source",
            "record_type": "routing_selection",
            "routing_stage": "encoder",
            "selected_experts": [],
            "selected_probabilities": [],
            "token_id": 11,
            "token_position": 1,
            "unassigned_reason": "capacity",
        },
        {
            "assignment_state": "assigned",
            "layer": 0,
            "phase": "decoder_prompt",
            "record_type": "routing_selection",
            "routing_stage": "decoder",
            "selected_experts": [1],
            "selected_probabilities": [0.7],
            "token_id": 20,
            "token_position": 0,
        },
        {
            "assignment_state": "assigned",
            "layer": 0,
            "phase": "decoder_generated",
            "record_type": "routing_selection",
            "routing_stage": "decoder",
            "selected_experts": [2],
            "selected_probabilities": [0.6],
            "token_id": 21,
            "token_position": 1,
        },
        {
            "assignment_state": "assigned",
            "layer": 0,
            "phase": "decoder_generated",
            "record_type": "routing_selection",
            "routing_stage": "decoder",
            "selected_experts": [1],
            "selected_probabilities": [0.9],
            "token_id": 22,
            "token_position": 2,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit the deterministic synthetic external-producer trace."
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    payload = "".join(
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for record in _records()
    ).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(args.output)


if __name__ == "__main__":
    main()
