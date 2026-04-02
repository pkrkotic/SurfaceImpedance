from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def build_output_rows(
    freq_hz: np.ndarray,
    impedance: np.ndarray,
    metadata: dict[str, str | float] | None = None,
) -> list[dict[str, float | str]]:
    metadata = metadata or {}
    return [
        {
            **metadata,
            "frequency_hz": float(freq),
            "real_ohm": float(value.real),
            "imag_ohm": float(value.imag),
            "magnitude_ohm": float(abs(value)),
            "phase_deg": float(np.degrees(np.angle(value))),
        }
        for freq, value in zip(freq_hz, impedance, strict=True)
    ]


def export_data(path: str | Path, rows: list[dict[str, float | str]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.suffix.lower() == ".csv":
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return target

    if target.suffix.lower() == ".json":
        with target.open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
        return target

    raise ValueError("Export path must end with .csv or .json")
