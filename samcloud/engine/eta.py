"""Progress parsing and ETA estimation for the SAMcloud processing pipeline."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# Seconds per unit measured on the Gruettistrasse test project
# (291 images à 20 MP, RTX 4090, 2026-11-03). Learned rates override these.
DEFAULT_RATES: dict[str, float] = {
    "feature_extraction": 0.4,     # per image
    "matching": 0.0077,            # per image pair
    "mapping": 1.9,                # per image
    "tiepoint_filtering": 0.64,    # per image
    "control_points": 0.4,         # per image (detection + control-point adjustment)
    "undistortion": 0.03,          # per image
    "dense_stereo": 15.4,          # per image and pass at 3200 px / 3 iterations
    "dense_fusion": 1.45,          # per image
    "classification": 4.6,         # per image
    "geometry_refinement": 0.47,   # per image
}

STAGE_LABELS: dict[str, str] = {
    "feature_extraction": "Merkmale extrahieren",
    "matching": "Bilder verknüpfen",
    "mapping": "Kameras ausrichten",
    "tiepoint_filtering": "Verknüpfungspunkte filtern",
    "control_points": "Passpunkte",
    "undistortion": "Bilder entzerren",
    "dense_stereo": "Tiefenkarten",
    "dense_fusion": "Dichte Punktwolke fusionieren",
    "classification": "SAM-Klassifizierung",
    "geometry_refinement": "Geometrische Verfeinerung",
}

_PATTERNS: dict[str, re.Pattern[str]] = {
    "feature_extraction": re.compile(r"Processed file \[(\d+)/(\d+)\]"),
    "matching_block2": re.compile(r"Processing block \[(\d+)/(\d+), (\d+)/(\d+)\]"),
    "matching_block1": re.compile(r"Processing block \[(\d+)/(\d+)\]"),
    "matching_image": re.compile(r"Processing image \[(\d+)/(\d+)\]"),
    "mapping": re.compile(r"Registering image #\d+ \(num_reg_frames=(\d+)\)"),
    "dense_stereo": re.compile(r"Processing view (\d+) / (\d+)"),
    "dense_fusion": re.compile(r"Fusing image \[(\d+)/(\d+)\]"),
    "classification": re.compile(r"images:\s+\d+%\|.*?\|\s*(\d+)/(\d+)"),
}


def stage_units(
    stage: str, *, images: int, max_image_size: int = 3200, num_iterations: int = 3,
    geometric_consistency: bool = True, matching: str = "exhaustive", spatial_neighbors: int = 50,
) -> float:
    """Work units of a stage, matching the unit of DEFAULT_RATES."""
    images = max(int(images), 1)
    if stage == "matching":
        if matching == "exhaustive":
            return images * (images - 1) / 2.0
        return images * min(spatial_neighbors, images - 1) / 2.0
    if stage == "dense_stereo":
        passes = 2 if geometric_consistency else 1
        return images * passes * (max_image_size / 3200.0) ** 2 * (num_iterations / 3.0)
    return float(images)


class RateStore:
    """Self-learning seconds-per-unit rates, persisted as JSON (exponential moving average)."""

    def __init__(self, path: Path | None, alpha: float = 0.4) -> None:
        self.path = path
        self.alpha = alpha
        self.rates: dict[str, float] = dict(DEFAULT_RATES)
        if path is not None and path.is_file():
            try:
                stored = json.loads(path.read_text(encoding="utf-8"))
                for key, value in stored.get("rates", {}).items():
                    if key in self.rates and float(value) > 0:
                        self.rates[key] = float(value)
            except (OSError, ValueError, TypeError):
                pass

    def rate(self, stage: str) -> float:
        return self.rates.get(stage, 1.0)

    def update(self, stage: str, seconds: float, units: float) -> None:
        if units <= 0 or seconds <= 0 or stage not in self.rates:
            return
        measured = seconds / units
        self.rates[stage] = (1 - self.alpha) * self.rates[stage] + self.alpha * measured

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"rates": self.rates}, indent=2), encoding="utf-8")


@dataclass
class _Stage:
    name: str
    units: float
    status: str = "pending"          # pending | running | done | skipped
    done_fraction: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None


@dataclass
class ProgressEstimator:
    """Track the planned stages of one run and estimate the remaining time."""

    plan: list[tuple[str, float]]
    rates: RateStore
    clock: Callable[[], float] = time.monotonic
    stages: list[_Stage] = field(init=False)
    _stereo_pass: int = field(init=False, default=0)
    _stereo_last_view: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.stages = [_Stage(name, float(units)) for name, units in self.plan]
        self._run_started = self.clock()

    def _find(self, name: str) -> _Stage | None:
        return next((stage for stage in self.stages if stage.name == name), None)

    @property
    def current(self) -> _Stage | None:
        return next((stage for stage in self.stages if stage.status == "running"), None)

    def start_stage(self, name: str, units: float | None = None) -> None:
        running = self.current
        if running is not None and running.name != name:
            self.finish_stage()
        stage = self._find(name)
        if stage is None:
            stage = _Stage(name, float(units or 1.0))
            self.stages.append(stage)
        if units is not None:
            stage.units = float(units)
        stage.status, stage.started_at, stage.done_fraction = "running", self.clock(), 0.0
        if name == "dense_stereo":
            self._stereo_pass, self._stereo_last_view = 0, 0

    def update(self, done: float, total: float) -> None:
        stage = self.current
        if stage is not None and total > 0:
            stage.done_fraction = max(stage.done_fraction, min(1.0, done / total))

    def finish_stage(self, *, learn: bool = True) -> None:
        stage = self.current
        if stage is None:
            return
        stage.status, stage.done_fraction, stage.finished_at = "done", 1.0, self.clock()
        if learn and stage.started_at is not None:
            self.rates.update(stage.name, stage.finished_at - stage.started_at, stage.units)

    def skip_stage(self, name: str) -> None:
        stage = self._find(name)
        if stage is not None and stage.status == "pending":
            stage.status = "skipped"

    def parse_line(self, line: str) -> bool:
        """Update the running stage from a COLMAP log line; returns True when the line was used."""
        stage = self.current
        if stage is None:
            return False
        if stage.name == "feature_extraction" and (match := _PATTERNS["feature_extraction"].search(line)):
            self.update(int(match.group(1)), int(match.group(2)))
            return True
        if stage.name == "matching":
            if match := _PATTERNS["matching_block2"].search(line):
                row, rows, col, cols = (int(value) for value in match.groups())
                self.update((row - 1) * cols + col, rows * cols)
                return True
            for key in ("matching_block1", "matching_image"):
                if match := _PATTERNS[key].search(line):
                    self.update(int(match.group(1)), int(match.group(2)))
                    return True
        if stage.name == "mapping" and (match := _PATTERNS["mapping"].search(line)):
            images = max(stage.units, 1.0)
            self.update(int(match.group(1)), images)
            return True
        if stage.name == "dense_stereo" and (match := _PATTERNS["dense_stereo"].search(line)):
            view, views = int(match.group(1)), int(match.group(2))
            if self._stereo_pass == 0 or view < self._stereo_last_view:
                self._stereo_pass += 1
            self._stereo_last_view = view
            passes = 2 if stage.units > 0 and self._stereo_pass <= 2 else self._stereo_pass
            self.update((self._stereo_pass - 1) * views + view - 1, max(passes, self._stereo_pass) * views)
            return True
        if stage.name == "dense_fusion" and (match := _PATTERNS["dense_fusion"].search(line)):
            self.update(int(match.group(1)), int(match.group(2)))
            return True
        if stage.name == "classification" and (match := _PATTERNS["classification"].search(line)):
            self.update(int(match.group(1)), int(match.group(2)))
            return True
        return False

    def _expected(self, stage: _Stage) -> float:
        return self.rates.rate(stage.name) * stage.units

    def _speed_factor(self) -> float:
        predicted = actual = 0.0
        for stage in self.stages:
            if stage.status == "done" and stage.started_at is not None and stage.finished_at is not None:
                expected = self._expected(stage)
                if expected >= 30.0:
                    predicted += expected
                    actual += stage.finished_at - stage.started_at
        if predicted <= 0:
            return 1.0
        return min(3.0, max(0.33, actual / predicted))

    def snapshot(self) -> dict:
        """Return a JSON-serialisable progress snapshot for the dashboard."""
        now = self.clock()
        factor = self._speed_factor()
        stage = self.current
        stage_eta = None
        if stage is not None:
            prior_total = self._expected(stage) * factor
            prior_remaining = prior_total * (1.0 - stage.done_fraction)
            elapsed = now - stage.started_at if stage.started_at is not None else 0.0
            if stage.done_fraction > 0.02 and elapsed > 10.0:
                live_remaining = elapsed / stage.done_fraction * (1.0 - stage.done_fraction)
                weight = min(1.0, stage.done_fraction * 4.0)
                stage_eta = weight * live_remaining + (1.0 - weight) * prior_remaining
            else:
                stage_eta = max(prior_remaining - elapsed, prior_total * 0.05)
        pending = sum(self._expected(item) * factor for item in self.stages if item.status == "pending")
        eta = (stage_eta or 0.0) + pending
        done_seconds = now - self._run_started
        total = done_seconds + eta
        return {
            "stage_key": stage.name if stage else None,
            "stage": STAGE_LABELS.get(stage.name, stage.name) if stage else "Fertig",
            "stage_percent": round(100.0 * stage.done_fraction, 1) if stage else 100.0,
            "stage_eta_seconds": round(stage_eta, 1) if stage_eta is not None else None,
            "eta_seconds": round(eta, 1),
            "percent": round(100.0 * done_seconds / total, 1) if total > 0 else 100.0,
            "speed_factor": round(factor, 3),
            "stages": [
                {
                    "key": item.name,
                    "label": STAGE_LABELS.get(item.name, item.name),
                    "status": item.status,
                    "percent": round(100.0 * item.done_fraction, 1),
                    "expected_seconds": round(self._expected(item) * factor, 1),
                }
                for item in self.stages
            ],
        }
