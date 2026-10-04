"""Leakage-safe implementation of the paper's 20 feature types.

The original repository creates these features in ``build_advanced_features``:

* 12 aggregate statistics from a level-3 Daubechies-1 decomposition;
* rolling mean and rolling standard deviation;
* spatial/peer consensus;
* the corrected sensor value and hour;
* one rolling correlation for each of the three peer sensors.

This version keeps exactly those 20 columns, but computes them causally for
each timestamp.  A test row is allowed to use the preceding train history,
never future test rows and never ``Ref``.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

try:  # PyWavelets is optional; the db1 fallback below removes the hard dependency.
    import pywt
except Exception:  # pragma: no cover - exercised only on minimal installs
    pywt = None


PAPER20_FEATURE_NAMES = [
    name
    for level in range(4)
    for name in (
        f"wave_mean_{level}",
        f"wave_std_{level}",
        f"wave_max_{level}",
    )
] + [
    "roll_mean",
    "roll_std",
    "spatial_consensus",
    "sensor_val",
    "hour",
    "corr_peer_1",
    "corr_peer_2",
    "corr_peer_3",
]


PAPER20_FEATURE_DESCRIPTIONS = {
    **{
        f"wave_{stat}_{level}": f"db1 level-3 coefficient {level}: {stat}"
        for level in range(4)
        for stat in ("mean", "std", "max")
    },
    "roll_mean": "24-sample causal rolling mean of corrected sensor",
    "roll_std": "24-sample causal rolling standard deviation of corrected sensor",
    "spatial_consensus": "trust-weighted mean of the three corrected peer sensors",
    "sensor_val": "current corrected target sensor value",
    "hour": "hour of day extracted from Date",
    "corr_peer_1": "24-sample causal rolling correlation with peer 1",
    "corr_peer_2": "24-sample causal rolling correlation with peer 2",
    "corr_peer_3": "24-sample causal rolling correlation with peer 3",
}


def _numeric_history(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    return values.ffill().fillna(0.0).to_numpy(dtype=float)


def _haar_coefficients(values: np.ndarray, level: int = 3) -> list[np.ndarray]:
    """Return db1-equivalent [cA3, cD3, cD2, cD1] coefficients."""

    current = np.asarray(values, dtype=float)
    details: list[np.ndarray] = []
    for _ in range(int(level)):
        if len(current) < 2:
            current = np.pad(current, (0, 1), mode="edge")
        if len(current) % 2:
            current = np.pad(current, (0, 1), mode="edge")
        left = current[0::2]
        right = current[1::2]
        current, detail = (
            (left + right) / np.sqrt(2.0),
            (left - right) / np.sqrt(2.0),
        )
        details.insert(0, detail)
    return [current, *details]


def _wavelet_coefficients(values: np.ndarray, level: int = 3) -> list[np.ndarray]:
    # Haar is Daubechies-1 (db1).  Use the self-contained implementation so
    # compiled PyWavelets buffer handling cannot fail on newer environments.
    values = np.array(values, dtype=float, copy=True, order="C")
    minimum_length = 2 ** int(level)
    if len(values) < minimum_length:
        values = np.pad(values, (0, minimum_length - len(values)), mode="edge")
    return _haar_coefficients(values, level=level)


def _wavelet_stats(values: np.ndarray, level: int = 3) -> list[float]:
    output: list[float] = []
    for coeff in _wavelet_coefficients(values, level=level):
        output.extend(
            [
                float(np.mean(np.abs(coeff))),
                float(np.std(coeff)),
                float(np.max(np.abs(coeff))),
            ]
        )
    return output


def _rolling_std(values: np.ndarray) -> np.ndarray:
    result = np.zeros(len(values), dtype=float)
    for index in range(len(values)):
        segment = values[max(0, index - 23) : index + 1]
        result[index] = float(np.std(segment, ddof=1)) if len(segment) > 1 else 0.0
    return result


class Paper20FeatureBuilder:
    """Build exactly 20 paper-style features without future information."""

    feature_names = tuple(PAPER20_FEATURE_NAMES)

    def __init__(self, window_size: int = 24, wavelet_level: int = 3):
        self.window_size = int(window_size)
        self.wavelet_level = int(wavelet_level)
        if self.window_size < 2:
            raise ValueError("window_size must be at least 2")
        if self.wavelet_level != 3:
            raise ValueError("The paper feature definition uses level=3")

    @staticmethod
    def _validate(data: pd.DataFrame, sensors: Iterable[str]) -> None:
        missing = [s for s in sensors if s not in data.columns]
        if missing:
            raise KeyError(f"Missing sensor columns: {missing}")

    def _wavelet_frame(self, values: np.ndarray) -> np.ndarray:
        rows = []
        for index in range(len(values)):
            start = max(0, index - self.window_size + 1)
            rows.append(
                _wavelet_stats(
                    values[start : index + 1],
                    level=self.wavelet_level,
                )
            )
        return np.asarray(rows, dtype=float)

    def build(
        self,
        data: pd.DataFrame,
        sensor: str,
        sensors: list[str],
        trust_scores: dict[str, float] | None = None,
        history: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Build 20 features for ``data`` using optional preceding history."""

        self._validate(data, sensors)
        current = data.loc[:, [c for c in ["Date", *sensors] if c in data.columns]].copy()
        current = current.reset_index(drop=True)
        if history is not None and len(history):
            previous = history.loc[:, [c for c in ["Date", *sensors] if c in history.columns]].copy()
            previous = previous.tail(self.window_size - 1).reset_index(drop=True)
            combined = pd.concat([previous, current], ignore_index=True)
            n_history = len(previous)
        else:
            combined = current
            n_history = 0

        numeric = {s: _numeric_history(combined[s]) for s in sensors}
        target = numeric[sensor]
        peers = [s for s in sensors if s != sensor]
        if len(peers) != 3:
            raise ValueError("Paper20FeatureBuilder expects exactly three peer sensors")

        if trust_scores:
            weights = np.asarray(
                [max(float(trust_scores.get(peer, 1.0)), 1e-8) for peer in peers],
                dtype=float,
            )
            weights /= np.sum(weights)
        else:
            weights = np.ones(3, dtype=float) / 3.0
        peer_matrix = np.column_stack([numeric[peer] for peer in peers])
        consensus = peer_matrix @ weights

        if "Date" in combined.columns:
            date = pd.to_datetime(combined["Date"], errors="coerce")
            hour = date.dt.hour.fillna(0).to_numpy(dtype=float)
        else:
            hour = np.zeros(len(combined), dtype=float)

        wavelet = self._wavelet_frame(target)
        output = {}
        for level in range(4):
            output[f"wave_mean_{level}"] = wavelet[:, level * 3]
            output[f"wave_std_{level}"] = wavelet[:, level * 3 + 1]
            output[f"wave_max_{level}"] = wavelet[:, level * 3 + 2]
        output["roll_mean"] = pd.Series(target).rolling(
            self.window_size, min_periods=1
        ).mean().to_numpy(dtype=float)
        output["roll_std"] = _rolling_std(target)
        output["spatial_consensus"] = consensus
        output["sensor_val"] = target
        output["hour"] = hour

        for index, peer in enumerate(peers, start=1):
            corr = pd.Series(target).rolling(
                self.window_size,
                min_periods=self.window_size,
            ).corr(pd.Series(numeric[peer])).fillna(0.0)
            output[f"corr_peer_{index}"] = corr.to_numpy(dtype=float)

        frame = pd.DataFrame(output).loc[:, list(self.feature_names)]
        frame = frame.replace([np.inf, -np.inf], np.nan)
        if n_history:
            frame = frame.iloc[n_history:]
        return frame.reset_index(drop=True).astype(float)


__all__ = [
    "PAPER20_FEATURE_DESCRIPTIONS",
    "PAPER20_FEATURE_NAMES",
    "Paper20FeatureBuilder",
]
