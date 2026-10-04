"""30 controlled VQC-Hybrid cases: ten depths by three Z/ZZ readouts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.quantum_models import QuantumSpec, _expect_z, _expect_zz, _simulate_state


READOUT_COMBINATIONS = {
    "Z": ("Z",),
    "ZZ": ("ZZ",),
    "Z_ZZ": ("Z", "ZZ"),
}


@dataclass(frozen=True)
class LayerConfig:
    name: str
    depth: int
    blend: float = 0.30


@dataclass(frozen=True)
class LayerReadoutCase:
    name: str
    layer: int
    readout: str
    description: str


def _clean(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    if result.ndim != 2:
        raise ValueError("X must be a 2-D matrix")
    result[~np.isfinite(result)] = np.nan
    return result


class Linear32Lift:
    """Fixed and auditable Linear(20,32) lift."""

    def __init__(self):
        self.weight_ = np.zeros((32, 20), dtype=float)
        self.bias_ = np.zeros(32, dtype=float)
        self.weight_[:20, :20] = np.eye(20, dtype=float)
        for index in range(12):
            self.weight_[20 + index, index] = 0.5
            self.weight_[20 + index, index + 1] = 0.5

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) @ self.weight_.T + self.bias_


class VQCLayerFeatureMap:
    """Trainable VQC state map exposing only Z and nearest-neighbour ZZ."""

    def __init__(self, layer: int, spsa_steps: int = 6, max_fit_samples: int = 160, random_state: int = 42):
        self.layer = int(layer)
        if not 1 <= self.layer <= 10:
            raise ValueError("layer must be in the range 1..10")
        self.spsa_steps = max(0, int(spsa_steps))
        self.max_fit_samples = max(1, int(max_fit_samples))
        self.random_state = int(random_state)
        self.spec = QuantumSpec("angle", self.layer, "local", "chain", "z", True)
        self.imputer_ = SimpleImputer(strategy="mean")
        self.scaler_ = StandardScaler()
        self.linear32_ = Linear32Lift()

    @property
    def theta_size(self) -> int:
        return 2 * self.layer * 5

    @staticmethod
    def _angles(hidden: np.ndarray) -> np.ndarray:
        groups = np.array_split(np.arange(32), 5)
        return np.column_stack([np.mean(hidden[:, group], axis=1) for group in groups])

    @staticmethod
    def _blocks_for_state(state: np.ndarray) -> dict[str, np.ndarray]:
        # Compute only the two requested observable families.  In particular,
        # do not call the generic readout helper because it also evaluates X,
        # Y, and parity observables internally.
        z = np.asarray([_expect_z(state, q) for q in range(5)], dtype=float)
        zz = np.asarray([_expect_zz(state, q, q + 1) for q in range(4)], dtype=float)
        return {"Z": z, "ZZ": zz}

    def _blocks(self, angles: np.ndarray, theta: np.ndarray) -> dict[str, np.ndarray]:
        rows = {"Z": [], "ZZ": []}
        for row in np.asarray(angles, dtype=float):
            blocks = self._blocks_for_state(_simulate_state(row, self.spec, theta))
            for name in rows:
                rows[name].append(blocks[name])
        return {name: np.asarray(values, dtype=float) for name, values in rows.items()}

    def _loss(self, theta: np.ndarray, scaled: np.ndarray, angles: np.ndarray, residual: np.ndarray) -> float:
        blocks = self._blocks(angles, theta)
        quantum = np.concatenate([blocks["Z"], blocks["ZZ"]], axis=1)
        design = np.concatenate([scaled, quantum], axis=1)
        readout = make_pipeline(StandardScaler(), Ridge(alpha=12.0))
        readout.fit(design, residual)
        prediction = readout.predict(design)
        scale = max(float(np.std(residual)), 1e-8)
        return float(np.mean(((prediction - residual) / scale) ** 2))

    def fit(self, X: np.ndarray, residual: np.ndarray) -> "VQCLayerFeatureMap":
        values = _clean(X)
        if values.shape[1] != 20:
            raise ValueError(f"Expected exactly 20 paper features, got {values.shape[1]}")
        target = np.asarray(residual, dtype=float).reshape(-1)
        if len(values) != len(target) or not np.isfinite(target).all():
            raise ValueError("X and residual must contain equal finite rows")
        scaled = self.scaler_.fit_transform(self.imputer_.fit_transform(values))
        angles = self._angles(self.linear32_.transform(scaled))
        rng = np.random.default_rng(self.random_state + 11)
        theta = rng.uniform(-0.08, 0.08, size=self.theta_size).astype(float)
        losses = []
        if self.spsa_steps:
            count = min(len(values), self.max_fit_samples)
            indexes = np.linspace(0, len(values) - 1, count, dtype=int)
            direction_rng = np.random.default_rng(self.random_state + 97)
            for step in range(self.spsa_steps):
                direction = direction_rng.choice(np.asarray([-1.0, 1.0]), size=self.theta_size)
                ck = 0.08 / ((step + 1.0) ** 0.101)
                ak = 0.06 / ((step + 1.0) ** 0.602)
                plus = self._loss(theta + ck * direction, scaled[indexes], angles[indexes], target[indexes])
                minus = self._loss(theta - ck * direction, scaled[indexes], angles[indexes], target[indexes])
                theta = np.clip(theta - ak * ((plus - minus) / (2.0 * ck)) * direction, -np.pi, np.pi)
                losses.append(float(min(plus, minus)))
        self.theta_ = theta
        self.spsa_losses_ = losses
        self.train_scaled_ = scaled
        self.train_blocks_ = self._blocks(angles, theta)
        self.training_objective_ = self._loss(theta, scaled, angles, target)
        return self

    def transform(self, X: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if not hasattr(self, "theta_"):
            raise RuntimeError("Call fit before transform")
        values = _clean(X)
        if values.shape[1] != 20:
            raise ValueError(f"Expected exactly 20 paper features, got {values.shape[1]}")
        scaled = self.scaler_.transform(self.imputer_.transform(values))
        angles = self._angles(self.linear32_.transform(scaled))
        return scaled, self._blocks(angles, self.theta_)


class VQCLayerReadout:
    """Bayesian Ridge residual readout anchored to paper Hybrid."""

    def __init__(self, readout: str, blend: float = 0.30, random_state: int = 42):
        if readout not in READOUT_COMBINATIONS:
            raise KeyError(f"Unknown readout: {readout}")
        self.readout = readout
        self.blend = float(blend)
        self.random_state = int(random_state)

    def fit_from_map(self, feature_map: VQCLayerFeatureMap, X: np.ndarray, y: np.ndarray, anchor_train: np.ndarray):
        self.feature_map_ = feature_map
        scaled, blocks = feature_map.transform(X)
        qfeatures = np.concatenate([blocks[name] for name in READOUT_COMBINATIONS[self.readout]], axis=1)
        design = np.concatenate([scaled, qfeatures], axis=1)
        target = np.asarray(y, dtype=float).reshape(-1)
        anchor = np.asarray(anchor_train, dtype=float).reshape(-1)
        self.head_ = make_pipeline(StandardScaler(), BayesianRidge(alpha_1=1e-6, alpha_2=1e-6))
        self.head_.fit(design, target - anchor)
        residual = target - anchor
        self.low_, self.high_ = np.quantile(residual, [0.01, 0.99])
        return self

    def predict(self, X: np.ndarray, anchor_test: np.ndarray, paper_hybrid_test: np.ndarray) -> np.ndarray:
        scaled, blocks = self.feature_map_.transform(X)
        qfeatures = np.concatenate([blocks[name] for name in READOUT_COMBINATIONS[self.readout]], axis=1)
        design = np.concatenate([scaled, qfeatures], axis=1)
        residual = np.clip(self.head_.predict(design), self.low_, self.high_)
        quantum_candidate = np.asarray(anchor_test, dtype=float) + residual
        paper_hybrid = np.asarray(paper_hybrid_test, dtype=float)
        return (1.0 - self.blend) * paper_hybrid + self.blend * quantum_candidate


LAYER_CONFIGS = {f"L{layer}": LayerConfig(f"L{layer}", layer) for layer in range(1, 11)}


def make_layer_readout_cases() -> list[LayerReadoutCase]:
    return [
        LayerReadoutCase(
            name=f"VQC20_L{layer}_{readout}_bayesian_ridge",
            layer=layer,
            readout=readout,
            description=f"5-qubit angle VQC layer {layer}, RY/RZ + chain CNOT, readout {readout}, Bayesian Ridge residual + Hybrid anchor",
        )
        for layer in range(1, 11)
        for readout in READOUT_COMBINATIONS
    ]


CASE_DESCRIPTIONS = {case.name: case.description for case in make_layer_readout_cases()}


__all__ = [
    "CASE_DESCRIPTIONS",
    "LAYER_CONFIGS",
    "LayerReadoutCase",
    "READOUT_COMBINATIONS",
    "VQCLayerFeatureMap",
    "VQCLayerReadout",
    "make_layer_readout_cases",
]
