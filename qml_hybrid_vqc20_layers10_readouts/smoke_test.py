"""Smoke-test all 30 layer/readout cases on a small synthetic data set."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qml_hybrid_vqc20_layers10_readouts.models import READOUT_COMBINATIONS, VQCLayerFeatureMap, make_layer_readout_cases
from qml_hybrid_vqc20_layers10_readouts.paper_features import PAPER20_FEATURE_NAMES, Paper20FeatureBuilder
from qml_hybrid_vqc20_layers10_readouts.run import run_benchmark


def main():
    rng = np.random.default_rng(31)
    data = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=72, freq="h"),
        "Ref": rng.normal(10, 1, 72),
        "S1": rng.normal(10, 1, 72), "S2": rng.normal(10, 1, 72),
        "S3": rng.normal(10, 1, 72), "S4": rng.normal(10, 1, 72),
    })
    feature_frame = Paper20FeatureBuilder().build(data, "S1", ["S1", "S2", "S3", "S4"])
    assert feature_frame.shape == (72, 20)
    assert list(feature_frame.columns) == PAPER20_FEATURE_NAMES
    fmap = VQCLayerFeatureMap(2, spsa_steps=1, max_fit_samples=20).fit(
        feature_frame.to_numpy(), data["Ref"].to_numpy()
    )
    _, blocks = fmap.transform(feature_frame.to_numpy()[:5])
    assert blocks["Z"].shape == (5, 5)
    assert blocks["ZZ"].shape == (5, 4)
    cases = make_layer_readout_cases()
    assert len(cases) == 30
    assert [case.layer for case in cases[:3]] == [1] * 3
    assert set(case.readout for case in cases) == set(READOUT_COMBINATIONS)
    metrics, predictions, metadata = run_benchmark(
        data, train_ratio=0.70, spsa_steps=0, max_fit_samples=20,
        cases=cases, include_paper=False, verbose=False,
    )
    assert len(metrics) == 30 * 4
    assert len(predictions) == 30 * 4 * 22
    assert metadata["layers_tested"] == list(range(1, 11))
    print("VQC 10-layer x 3-readout smoke test: PASS")
    print(f"cases={len(cases)} features=20 qubits=5")


if __name__ == "__main__":
    main()
