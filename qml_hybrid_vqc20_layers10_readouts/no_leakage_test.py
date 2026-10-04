"""Verify test Ref is not used by the 30-case evaluator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qml_hybrid_vqc20_layers10_readouts.models import make_layer_readout_cases
from qml_hybrid_vqc20_layers10_readouts.run import run_benchmark


def main():
    data = pd.read_csv("combined_data.csv").iloc[:120].copy()
    kwargs = {
        "train_ratio": 0.70, "spsa_steps": 0, "max_fit_samples": 20,
        "random_state": 9, "cases": [make_layer_readout_cases()[0]],
        "include_paper": False, "verbose": False,
    }
    _, first, _ = run_benchmark(data, **kwargs)
    changed = data.copy()
    split = int(len(changed) * kwargs["train_ratio"])
    changed.loc[split:, "Ref"] += 99999.0
    _, second, _ = run_benchmark(changed, **kwargs)
    first = first.sort_values(["Method", "Sensor", "Date"]).reset_index(drop=True)
    second = second.sort_values(["Method", "Sensor", "Date"]).reset_index(drop=True)
    assert np.allclose(first["Prediction"], second["Prediction"], atol=1e-10)
    print("VQC 10-layer x 3-readout no-leakage test: PASS")


if __name__ == "__main__":
    main()
