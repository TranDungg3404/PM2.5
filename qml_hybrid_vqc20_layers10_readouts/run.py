"""Benchmark 30 VQC-Hybrid cases: layers 1..10 and Z/ZZ/Z+ZZ readouts."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qml_hybrid_vqc20_layers10_readouts.models import (  # noqa: E402
    CASE_DESCRIPTIONS,
    LAYER_CONFIGS,
    READOUT_COMBINATIONS,
    LayerReadoutCase,
    VQCLayerFeatureMap,
    VQCLayerReadout,
    make_layer_readout_cases,
)
from qml_hybrid_vqc20_layers10_readouts.paper_features import (  # noqa: E402
    PAPER20_FEATURE_DESCRIPTIONS,
    PAPER20_FEATURE_NAMES,
    Paper20FeatureBuilder,
)
from src.baselines import evaluate_classical_baselines  # noqa: E402
from src.calibrator import LeakageFreeHybridCalibrator  # noqa: E402
from src.evaluation import compute_full_metrics, save_evaluation_outputs  # noqa: E402


PAPER_MODELS = ["Uncalibrated", "OffsetScale", "LinearRegression", "RandomForest", "GradientBoosting", "SVR", "Hybrid"]


def _seed(text: str, base: int) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little") % 100_000 + base


def _sort(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    if "Date" not in df.columns:
        raise KeyError("The dataset must contain Date")
    date = pd.to_datetime(df["Date"], format="%d.%m.%Y %H:%M", errors="coerce")
    if date.isna().any():
        date = pd.to_datetime(df["Date"], format="mixed", errors="coerce")
    if date.isna().any() or date.duplicated().any():
        raise ValueError("Date contains invalid or duplicate timestamps")
    df["Date"] = date
    return df.sort_values("Date", kind="stable").reset_index(drop=True)


def _split(df: pd.DataFrame, ratio: float):
    index = int(len(df) * ratio)
    if index < 2 or index >= len(df) - 1:
        raise ValueError("Temporal split leaves too few rows")
    return df.iloc[:index].copy(), df.iloc[index:].copy(), index


def _feature_pair(train_corrected, test_corrected, sensor, sensors, trust, builder):
    train_features = builder.build(
        train_corrected.loc[:, ["Date", *sensors]], sensor, sensors, trust_scores=trust
    )
    history = train_corrected.loc[:, ["Date", *sensors]].tail(23).reset_index(drop=True)
    test_features = builder.build(
        test_corrected.loc[:, ["Date", *sensors]], sensor, sensors,
        trust_scores=trust, history=history
    )
    if list(train_features.columns) != PAPER20_FEATURE_NAMES:
        raise RuntimeError("Feature order is not the paper's 20-feature order")
    return train_features.to_numpy(dtype=float), test_features.to_numpy(dtype=float)


def _fit_sensor(x_train, x_test, y_train, anchor_train, anchor_test, hybrid_test, selected, sensor, steps, max_samples, seed, verbose):
    residual = y_train - anchor_train
    layer_maps = {}
    diagnostics = {}
    for layer in sorted({case.layer for case in selected}):
        if verbose:
            print(f"[VQC] sensor={sensor} layer={layer}")
        feature_map = VQCLayerFeatureMap(
            layer, spsa_steps=steps, max_fit_samples=max_samples,
            random_state=_seed(f"{sensor}-L{layer}", seed),
        ).fit(x_train, residual)
        layer_maps[layer] = feature_map
        diagnostics[f"L{layer}"] = {
            "layer": layer,
            "n_qubits": 5,
            "encoding": "angle",
            "ansatz": "RY/RZ trainable rotations + chain CNOT",
            "theta_parameters": feature_map.theta_size,
            "spsa_steps": steps,
            "fit_samples_for_spsa": min(len(x_train), max_samples),
            "allowed_readout_blocks": {name: list(values) for name, values in READOUT_COMBINATIONS.items()},
            "training_objective": float(feature_map.training_objective_),
            "theta_values": feature_map.theta_.tolist(),
            "spsa_losses": list(feature_map.spsa_losses_),
        }

    predictions = {}
    for case in selected:
        model = VQCLayerReadout(case.readout, blend=0.30, random_state=_seed(case.name, seed))
        model.fit_from_map(layer_maps[case.layer], x_train, y_train, anchor_train)
        predictions[case.name] = model.predict(x_test, anchor_test, hybrid_test)
    return predictions, diagnostics


def run_benchmark(data: pd.DataFrame, train_ratio=0.70, spsa_steps=6, max_fit_samples=160,
                  random_state=42, cases=None, include_paper=True, output_dir=None, verbose=True):
    df = _sort(data)
    sensors = ["S1", "S2", "S3", "S4"]
    missing = [c for c in [*sensors, "Ref"] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    train_df, test_df, split_index = _split(df, train_ratio)
    selected = list(cases if cases is not None else make_layer_readout_cases())

    calibrator = LeakageFreeHybridCalibrator(
        window_size=24, error_scale=2.0, min_trust_score=0.2,
        sensor_positions={}, use_spatial=False, random_state=random_state,
    )
    calibrator.fit(train_df, sensors, "Ref")
    corrected_train = calibrator.apply_offset_scale(train_df, sensors)
    corrected_test = calibrator.apply_offset_scale(test_df, sensors)
    hybrid_test = calibrator.transform(test_df, sensors=sensors, ref="Ref")

    predictions = {}
    if include_paper:
        predictions.update(evaluate_classical_baselines(train_df, test_df, sensors, "Ref", calibrator, random_state))
        predictions["Hybrid"] = {s: hybrid_test[s].to_numpy(dtype=float) for s in sensors}

    builder = Paper20FeatureBuilder(window_size=24, wavelet_level=3)
    y_train = train_df["Ref"].to_numpy(dtype=float)
    diagnostics = {}
    feature_tables = {}
    for index, sensor in enumerate(sensors):
        x_train, x_test = _feature_pair(corrected_train, corrected_test, sensor, sensors, calibrator.trust_scores_, builder)
        feature_tables[sensor] = {
            "train": pd.DataFrame(x_train, columns=PAPER20_FEATURE_NAMES),
            "test": pd.DataFrame(x_test, columns=PAPER20_FEATURE_NAMES),
        }
        sensor_predictions, sensor_diagnostics = _fit_sensor(
            x_train, x_test, y_train,
            corrected_train[sensor].to_numpy(dtype=float),
            corrected_test[sensor].to_numpy(dtype=float),
            hybrid_test[sensor].to_numpy(dtype=float), selected, sensor,
            spsa_steps, max_fit_samples, random_state + index * 10007, verbose,
        )
        for method, values in sensor_predictions.items():
            predictions.setdefault(method, {})[sensor] = values
        diagnostics[sensor] = sensor_diagnostics

    metric_rows, prediction_rows = [], []
    y_test = test_df["Ref"].to_numpy(dtype=float)
    for method, by_sensor in predictions.items():
        for sensor in sensors:
            metric = compute_full_metrics(y_test, by_sensor[sensor])
            raw_mae = compute_full_metrics(y_test, test_df[sensor].to_numpy(dtype=float))["MAE"]
            metric_rows.append({"Sensor": sensor, "Method": method, **metric, "RawTestMAE": raw_mae,
                                "MAEReductionVsRawTest_percent": (raw_mae - metric["MAE"]) / raw_mae * 100.0})
            for date, ref, value in zip(test_df["Date"], test_df["Ref"], by_sensor[sensor]):
                prediction_rows.append({"Date": date, "Ref": float(ref), "Sensor": sensor,
                                        "Method": method, "Prediction": float(value)})

    metrics = pd.DataFrame(metric_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    metadata = {
        "n_rows": len(df), "train_rows": len(train_df), "test_rows": len(test_df),
        "train_ratio": train_ratio, "split_index": split_index,
        "protocol": "chronological_train_fit_transform_test",
        "input_feature_count": 20, "input_feature_names": PAPER20_FEATURE_NAMES,
        "input_feature_descriptions": PAPER20_FEATURE_DESCRIPTIONS,
        "n_qubits": 5, "layers_tested": list(range(1, 11)),
        "readouts_tested": list(READOUT_COMBINATIONS),
        "quantum_readout_rule": "only Z, nearest-neighbour ZZ, and Z+ZZ; no X/Y/parity",
        "cases": [case.name for case in selected],
        "case_descriptions": {case.name: CASE_DESCRIPTIONS[case.name] for case in selected},
        "paper_models": PAPER_MODELS if include_paper else [],
        "vqc_diagnostics": diagnostics,
    }
    if output_dir is not None:
        save_evaluation_outputs(output_dir, metrics, predictions_df, metadata, calibrator)
        output_path = Path(output_dir)
        for sensor, tables in feature_tables.items():
            tables["train"].to_csv(output_path / f"paper20_features_train_{sensor}.csv", index=False)
            tables["test"].to_csv(output_path / f"paper20_features_test_{sensor}.csv", index=False)
    return metrics, predictions_df, metadata


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(PROJECT_ROOT / "combined_data.csv"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "qml_hybrid_vqc20_layers10_readouts"))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--spsa-steps", type=int, default=6)
    parser.add_argument("--max-fit-samples", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", default="all", help="case names separated by comma, or all")
    parser.add_argument("--no-paper", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    path = Path(args.data)
    if not path.is_absolute():
        path = Path.cwd() / path
    all_cases = make_layer_readout_cases()
    if args.models.lower() == "all":
        selected = all_cases
    else:
        lookup = {case.name: case for case in all_cases}
        names = [item.strip() for item in args.models.split(",") if item.strip()]
        unknown = [item for item in names if item not in lookup]
        if unknown:
            raise KeyError(f"Unknown cases: {unknown}")
        selected = [lookup[item] for item in names]
    if args.limit > 0:
        selected = selected[:args.limit]
    metrics, _, metadata = run_benchmark(
        pd.read_csv(path), train_ratio=args.train_ratio, spsa_steps=args.spsa_steps,
        max_fit_samples=args.max_fit_samples, random_state=args.seed, cases=selected,
        include_paper=not args.no_paper, output_dir=args.output_dir, verbose=not args.quiet,
    )
    ranking = metrics.groupby("Method", as_index=False).agg(MAE=("MAE", "mean"), RMSE=("RMSE", "mean"), Correlation=("Correlation", "mean")).sort_values(["MAE", "RMSE"])
    print(ranking.to_string(index=False))
    print(f"\nSaved {len(metrics)} metric rows for {metadata['train_rows']} train and {metadata['test_rows']} test observations.")


if __name__ == "__main__":
    main()
