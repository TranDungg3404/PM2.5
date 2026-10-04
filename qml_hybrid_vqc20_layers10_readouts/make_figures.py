"""Create paper-style plots from a completed benchmark directory."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CASE_RE = re.compile(r"^VQC20_L(?P<layer>\d+)_(?P<readout>Z_ZZ|Z|ZZ)_bayesian_ridge$")


def _save(fig, path: Path):
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _read(output_dir: Path):
    metrics = pd.read_csv(output_dir / "metrics.csv")
    predictions = pd.read_csv(output_dir / "test_predictions_long.csv", parse_dates=["Date"])
    metrics["Layer"] = np.nan
    metrics["Readout"] = "Paper"
    for i, method in metrics["Method"].items():
        match = CASE_RE.match(str(method))
        if match:
            metrics.loc[i, "Layer"] = int(match.group("layer"))
            metrics.loc[i, "Readout"] = match.group("readout")
    return metrics, predictions


def _ranking(metrics: pd.DataFrame, fig_dir: Path):
    summary = metrics.groupby("Method", as_index=False).agg(
        MAE=("MAE", "mean"), RMSE=("RMSE", "mean"),
        Correlation=("Correlation", "mean"), BiasAbs=("Bias", lambda x: x.abs().mean()),
    ).sort_values(["MAE", "RMSE"])
    summary.to_csv(fig_dir / "summary_by_method.csv", index=False)
    shown = summary.head(min(20, len(summary))).iloc[::-1]
    colors = ["#c44e52" if str(v).startswith("VQC20_") else "#4c72b0" for v in shown["Method"]]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(shown["Method"], shown["MAE"], color=colors)
    ax.set_xlabel("Mean test MAE")
    ax.set_title("Model ranking by mean test MAE")
    ax.grid(axis="x", alpha=0.25)
    _save(fig, fig_dir / "01_model_ranking_mae.png")
    return summary


def _layer_curves(metrics: pd.DataFrame, fig_dir: Path):
    vqc = metrics[metrics["Layer"].notna()].copy()
    grouped = vqc.groupby(["Layer", "Readout"], as_index=False).agg(
        MAE=("MAE", "mean"), RMSE=("RMSE", "mean"), Correlation=("Correlation", "mean")
    )
    grouped.to_csv(fig_dir / "summary_by_layer_readout.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for readout, group in grouped.groupby("Readout"):
        group = group.sort_values("Layer")
        axes[0].plot(group["Layer"], group["MAE"], marker="o", label=readout)
        axes[1].plot(group["Layer"], group["Correlation"], marker="o", label=readout)
    hybrid = metrics.loc[metrics["Method"] == "Hybrid"]
    if len(hybrid):
        axes[0].axhline(hybrid["MAE"].mean(), color="black", linestyle="--", label="Hybrid")
        axes[1].axhline(hybrid["Correlation"].mean(), color="black", linestyle="--", label="Hybrid")
    axes[0].set_ylabel("Mean test MAE")
    axes[1].set_ylabel("Mean test correlation")
    for ax in axes:
        ax.set_xlabel("VQC layer")
        ax.set_xticks(range(1, 11))
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Effect of VQC depth and readout")
    _save(fig, fig_dir / "02_layer_readout_curves.png")


def _heatmap(metrics: pd.DataFrame, fig_dir: Path):
    summary = metrics.groupby("Method")["MAE"].mean().sort_values()
    selected = list(summary.head(15).index)
    if "Hybrid" in metrics["Method"].unique() and "Hybrid" not in selected:
        selected.append("Hybrid")
    table = metrics[metrics["Method"].isin(selected)].pivot_table(index="Method", columns="Sensor", values="MAE", aggfunc="mean")
    table = table.loc[[m for m in selected if m in table.index]]
    table.to_csv(fig_dir / "mae_by_method_sensor.csv")
    fig, ax = plt.subplots(figsize=(9, max(5, len(table) * 0.38)))
    image = ax.imshow(table.to_numpy(), aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set_xlabel("Sensor")
    ax.set_title("Test MAE by model and sensor")
    fig.colorbar(image, ax=ax, label="MAE")
    _save(fig, fig_dir / "03_mae_heatmap_sensor.png")


def _prediction_plots(metrics: pd.DataFrame, predictions: pd.DataFrame, fig_dir: Path):
    vqc_metrics = metrics[metrics["Layer"].notna()]
    best_by_sensor = vqc_metrics.loc[vqc_metrics.groupby("Sensor")["MAE"].idxmin(), ["Sensor", "Method", "MAE"]]
    best_by_sensor.to_csv(fig_dir / "best_vqc_by_sensor.csv", index=False)
    methods = ["Uncalibrated", "Hybrid"]
    for method in best_by_sensor["Method"]:
        if method not in methods:
            methods.append(method)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=False)
    for ax, sensor in zip(axes.ravel(), sorted(predictions["Sensor"].unique())):
        part = predictions[predictions["Sensor"] == sensor].sort_values("Date")
        for method in methods:
            line = part[part["Method"] == method]
            if len(line):
                ax.plot(line["Date"], line["Prediction"], linewidth=1.0, label=method)
        ax.plot(part["Date"].drop_duplicates(), part.drop_duplicates("Date")["Ref"], color="black", linewidth=1.5, label="Ref")
        ax.set_title(sensor)
        ax.set_ylabel("PM2.5")
        ax.grid(alpha=0.2)
        ax.tick_params(axis="x", rotation=25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.suptitle("Reference and calibrated test predictions")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, fig_dir / "04_test_predictions_time_series.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, sensor in zip(axes.ravel(), sorted(predictions["Sensor"].unique())):
        best = best_by_sensor.loc[best_by_sensor["Sensor"] == sensor, "Method"].iloc[0]
        part = predictions[(predictions["Sensor"] == sensor) & (predictions["Method"] == best)]
        ax.scatter(part["Ref"], part["Prediction"], s=10, alpha=0.55)
        lo = min(part["Ref"].min(), part["Prediction"].min())
        hi = max(part["Ref"].max(), part["Prediction"].max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_title(f"{sensor}: {best}")
        ax.set_xlabel("Reference PM2.5")
        ax.set_ylabel("Prediction PM2.5")
        ax.grid(alpha=0.2)
    fig.suptitle("Reference–prediction agreement for the best VQC case")
    fig.tight_layout()
    _save(fig, fig_dir / "05_best_vqc_scatter.png")

    residual_rows = []
    for _, row in best_by_sensor.iterrows():
        part = predictions[(predictions["Sensor"] == row["Sensor"]) & (predictions["Method"] == row["Method"])].copy()
        part["Residual"] = part["Prediction"] - part["Ref"]
        residual_rows.append(part[["Sensor", "Method", "Date", "Residual"]])
    residual = pd.concat(residual_rows, ignore_index=True)
    residual.to_csv(fig_dir / "best_vqc_residuals.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    residual.boxplot(column="Residual", by="Sensor", ax=ax, grid=False)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Residual distribution of best VQC model per sensor")
    fig.suptitle("")
    ax.set_ylabel("Prediction − reference")
    _save(fig, fig_dir / "06_best_vqc_residuals.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/qml_hybrid_vqc20_layers10_readouts")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics, predictions = _read(output_dir)
    summary = _ranking(metrics, fig_dir)
    _layer_curves(metrics, fig_dir)
    _heatmap(metrics, fig_dir)
    _prediction_plots(metrics, predictions, fig_dir)
    print(f"Saved {len(list(fig_dir.glob('*.png')))} PNG figures to {fig_dir}")
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
