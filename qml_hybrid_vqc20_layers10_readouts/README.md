# VQC20: 10 layers × 3 Z/ZZ readouts

This is a drop-in subfolder for `PM25`. It evaluates all 30 combinations:

```text
VQC layers: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
Readouts: Z, ZZ, Z+ZZ
Total: 10 × 3 = 30 cases
```

## Architecture

The input is exactly the paper's 20 feature columns. No PCA, X/Y readout,
parity readout or extra raw feature is added.

```text
20 paper features
→ train-only imputation and standardization
→ fixed Linear(20,32)
→ 5 angle values on 5 qubits
→ L trainable RY/RZ layers + chain CNOTs
→ selected Z/nearest-neighbour ZZ readouts
→ concatenate with the 20 classical features
→ Bayesian Ridge residual readout
→ conservative blend with paper Hybrid
```

For a layer (L), the VQC has (10L) trainable rotation parameters:

```text
2 rotations × 5 qubits × L layers = 10L parameters
```

The VQC is trained only on the training residual. The final prediction is
anchored to the paper Hybrid prediction. Test `Ref` is used only when metrics
are calculated, never during feature construction, VQC fitting or Bayesian
Ridge fitting.

The `db1` wavelet feature implementation is self-contained as Haar, so the
PyWavelets buffer error cannot occur.

## Run

```bash
cd /home/sparc/workdir/sonnguyen/PM25
unzip -o /path/to/qml_hybrid_vqc20_layers10_readouts.zip

bash qml_hybrid_vqc20_layers10_readouts/run_smoke.sh
bash qml_hybrid_vqc20_layers10_readouts/run_no_leakage.sh

# Run all 30 cases plus the seven paper baselines
bash qml_hybrid_vqc20_layers10_readouts/run_all.sh

# Generate paper-style figures from the saved outputs
bash qml_hybrid_vqc20_layers10_readouts/run_figures.sh
```

For a faster test:

```bash
bash qml_hybrid_vqc20_layers10_readouts/run_all.sh \
  --spsa-steps 1 --max-fit-samples 48 --quiet
```

To run selected cases:

```bash
python3 -m qml_hybrid_vqc20_layers10_readouts.run \
  --models VQC20_L2_Z_bayesian_ridge,VQC20_L2_Z_ZZ_bayesian_ridge \
  --spsa-steps 6 --max-fit-samples 160
```

Results are saved to `results/qml_hybrid_vqc20_layers10_readouts/`. The run also saves
all 20 paper feature columns for train/test per sensor, VQC theta values and
SPSA losses in `run_metadata.json`, and a `figures/` directory after
`run_figures.sh`. With the seven paper comparators enabled, the complete run
contains 148 metric rows: 37 methods × 4 sensors. Each figure is saved as
both PNG and PDF.

The delivery ZIP includes `example_results_layers1_10_spsa0/`, a complete
30-case validation snapshot with all CSV/metadata/figures. It was generated
with `--spsa-steps 0` only to validate the full layer/readout export path;
rerun `run_all.sh` with the default SPSA settings for the final experiment.
