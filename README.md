# Frequency-aware conformal calibration for temporal knowledge graph forecasting

Code for the AISTATS 2027 submission *Conformal Prediction over Open, Drifting Label Spaces:
Frequency-Aware Calibration for Temporal Knowledge Graph Forecasting* (anonymous).

All calibration methods run on stored forecaster scores, so every method in a comparison sees
exactly the same predictive distribution.

## Pipeline

1. **Dump scores** from a trained forecaster (CyGNet / CENET style models):
   `cp_dump.py` writes `{prefix}_probs.npy` (`[Q, |E|]`, softmax over all entities, test order)
   and `{prefix}_meta.npz` (answers `y`, training answer frequency `ent_freq`, timestamps,
   last-seen times for recency). Benchmarks (ICEWS18, GDELT, WIKI) are the standard public
   versions distributed with prior TKG work.
2. **Random-split evaluation** (Table 1, appendices A/B):
   ```
   python cp_harness.py --prefix scores/copy_icews18 --alpha 0.1 --splits 100 --out results/icews18.json
   ```
   Methods: `std` (split CP), `pas`, `interp` (Interp-Q), `clustered`, `mond_f` (Mondrian-F),
   `mond_bo` (FRB), `fcp` (Feature-norm.), `fnm` (FNM, proposed).
   `cp_harness_f64.py` is the same harness with double-precision scores (precision audit).
3. **Forward-only rolling evaluation** with ACI, Mondrian-ACI and DA-FNM:
   `python rolling_cp.py --prefix scores/copy_icews18 --blocks 10`
4. **Controlled within-stratum drift simulation**: `python synth_shift.py --out paper`
5. **Diagnostics**: `drift_kstrata.py` (mixture-shift prediction), `tie_check.py`
   (ties / precision), `make_struct.py` (cold-candidate perturbation), `make_open.py`
   (aggregated NEW label), `make_feats_harness.py` (feature ablation), `add_time.py`
   (attach timestamps / recency to a meta file).
6. **Tables and figures**: `make_tables.py`, `make_figs.py`, `make_bin_fig.py`,
   `make_open_table.py`, `make_rolling_table.py`, `make_rolling_compact.py`
   turn the JSON outputs into the LaTeX macros, tables and figures of the paper.

Run any script with `-h` for its options.

## Requirements
Python >= 3.9, see `requirements.txt`. A GPU is used for set-size evaluation when available.

## Data and scores
Processed metadata and stored forecaster scores are large (Q x |E| matrices) and will be
released alongside this repository.
