# Tabular learning: ModernNCA and encoder heads

[Русская версия](README.ru.md)

Research comparison of four models on 30 tabular datasets, including `walking-activity`: 15 seeds per dataset and model, 1,800 test evaluations. The repository contains data, experiment code and archived evidence.

## Models

| CLI name | Method |
| --- | --- |
| `modernnca` | Neural encoder with numerical PLR embeddings and a differentiable distance-weighted neighbor predictor. |
| `catboost` | CatBoost on original features, with native categorical handling. |
| `encoder_linear` | Encoder and linear head trained jointly; cross-entropy for classification, one output for regression. |
| `encoder_catboost` | Encoder trained with the ModernNCA neighbor objective, then frozen; CatBoost fitted on its embeddings. |

The hybrid always takes encoder hyperparameters from `modernnca_best_params.json`. During head tuning, embeddings are computed once; final evaluations retrain the encoder for each seed. Saved hyperparameters are not trained network weights.

## Protocol

- 20 classification and 10 regression datasets. Exact names are the keys of `modernnca_best_params.json`. Five predefined user groups cover 29 datasets; `walking-activity` is separate.
- Each seed generates a 64%/16%/20% train/validation/test split, stratified for classification. Neural preprocessing is fitted on train: mean imputation and standardization for numerical features; a missing-value category and one-hot encoding for categorical inputs.
- Optuna TPE uses validation accuracy (maximize) or RMSE (minimize), tuning seed 0, at most 100 trials, and stops after 10 consecutive trials without improvement. The hybrid reuses ModernNCA encoder parameters and tunes its CatBoost head separately.
- Final test evaluations use seeds 0–14. Neural training: up to 200 epochs, patience 20, batch size 1,024. CatBoost: up to 2,000 iterations, early stopping patience 50.
- `hidden_dim` and `embed_dim` search bounds are 64–512. This change and the additional heads make this an adaptation, rather than an exact reproduction of the reference paper, included as [KNN.pdf](KNN.pdf).

## Setup and execution

Run from the repository root on Linux. Server runs used Python 3.12, PyTorch 2.5.1 with CUDA 12.1, and NVIDIA RTX A4000 GPUs (16 GB). Other dependencies in `requirements.txt` are not fully version-locked.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
tmux new -s benchmark
```

Example: four models on one dataset, using an assigned GPU:

```bash
python -u run_benchmark.py --user user3 --gpu 0 --datasets stock \
  --models modernnca catboost encoder_linear encoder_catboost \
  --results-dir runs/stock 2>&1 | tee runs_stock.log
```

Omit `--datasets` for the group assigned to `--user user1`–`user5`; use `--datasets walking-activity` for the separate dataset. `--gpu` selects the physical GPU index. Each person should use their own checkout and an agreed GPU. By default, only `modernnca catboost` are selected.

Add `--smoke` for a minimal check: one trial, one seed, one epoch, two trees; dataset defaults to `stock`. For CPU, replace `--gpu 0` with `--device cpu`. Other options: `python run_benchmark.py --help`. Detach from tmux with Ctrl+B, then D; reconnect with `tmux attach -t benchmark`.

Use a fresh `--results-dir runs/<name>` for new experiments to preserve the tracked archive. Outputs include metrics CSV, timing CSV, progress JSON and `optuna.db`. Console logs show dataset/model/stage, elapsed time and an evolving completion estimate; capture them with `tee`. Repeating a command in the same directory reuses compatible Optuna studies; final seed evaluations run again.

## Archived evidence

| File in `results/` | Contents |
| --- | --- |
| `benchmark_all_30_datasets_4_models_1800_results.csv` | Final metrics, seeds, configurations, sources and available timing measurements. |
| `benchmark_all_30_datasets_4_models_full.log` | Consolidated source logs and timing records. |
| `optuna.db` | Merged tuning history: 120 studies, 30 datasets, four models, 2,707 trials. |

The final CSV is the metric record. The database preserves available archived studies; older logs can contain other runs of the same study. Detailed training/inference timing is available for `walking-activity` and the two encoder-head models. Other earlier runs have Optuna logs only; blank timing fields mean unavailable measurements. Inference timings measure test-batch prediction, not single-request latency.

Compare mean scores across seeds within each dataset: higher accuracy and lower RMSE are better. Do not average raw accuracy and RMSE across tasks. Subset selection, changing splits across seeds, unpinned dependencies and shared-server load limit generalization and exact timing reproducibility.

## Code and checks

`Stream1_DataExtraction.py`: loading/splits; `Stream2_Preprocessing.py`: features; `Stream3_ModernNCA.py`: encoder; `Stream4_Train.py`: neighbor training; `Stream5_NeuralHeads.py`: alternative heads; `steram5.py`: baseline/tuning utilities. `run_benchmark.py` is the CLI entry point, `run_progress.py` handles timing/progress, and `Pipeline.ipynb` provides interactive execution.

```bash
python -m unittest discover -s tests
```
