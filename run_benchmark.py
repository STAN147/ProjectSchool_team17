"""Запуск бенчмарка из командной строки без Jupyter.

python -u run_benchmark.py --user user1 --gpu 0
python -u run_benchmark.py --user user3 --gpu 2 --datasets stock
"""

import argparse
import csv
import hashlib
import json
import os
import math
from datetime import datetime, timezone
from pathlib import Path

USER_DATASETS = {
    "user1": [
        "adult"
    ],
    "user2": [
        "fried",
        "taiwanese_bankruptcy_prediction",
        "led7",
        "Marketing_Campaign",
        "Goodreads-Computer-Books"
    ],
    "user3": [
        "gas_turbine_CO_and_NOx_emission",
        "Bank_Customer_Churn_Dataset",
        "Bias_correction_r",
        "Wilt",
        "Contaminant-detection-in-packaged-cocoa-hazelnut-spread-jars-using-Microwaves-Sensing-and-Machine-Learning-9.5GHz(Urbinati)",
        "MIC",
        "stock"
    ],
    "user4": [
        "naticusdroid+android+permissions+dataset",
        "BNG(echoMonths)",
        "JapaneseVowels",
        "telco-customer-churn",
        "wine-quality-white",
        "seismic+bumps",
        "estimation_of_obesity_levels"
    ],
    "user5": [
        "letter",
        "Large-scale_Wave_Energy_Farm_Sydney_49",
        "dry_bean_dataset",
        "sulfur",
        "FICO-HELOC-cleaned",
        "rl",
        "Mobile_Phone_Market_in_Ghana",
        "Large-scale_Wave_Energy_Farm_Sydney_100",
        "maternal_health_risk"
    ]
}


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Значение должно быть больше нуля")
    return number


def gpu_index(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Индекс GPU не может быть отрицательным")
    return number


def positive_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Log interval must be positive and finite")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user", required=True, choices=USER_DATASETS, help="Назначенная группа датасетов")
    hardware = parser.add_mutually_exclusive_group(required=True)
    hardware.add_argument("--gpu", type=gpu_index, help="Физический индекс выделенной GPU; внутри процесса будет cuda:0")
    hardware.add_argument("--device", choices=["cpu"], help="Запуск на CPU")
    parser.add_argument("--datasets", nargs="+", choices=[name for group in USER_DATASETS.values() for name in group], metavar="DATASET", help="Конкретные датасеты вместо всей группы")
    parser.add_argument("--models", nargs="+", choices=["modernnca", "catboost", "mlp", "xgboost"], default=["modernnca", "catboost"])
    parser.add_argument("--results-dir", default="results", help="Каталог результатов относительно папки проекта")
    parser.add_argument("--n-trials", type=positive_int, default=100)
    parser.add_argument("--n-seeds", type=positive_int, default=15)
    parser.add_argument("--max-epochs", type=positive_int, default=200)
    parser.add_argument("--max-boost-rounds", type=positive_int, default=2000)
    parser.add_argument("--optuna-patience", type=positive_int, default=10)
    parser.add_argument("--cpu-threads", type=positive_int, default=4)
    parser.add_argument("--log-interval", type=positive_seconds, default=60, help="Training progress interval in seconds")
    parser.add_argument("--smoke", action="store_true", help="Короткий запуск: stock по умолчанию, 1 trial, 1 сид, 1 эпоха, 2 дерева")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)
    os.chdir(Path(__file__).resolve().parent)

    # Выбор GPU и ограничение потоков задаются до импорта библиотек.
    import torch
    import optuna
    import pandas as pd
    from Stream1_DataExtraction import load_dataset
    from Stream2_Preprocessing import TabularPreprocessor, PLREmbeddings
    from Stream3_ModernNCA import ModernNCAEncoder
    from Stream4_Train import Trainer
    from steram5 import (
        compute_metric, optuna_patience_reached, stop_optuna_if_no_improvement,
        tune_hyperparameters, run_experiment,
    )

    device = torch.device("cuda:0" if args.gpu is not None else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна. Проверьте индекс GPU и установку CUDA-версии PyTorch; для CPU используйте --device cpu.")
    torch.set_num_threads(args.cpu_threads)
    config = {
        "datasets": args.datasets or (["stock"] if args.smoke else USER_DATASETS[args.user]),
        "results_dir": args.results_dir,
        "gpu_memory_fraction": 0.6,
        "models": args.models,
        "seeds": list(range(args.n_seeds)),
        "tune_seed": 0,
        "n_trials": args.n_trials,
        "optuna_patience": args.optuna_patience,
        "max_epochs": args.max_epochs,
        "patience": 20,
        "max_boost_rounds": args.max_boost_rounds,
        "early_stopping_rounds": 50,
        "batch_size": 1024,
    }
    if args.smoke:
        config.update(seeds=[0], n_trials=1, max_epochs=1, max_boost_rounds=2, patience=1, early_stopping_rounds=1)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(config["gpu_memory_fraction"], device=device)
        print("GPU", args.gpu, torch.cuda.get_device_name(device))
    print("User", args.user, "Device", device)
    print("Config", json.dumps(config, ensure_ascii=False))
    from run_progress import RunProgress
    progress = RunProgress(args.user, config, args.log_interval,
                           synchronize=(lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None))
    print("Progress", progress.path, "Timings", progress.timings_path, flush=True)

    def prepare_run(dataset_name, seed):
        dataset = load_dataset(dataset_name, seed=seed)
        prep = TabularPreprocessor(dataset.numerical_columns, dataset.categorical_columns)
        prep.fit(dataset.X_train)
        train = prep.transform(dataset.X_train, dataset.y_train)
        val = prep.transform(dataset.X_val, dataset.y_val)
        test = prep.transform(dataset.X_test, dataset.y_test)
        return dataset, prep, train, val, test


    def build_modern_nca(params, seed, train):
        torch.manual_seed(seed)
        n_num = train.x_num.shape[1] if train.x_num is not None else 0
        n_cat = train.x_cat.shape[1] if train.x_cat is not None else 0
        embedder = PLREmbeddings(n_num, n_frequencies=params["n_frequencies"], d_embedding=params["plr_dim"], scale=params["frequency_scale"]) if n_num else None

        return ModernNCAEncoder(
            plr_module=embedder,
            num_continuous=n_num,
            plr_dim=params["plr_dim"],
            num_categorical=n_cat,
            hidden_dim=params["hidden_dim"],
            embed_dim=params["embed_dim"],
            num_layers=params["num_layers"],
            dropout_rate=params["dropout"],
            post_mlp_layers=[],
        )


    def tune_modern_nca(dataset_run, train_run, val_run):
        train_data = (train_run.x_num, train_run.x_cat, train_run.y)
        val_data = (val_run.x_num, val_run.x_cat, val_run.y)
        nca_params = {
            "device": str(device),
            "batch_size": config["batch_size"],
            "patience": config["patience"],
            "verbose": False,
        }

        def nca_objective(trial):
            params = {
                **nca_params,
                # TALENT/configs/opt_space/modernNCA.json; имена параметров нашего энкодера.
                "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                "hidden_dim": trial.suggest_int("hidden_dim", 64, 512),
                "num_layers": trial.suggest_int("num_layers", 0, 2) if trial.suggest_categorical("optional_num_layers", [False, True]) else 0,
                "embed_dim": trial.suggest_int("embed_dim", 64, 512),
                "n_frequencies": trial.suggest_int("n_frequencies", 16, 96),
                "frequency_scale": trial.suggest_float("frequency_scale", 0.005, 10.0, log=True),
                "plr_dim": trial.suggest_int("plr_dim", 16, 64),
                "sample_rate": trial.suggest_float("sample_rate", 0.05, 0.6),
                "lr": trial.suggest_float("lr", 1e-5, 0.1, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True) if trial.suggest_categorical("optional_weight_decay", [False, True]) else 0.0,
            }
            trial.set_user_attr("config", params)
            trainer_trial = Trainer(build_modern_nca(params, config["tune_seed"], train_run), params)
            with progress.measure(dataset_run.dataset_name, "modernnca", phase="optuna", trial=trial.number, seed=config["tune_seed"]) as timing:
                trainer_trial.fit(train_data, config["max_epochs"], params["sample_rate"], dataset_run.task_type, dataset_run.n_classes, val_data)
            trial.set_user_attr("training_seconds", timing["seconds"])
            return trainer_trial.best_score

        study = optuna.create_study(
            direction="minimize" if dataset_run.task_type == "regression" else "maximize",
            sampler=optuna.samplers.TPESampler(seed=config["tune_seed"]),
            storage=optuna_storage,
            study_name=f"{dataset_run.dataset_name}:modernnca:{tuning_tag}",
            load_if_exists=True,
        )
        completed = len(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)))
        progress.register_study(dataset_run.dataset_name, "modernnca", study)
        remaining = max(0, config["n_trials"] - completed)
        print("Optuna", study.study_name, "completed", completed, "remaining", remaining)
        if remaining and not optuna_patience_reached(study, config["optuna_patience"]):
            study.optimize(
                nca_objective,
                n_trials=remaining,
                callbacks=[lambda study, trial: progress.trial_finished(dataset_run.dataset_name, "modernnca", study, trial),
                           lambda study, trial: stop_optuna_if_no_improvement(study, trial, config["optuna_patience"])],
            )
        elif remaining:
            print("Optuna", study.study_name, "already stopped: no improvement for", config["optuna_patience"], "trials")
        progress.tuning_finished(dataset_run.dataset_name, "modernnca")
        return {**study.best_trial.user_attrs["config"], "device": str(device)}


    def run_modern_nca(seed, dataset_run, train_run, val_run, test_run, params):
        train_data = (train_run.x_num, train_run.x_cat, train_run.y)
        val_data = (val_run.x_num, val_run.x_cat, val_run.y)
        model = build_modern_nca(params, seed, train_run)
        trainer = Trainer(model, params)
        with progress.measure(dataset_run.dataset_name, "modernnca", seed=seed):
            trainer.fit(train_data, config["max_epochs"], params["sample_rate"], dataset_run.task_type, dataset_run.n_classes, val_data)
        print(dataset_run.dataset_name, "modernnca", seed, "best epoch", trainer.best_epoch, "epochs trained", trainer.epochs_trained)

        with progress.measure(dataset_run.dataset_name, "modernnca", operation="predict", seed=seed):
            predictions = trainer.predict(
                train_data,
                (test_run.x_num, test_run.x_cat),
                dataset_run.task_type,
                dataset_run.n_classes,
            )

        if dataset_run.task_type in ("binary", "multiclass"):
            predictions = predictions.argmax(dim=1)

        metric_name, metric_value = compute_metric(
            dataset_run.task_type,
            test_run.y.cpu().numpy(),
            predictions.cpu().numpy(),
        )
        print(dataset_run.dataset_name, "modernnca", seed, metric_name, metric_value)
        return {
            "dataset": dataset_run.dataset_name,
            "model": "modernnca",
            "seed": seed,
            "task_type": dataset_run.task_type,
            "metric": metric_name,
            "metric_value": metric_value,
            "config": params.copy(),
        }


    baseline_names = [name for name in config["models"] if name != "modernnca"]
    baseline_configs = {
        "catboost": {"iterations": config["max_boost_rounds"], "early_stopping_rounds": config["early_stopping_rounds"]},
        "xgboost": {"n_estimators": config["max_boost_rounds"], "early_stopping_rounds": config["early_stopping_rounds"]},
        "mlp": {"max_epoch": config["max_epochs"], "patience": config["patience"], "batch_size": config["batch_size"]},
    }
    baseline_configs["catboost"]["thread_count"] = args.cpu_threads
    baseline_configs["xgboost"]["n_jobs"] = args.cpu_threads
    baseline_configs["mlp"]["device"] = str(device)
    baseline_configs["xgboost"].update(device=str(device), tree_method="hist")
    if device.type == "cuda":
        baseline_configs["catboost"].update(task_type="GPU", devices=str(device.index or 0), gpu_ram_part=config["gpu_memory_fraction"], boosting_type="Plain")

    records = []
    result_columns = ["dataset", "model", "seed", "task_type", "metric", "metric_value", "config"]
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    optuna_storage = f"sqlite:///{(results_dir / 'optuna.db').resolve().as_posix()}"
    tuning_settings = {key: value for key, value in config.items() if key not in ("datasets", "models", "seeds", "n_trials", "results_dir", "gpu_memory_fraction", "optuna_patience")}
    tuning_settings["search_space"] = "talent-modernnca-max512-v2"  # Не продолжать старый подбор с другими диапазонами.
    if device.type == "cuda":
        tuning_settings["device"] = device.type  # Подбор CPU и GPU хранится отдельно.
    tuning_tag = hashlib.sha256(json.dumps(tuning_settings, sort_keys=True).encode()).hexdigest()[:12]
    results_path = results_dir / f"benchmark_{args.user}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S_%fZ}.csv"

    with results_path.open("x", encoding="utf-8", newline="") as output:
        csv.DictWriter(output, fieldnames=result_columns).writeheader()
    print("Results", results_path)

    for dataset_name in config["datasets"]:
        for model_name in config["models"]:
            try:
                study = optuna.load_study(study_name=f"{dataset_name}:{model_name}:{tuning_tag}", storage=optuna_storage)
            except KeyError:
                continue
            progress.register_study(dataset_name, model_name, study)


    def save_metric(record):
        with results_path.open("a", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=result_columns)
            writer.writerow({**record, "config": json.dumps(record["config"], ensure_ascii=False)})
        records.append(record)
        progress.final_finished(record["dataset"], record["model"])

    for dataset_name in config["datasets"]:
        print("Dataset", dataset_name)
        with progress.measure(dataset_name, phase="prepare", operation="prepare", seed=config["tune_seed"]):
            dataset_tune, prep_tune, train_tune, val_tune, test_tune = prepare_run(dataset_name, config["tune_seed"])
        best_params = {}

        if "modernnca" in config["models"]:
            best_params["modernnca"] = tune_modern_nca(dataset_tune, train_tune, val_tune)

        for model_name in baseline_names:
            best_params[model_name] = tune_hyperparameters(
                model_name=model_name,
                dataset=dataset_tune,
                seed=config["tune_seed"],
                preprocessor=prep_tune,
                n_trials=config["n_trials"],
                optuna_patience=config["optuna_patience"],
                model_config=baseline_configs[model_name],
                storage=optuna_storage,
                study_name=f"{dataset_name}:{model_name}:{tuning_tag}",
                progress=progress,
            )

        for seed in config["seeds"]:
            with progress.measure(dataset_name, phase="prepare", operation="prepare", seed=seed):
                dataset_run, prep_run, train_run, val_run, test_run = prepare_run(dataset_name, seed)

            if "modernnca" in config["models"]:
                save_metric(run_modern_nca(seed, dataset_run, train_run, val_run, test_run, best_params["modernnca"]))

            for model_name in baseline_names:
                result = run_experiment(
                    model_name=model_name,
                    dataset=dataset_run,
                    seed=seed,
                    config=best_params[model_name],
                    preprocessor=prep_run,
                    progress=progress,
                )
                save_metric({
                    "dataset": result.dataset,
                    "model": result.model,
                    "seed": result.seed,
                    "task_type": result.task_type,
                    "metric": result.metric_name,
                    "metric_value": result.metric_value,
                    "config": result.config.copy(),
                })
                print(result.dataset, result.model, result.seed, result.metric_name, result.metric_value)

    results = pd.DataFrame(records, columns=result_columns)
    print(results[["dataset", "model", "seed", "metric", "metric_value"]])
    progress.close()


if __name__ == "__main__":
    main()
