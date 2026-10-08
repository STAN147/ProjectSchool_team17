"""Load one best ModernNCA hyperparameter configuration per dataset."""

import json
from pathlib import Path


BEST_PARAMS = Path(__file__).resolve().parent / "modernnca_best_params.json"


def load_modernnca_config(dataset_name, path=BEST_PARAMS):
    configs = json.loads(Path(path).read_text(encoding="utf-8"))
    if dataset_name not in configs:
        raise ValueError(f"Saved ModernNCA parameters not found for dataset: {dataset_name}")
    config = configs[dataset_name]
    required = {"hidden_dim", "embed_dim", "num_layers", "dropout", "n_frequencies", "frequency_scale", "plr_dim", "sample_rate", "lr", "weight_decay"}
    if not required <= config.keys():
        raise ValueError(f"Incomplete ModernNCA config: {dataset_name}; missing {sorted(required - config.keys())}")
    return config
