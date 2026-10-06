import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from typing import Literal



# <<< Tabular dataset class declaration >>>
class TabularDataset:
    X_train: pd.DataFrame
    y_train: pd.Series

    X_val: pd.DataFrame
    y_val: pd.Series

    X_test: pd.DataFrame
    y_test: pd.Series

    n_classes: int

    numerical_columns: list[str]
    categorical_columns: list[str]

    task_type: Literal[
        "binary",
        "multiclass",
        "regression",
    ]

    dataset_name: str



    def __init__ (
        self, dataset_name,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        n_classes,
        numerical_columns, categorical_columns,
        task_type
    ) :
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.n_classes = n_classes
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.task_type = task_type
        self.dataset_name = dataset_name
# <<< Tabular dataset class declaration >>> 




def extract_from_raw (dataset_dir) :
    '''
      Extracts dataset files from @dataset_dir@
      Merges pre-defined train-test split back into a unified dataset
      Converts to a list of 
      [ 
        X (unified dataset): pd.Dataframe,
        y (unified target values): pd.Series,
        metadata: {}
      ]
    '''

# <<< Dataset files extraction >>>
    def load_optional (name):
        path = dataset_dir / name
        return np.load(path, allow_pickle=True) if path.exists() else None

    N_train = load_optional("N_train.npy")
    N_val = load_optional("N_val.npy")
    N_test = load_optional("N_test.npy")

    C_train = load_optional("C_train.npy")
    C_val = load_optional("C_val.npy")
    C_test = load_optional("C_test.npy")

    y_train = np.load(dataset_dir / "y_train.npy", allow_pickle=True)
    y_val = np.load(dataset_dir / "y_val.npy", allow_pickle=True)
    y_test = np.load(dataset_dir / "y_test.npy", allow_pickle=True)

    with open(dataset_dir / "info.json") as info:
        metadata = json.load(info)

    metadata.setdefault("num_feature_intro", {
        f"N_{j}": f"N_{j}"
        for j in range(N_train.shape[1] if N_train is not None else 0)
    })
    metadata.setdefault("cat_feature_intro", {
        f"C_{j}": f"C_{j}"
        for j in range(C_train.shape[1] if C_train is not None else 0)
    })
# <<< Dataset files extraction >>>

# <<< Merging train-test split and numerical/categorical columns >>>
    parts = []

    if N_train is not None:
        parts.append(pd.DataFrame(
            np.concatenate([N_train, N_val, N_test], axis=0),
            columns=metadata["num_feature_intro"]
        ))

    if C_train is not None:
        parts.append(pd.DataFrame(
            np.concatenate([C_train, C_val, C_test], axis=0),
            columns=metadata["cat_feature_intro"]
        ))

    targets = np.concatenate([y_train, y_val, y_test], axis=0).reshape(-1)
# <<< Merging train-test split and numerical/categorical columns >>>

    return pd.concat(parts, axis=1), pd.Series(targets, name="target"), metadata



def load_dataset (dataset_name, seed) :
    '''
      Loads dataset @dataset_name@
      Splits 64%-16%-20% with a given seed
      Returns an object of TabularDataset type
    '''

    X, y, metadata = extract_from_raw(Path("./data") / dataset_name)

    task_type = {"binclass": "binary"}.get(metadata["task_type"], metadata["task_type"])

    if task_type in ["binary", "multiclass"]:
        # 64% train, 36% temporary
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X,
            y,
            test_size=0.36,
            random_state=seed,
            stratify=y,
        )

        # 36% temporary -> 16% validation, 20% test
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp,
            y_tmp,
            test_size=20 / 36,
            random_state=seed,
            stratify=y_tmp,
        )
    else:
        # 64% train, 36% temporary
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X,
            y,
            test_size=0.36,
            random_state=seed,
        )

        # 36% temporary -> 16% validation, 20% test
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp,
            y_tmp,
            test_size=20 / 36,
            random_state=seed,
        )

    n_classes = 0
    if task_type in ("binary", "multiclass"):
        label_encoder = LabelEncoder().fit(y_train)
        y_train, y_val, y_test = [
            pd.Series(label_encoder.transform(part), index=part.index, name="target")
            for part in (y_train, y_val, y_test)
        ]
        n_classes = len(label_encoder.classes_)

    return TabularDataset(
        dataset_name,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        n_classes=n_classes,
        numerical_columns=list(metadata["num_feature_intro"].keys()),
        categorical_columns=list(metadata["cat_feature_intro"].keys()),
        task_type=task_type
    )
