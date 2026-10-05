from dataclasses import dataclass

import torch
import torch.nn as nn

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


@dataclass
class ProcessedData:
    x_num: torch.Tensor | None
    x_cat: torch.Tensor | None
    y: torch.Tensor


class TabularPreprocessor:
    def __init__(self, numerical_columns, categorical_columns):
        self.num_cols = numerical_columns
        self.cat_cols = categorical_columns

        self.num_imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
        self.cat_imputer = SimpleImputer(strategy="constant", fill_value="__missing__")

        self.scaler = StandardScaler()
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def fit(self, X):
        if self.num_cols:
            x = self.num_imputer.fit_transform(X[self.num_cols])
            self.scaler.fit(x)

        if self.cat_cols:
            x = self.cat_imputer.fit_transform(X[self.cat_cols].astype(str).where(X[self.cat_cols].notna()))
            self.ohe.fit(x)

    def transform(self, X, y):
        x_num = None
        x_cat = None

        if self.num_cols:
            x_num = self.num_imputer.transform(X[self.num_cols])
            x_num = self.scaler.transform(x_num)
            x_num = torch.tensor(x_num, dtype=torch.float32)

        if self.cat_cols:
            x_cat = self.cat_imputer.transform(X[self.cat_cols].astype(str).where(X[self.cat_cols].notna()))
            x_cat = self.ohe.transform(x_cat)
            x_cat = torch.tensor(x_cat, dtype=torch.float32)

        y = torch.tensor(y.to_numpy())

        return ProcessedData(x_num, x_cat, y)


class PLREmbeddings(nn.Module):
    def __init__(self, n_features, n_frequencies=77, d_embedding=34, scale=0.01):
        super().__init__()
        self.freq = nn.Parameter(torch.randn(n_features, n_frequencies) * scale) # xavier
        self.linear = nn.Linear(2 * n_frequencies, d_embedding)

    def forward(self, x_num):
        x = 2 * torch.pi * x_num[..., None] * self.freq
        x = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        return torch.relu(self.linear(x))
