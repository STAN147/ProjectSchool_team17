"""The shared encoder with a linear head or CatBoost on frozen embeddings."""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from steram5 import build_model, compute_metric, fit_model
from Stream4_Train import Trainer


class EncoderWithLinearHead(nn.Module):
    def __init__(self, encoder, embed_dim, output_dim):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embed_dim, output_dim)

    def forward(self, x_num, x_cat):
        return self.head(self.encoder(x_num, x_cat))


class LinearHeadTrainer:
    def __init__(self, encoder, config, task_type, num_classes):
        if task_type not in ("binary", "multiclass", "regression"):
            raise ValueError(f"Unknown task type: {task_type}")
        self.task_type = task_type
        self.num_classes = num_classes
        self.device = torch.device(config.get("device", "cpu"))
        self.batch_size = config.get("batch_size", 1024)
        self.patience = config.get("patience", 20)
        self.verbose = config.get("verbose", False)
        output_dim = 1 if task_type == "regression" else num_classes
        self.model = EncoderWithLinearHead(encoder, config["embed_dim"], output_dim).to(self.device)
        # Both the encoder and output head receive gradients from the same loss.
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 0.0))

    def _inputs(self, data, indices):
        return tuple(x[indices].to(self.device) if x is not None else None for x in data[:2])

    def _outputs(self, data, embeddings=False):
        self.model.to(self.device).eval()
        size = next(x for x in data[:2] if x is not None).shape[0]
        outputs = []
        with torch.no_grad():
            for start in range(0, size, self.batch_size):
                inputs = self._inputs(data, slice(start, start + self.batch_size))
                output = self.model.encoder(*inputs) if embeddings else self.model(*inputs)
                outputs.append(output.cpu())
        return torch.cat(outputs)

    def predict(self, data):
        output = self._outputs(data)
        return output.squeeze(-1) if self.task_type == "regression" else output.softmax(dim=1)

    def fit(self, train_data, val_data, epochs):
        size = len(train_data[2])
        if size < 2 or self.batch_size < 2:
            raise ValueError("At least two training examples and batch_size >= 2 are required")
        self.model.to(self.device)
        best_state, bad_epochs = None, 0
        self.best_score = float("inf") if self.task_type == "regression" else -float("inf")
        self.best_epoch = self.epochs_trained = 0
        for epoch in range(epochs):
            self.model.train()
            batches = list(torch.randperm(size).split(self.batch_size))
            # Merge a singleton tail so BatchNorm trains on every object.
            if len(batches) > 1 and len(batches[-1]) == 1:
                batches[-2] = torch.cat(batches[-2:])
                batches.pop()
            for indices in batches:
                output = self.model(*self._inputs(train_data, indices))
                target = train_data[2][indices].to(self.device).reshape(-1)
                loss = F.mse_loss(output.squeeze(-1), target.float()) if self.task_type == "regression" else F.cross_entropy(output, target.long())
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite neural head loss")
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.epochs_trained = epoch + 1
            predictions = self.predict(val_data)
            if self.task_type != "regression":
                predictions = predictions.argmax(dim=1)
            _, score = compute_metric(self.task_type, val_data[2].cpu().numpy(), predictions.numpy())
            improved = score < self.best_score if self.task_type == "regression" else score > self.best_score
            if improved:
                self.best_score, self.best_epoch = float(score), epoch + 1
                best_state = copy.deepcopy(self.model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
            if self.verbose:
                print("Epoch", epoch + 1, "val metric", score, flush=True)
            if self.patience is not None and bad_epochs >= self.patience:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self


class CatBoostEmbeddingTrainer:
    def __init__(self, encoder, config, task_type, num_classes, seed):
        self.encoder, self.config = encoder, config
        self.task_type, self.num_classes = task_type, num_classes
        self.device = torch.device(config.get("device", "cpu"))
        self.batch_size = config.get("batch_size", 1024)
        self.catboost_config, self.seed = config["catboost"], seed

    def _embeddings(self, data):
        self.encoder.to(self.device).eval()
        size = next(x for x in data[:2] if x is not None).shape[0]
        outputs = []
        with torch.no_grad():
            for start in range(0, size, self.batch_size):
                inputs = tuple(x[start:start + self.batch_size].to(self.device) if x is not None else None for x in data[:2])
                outputs.append(self.encoder(*inputs).cpu())
        return torch.cat(outputs).numpy()

    def fit_encoder(self, train_data, val_data, epochs):
        # Exactly the ModernNCA training loop, SNS, loss and best checkpoint.
        self.encoder.requires_grad_(True)
        nca = Trainer(self.encoder, self.config)
        nca.fit(train_data, epochs, self.config["sample_rate"], self.task_type, self.num_classes, val_data)
        self.encoder_score, self.best_epoch = nca.best_score, nca.best_epoch
        self.epochs_trained = nca.epochs_trained
        self.encoder.requires_grad_(False)
        train_embeddings = self._embeddings(train_data)
        val_embeddings = self._embeddings(val_data)
        # Release neural GPU allocations before CatBoost training.
        self.encoder.cpu()
        nca.optimizer = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return train_embeddings, val_embeddings

    def fit_head(self, train_embeddings, val_embeddings, y_train, y_val):
        print("CatBoost head: fitting frozen embeddings; encoder epochs", self.epochs_trained, flush=True)
        self.catboost = build_model("catboost", self.task_type, self.catboost_config, self.seed)
        fit_model(self.catboost, "catboost", train_embeddings, val_embeddings,
                  y_train.cpu().numpy(), y_val.cpu().numpy())
        # Optuna evaluates the final hybrid, rather than the NCA pretraining score.
        _, self.best_score = compute_metric(self.task_type, y_val.cpu().numpy(), self.catboost.predict(val_embeddings))
        return self

    def fit(self, train_data, val_data, epochs):
        train_embeddings, val_embeddings = self.fit_encoder(train_data, val_data, epochs)
        return self.fit_head(train_embeddings, val_embeddings, train_data[2], val_data[2])

    def predict(self, data):
        if not hasattr(self, "catboost"):
            raise RuntimeError("Fit the NCA encoder and CatBoost before prediction")
        embeddings = self._embeddings(data)
        if self.task_type == "regression":
            return torch.from_numpy(np.asarray(self.catboost.predict(embeddings)).reshape(-1))
        probabilities = np.zeros((len(embeddings), self.num_classes), dtype=np.float64)
        probabilities[:, np.asarray(self.catboost.classes_, dtype=int)] = self.catboost.predict_proba(embeddings)
        return torch.from_numpy(probabilities)
