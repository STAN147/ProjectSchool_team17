"""Small CPU checks; no datasets or full benchmark runs."""

import copy
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Stream2_Preprocessing import PLREmbeddings
from Stream3_ModernNCA import ModernNCAEncoder
from Stream4_Train import Trainer
from Stream5_NeuralHeads import LinearHeadTrainer, CatBoostEmbeddingTrainer


class NeuralHeadsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_trainer(self, task, hybrid=False, categorical_only=False):
        torch.manual_seed(7)
        config = {
            "device": "cpu", "batch_size": 8, "patience": 2, "embed_dim": 6,
            "lr": .01, "weight_decay": 0, "sample_rate": .3, "verbose": False,
            "catboost": {"iterations": 3, "depth": 2, "thread_count": 1, "early_stopping_rounds": 1},
        }
        encoder = ModernNCAEncoder(
            None if categorical_only else PLREmbeddings(3, n_frequencies=3, d_embedding=4),
            0 if categorical_only else 3, 4, 2, 8, 6, 1, 0.0,
        )
        classes = {"binary": 2, "multiclass": 3, "regression": 1}[task]
        trainer = (CatBoostEmbeddingTrainer(encoder, config, task, classes, 7)
                   if hybrid else LinearHeadTrainer(encoder, config, task, classes))

        def data(size):
            x_num = None if categorical_only else torch.randn(size, 3)
            x_cat = torch.nn.functional.one_hot(torch.arange(size) % 2, num_classes=2).float()
            y = torch.randn(size) if task == "regression" else torch.arange(size) % classes
            return x_num, x_cat, y

        return trainer, data(17), data(9), data(5)

    def test_shapes_probabilities_weights_and_frozen_encoder(self):
        for task in ("binary", "multiclass", "regression"):
            for hybrid in (False, True):
                with self.subTest(task=task, hybrid=hybrid):
                    trainer, train, val, test = self.make_trainer(task, hybrid)
                    encoder = trainer.encoder if hybrid else trainer.model.encoder
                    before_encoder = encoder.network[0].weight.detach().clone()
                    if not hybrid:
                        before_head = trainer.model.head.weight.detach().clone()
                    trainer.fit(train, val, epochs=2)
                    self.assertFalse(torch.equal(before_encoder, encoder.network[0].weight))
                    if not hybrid:
                        self.assertFalse(torch.equal(before_head, trainer.model.head.weight))
                    output = trainer.predict(test)
                    self.assertTrue(torch.isfinite(output).all())
                    if task == "regression":
                        self.assertEqual(tuple(output.shape), (5,))
                        if not hybrid:
                            self.assertEqual(trainer.model.head.out_features, 1)
                    else:
                        self.assertEqual(tuple(output.shape), (5, trainer.num_classes))
                        torch.testing.assert_close(output.sum(1), torch.ones(5, dtype=output.dtype))
                    self.assertGreaterEqual(trainer.best_epoch, 1)
                    if hybrid:
                        self.assertTrue(all(not p.requires_grad for p in encoder.parameters()))
                        frozen = copy.deepcopy(encoder.state_dict())
                        trainer.predict(test)
                        for key, value in frozen.items():
                            torch.testing.assert_close(value, encoder.state_dict()[key])

    def test_hybrid_encoder_matches_original_nca_training(self):
        for task in ("binary", "multiclass", "regression"):
            with self.subTest(task=task):
                hybrid, train, val, _ = self.make_trainer(task, hybrid=True)
                reference = Trainer(copy.deepcopy(hybrid.encoder), hybrid.config)
                torch.manual_seed(123)
                reference.fit(train, 2, hybrid.config["sample_rate"], task, hybrid.num_classes, val)
                torch.manual_seed(123)
                hybrid.fit(train, val, 2)
                self.assertEqual(reference.best_epoch, hybrid.best_epoch)
                self.assertEqual(reference.epochs_trained, hybrid.epochs_trained)
                self.assertEqual(reference.best_score, hybrid.encoder_score)
                for key, value in reference.encoder.state_dict().items():
                    torch.testing.assert_close(value, hybrid.encoder.state_dict()[key], rtol=0, atol=0)

    def test_gradients_reach_encoder_plr_and_head(self):
        trainer, train, _, _ = self.make_trainer("multiclass")
        logits = trainer.model(train[0][:8], train[1][:8])
        torch.nn.functional.cross_entropy(logits, train[2][:8]).backward()
        for parameter in (trainer.model.head.weight,
                          trainer.model.encoder.network[0].weight,
                          trainer.model.encoder.plr_module.freq):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_categorical_only(self):
        trainer, train, val, test = self.make_trainer("binary", hybrid=True, categorical_only=True)
        trainer.fit(train, val, epochs=1)
        self.assertEqual(tuple(trainer.predict(test).shape), (5, 2))

    def test_early_stop_restores_best_state(self):
        trainer, train, val, _ = self.make_trainer("regression")
        trainer.patience = 1
        before = copy.deepcopy(trainer.model.state_dict())
        checkpoint = {}

        def validation_prediction(data):
            if not checkpoint:
                checkpoint.update(copy.deepcopy(trainer.model.state_dict()))
                return data[2].clone()  # RMSE 0 after the first epoch.
            return data[2] + 1  # Worse validation after another real optimizer step.

        trainer.predict = validation_prediction
        trainer.fit(train, val, epochs=10)
        self.assertEqual(trainer.epochs_trained, 2)
        self.assertEqual(trainer.best_epoch, 1)
        self.assertFalse(torch.equal(before["head.weight"], checkpoint["head.weight"]))
        for key, value in checkpoint.items():
            torch.testing.assert_close(value, trainer.model.state_dict()[key])


if __name__ == "__main__":
    unittest.main()
