"""
Core regression tests for the thesis codebase.

Run from the project root:
    pytest tests/test_core.py -v

Covers the synthetic data loader, the MLP wrapper, the sklearn / NN / TabPFN
PGD attacks, the transfer-attack glue, and the small helpers in
helpers/attack_helpers.py and helpers/model_helpers.py.

TabPFN inference is slow on CPU, so the TabPFN-specific tests use a tiny
dataset and a single PGD restart. They are still marked `slow` so they can
be skipped with `pytest -m "not slow"`.
"""

import dataclasses

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from tabpfn import ModelVersion, TabPFNClassifier

from helpers.AttackResult import AttackResult
from helpers.attack_helpers import _to_numpy_1d, _to_python_label
from helpers.data_handlers import create_synthetic_dataset
from helpers.enable_grad_raw_predict import enable_grad_raw_predict, raw_predict_with_grad
from helpers.mlp import SimpleNN, StandardMLPClassifier
from helpers.model_helpers import _predict_target, _train_models
from helpers.pgd_attack import pgd_attack_nn, pgd_attack_sklearn, pgd_linf_restarts
from helpers.transfer_attack import transfer_attack


@pytest.fixture(scope="module")
def synthetic_data():
    return create_synthetic_dataset(
        n_samples=80, n_features=6, test_size=0.25, random_state=0
    )


@pytest.fixture(scope="module")
def tiny_data():
    return create_synthetic_dataset(
        n_samples=20, n_features=4, test_size=0.25, random_state=0
    )


@pytest.fixture(scope="module")
def trained_logreg(synthetic_data):
    X_train, y_train, _, _ = synthetic_data
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train.cpu().numpy(), y_train.cpu().numpy())
    return clf


@pytest.fixture(scope="module")
def trained_mlp(synthetic_data):
    X_train, y_train, _, _ = synthetic_data
    clf = StandardMLPClassifier(epochs=30, lr=1e-2)
    clf.fit(X_train, y_train)
    return clf


@pytest.fixture(scope="module")
def trained_tabpfn(tiny_data):
    X_train, y_train, _, _ = tiny_data
    clf = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2, differentiable_input=True
    )
    clf.fit(X_train, y_train)
    enable_grad_raw_predict(clf, raw_predict_with_grad)
    return clf


class TestAttackResult:
    def test_is_frozen(self):
        r = AttackResult(
            x_adv=torch.zeros(1, 3), best_loss=0.0, pred_adv=0, pred_nat=0, y_true=0
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.best_loss = 1.0  # type: ignore[misc]

    def test_fields_round_trip(self):
        x = torch.tensor([[1.0, 2.0]])
        r = AttackResult(x_adv=x, best_loss=0.5, pred_adv=1, pred_nat=0, y_true=1)
        assert torch.equal(r.x_adv, x)
        assert r.best_loss == 0.5
        assert r.pred_adv == 1 and r.pred_nat == 0 and r.y_true == 1


class TestAttackHelpers:
    def test_to_python_label_int(self):
        assert _to_python_label(3) == 3

    def test_to_python_label_tensor_scalar(self):
        assert _to_python_label(torch.tensor(7)) == 7

    def test_to_numpy_1d_from_tensor(self):
        out = _to_numpy_1d(torch.tensor([[1.0, 2.0, 3.0]]))
        assert isinstance(out, np.ndarray)
        assert out.shape == (3,)
        assert out.dtype == np.float64
        np.testing.assert_allclose(out, [1.0, 2.0, 3.0])

    def test_to_numpy_1d_from_list(self):
        out = _to_numpy_1d([1, 2, 3])
        assert out.shape == (3,)
        np.testing.assert_allclose(out, [1.0, 2.0, 3.0])


class TestSyntheticDataset:
    def test_shapes_and_dtypes(self, synthetic_data):
        X_train, y_train, X_test, y_test = synthetic_data
        assert X_train.shape == (60, 6)
        assert X_test.shape == (20, 6)
        assert y_train.shape == (60,)
        assert y_test.shape == (20,)
        assert X_train.dtype == torch.float32
        assert y_train.dtype == torch.long

    def test_standardized(self, synthetic_data):
        X_train, _, _, _ = synthetic_data
        assert torch.allclose(X_train.mean(dim=0), torch.zeros(6), atol=1e-5)
        assert torch.allclose(X_train.std(dim=0, unbiased=False), torch.ones(6), atol=1e-5)

    def test_determinism(self):
        a = create_synthetic_dataset(n_samples=40, n_features=4, random_state=123)
        b = create_synthetic_dataset(n_samples=40, n_features=4, random_state=123)
        for ta, tb in zip(a, b):
            assert torch.equal(ta, tb)


class TestSimpleNN:
    def test_forward_shape(self):
        net = SimpleNN(input_dim=5)
        out = net(torch.randn(4, 5))
        assert out.shape == (4, 2)

    def test_predict_proba_rows_sum_to_one(self):
        net = SimpleNN(input_dim=4)
        probs = net.predict_proba(torch.randn(3, 4))
        assert probs.shape == (3, 2)
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(3), atol=1e-5)

    def test_predict_labels_are_binary(self):
        net = SimpleNN(input_dim=4)
        preds = net.predict(torch.randn(5, 4))
        assert preds.shape == (5,)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_standard_mlp_fits_separable_data(self, synthetic_data, trained_mlp):
        _, _, X_test, y_test = synthetic_data
        preds = trained_mlp.predict(X_test)
        acc = (preds == y_test.cpu().numpy()).mean()
        assert acc >= 0.75


EPS = 0.25


def _linf(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


class TestPGDSklearn:
    def test_perturbation_within_eps_ball(self, synthetic_data, trained_logreg):
        _, _, X_test, y_test = synthetic_data
        x, y = X_test[0], y_test[0]

        result = pgd_attack_sklearn(
            trained_logreg, x, y, eps=EPS, steps=10, restarts=3, seed=0
        )

        assert isinstance(result, AttackResult)
        assert result.x_adv.shape == (1, X_test.shape[1])
        assert _linf(result.x_adv.squeeze(0), x) <= EPS + 1e-6
        assert result.y_true == int(y.item())

    def test_loss_does_not_decrease_with_attack(self, synthetic_data, trained_logreg):
        _, _, X_test, y_test = synthetic_data
        x, y = X_test[0], y_test[0]
        r0 = pgd_attack_sklearn(trained_logreg, x, y, eps=0.0, steps=1, restarts=1, seed=0)
        r1 = pgd_attack_sklearn(trained_logreg, x, y, eps=EPS, steps=10, restarts=3, seed=0)
        assert r1.best_loss >= r0.best_loss - 1e-6


class TestPGDNN:
    def test_perturbation_within_eps_ball(self, synthetic_data, trained_mlp):
        _, _, X_test, y_test = synthetic_data
        x, y = X_test[0], y_test[0]

        result = pgd_attack_nn(
            trained_mlp._model, x, y, eps=EPS, steps=10, restarts=2
        )

        assert isinstance(result, AttackResult)
        assert _linf(result.x_adv.squeeze(0), x) <= EPS + 1e-6
        assert result.pred_nat in (0, 1)
        assert result.pred_adv in (0, 1)


@pytest.mark.slow
class TestPGDTabPFN:
    def test_perturbation_within_eps_ball(self, tiny_data, trained_tabpfn):
        _, _, X_test, y_test = tiny_data
        x, y = X_test[0], y_test[0]

        result = pgd_linf_restarts(
            trained_tabpfn, x, y, eps=EPS, steps=3, restarts=1, seed=0
        )

        assert isinstance(result, AttackResult)
        assert result.x_adv.shape[-1] == X_test.shape[1]
        assert _linf(result.x_adv.squeeze(0), x) <= EPS + 1e-6
        assert result.y_true == int(y.item())

    def test_loss_does_not_decrease_with_attack(self, tiny_data, trained_tabpfn):
        _, _, X_test, y_test = tiny_data
        x, y = X_test[0], y_test[0]
        r0 = pgd_linf_restarts(trained_tabpfn, x, y, eps=0.0, steps=1, restarts=1, seed=0)
        r1 = pgd_linf_restarts(trained_tabpfn, x, y, eps=EPS, steps=3, restarts=1, seed=0)
        assert r1.best_loss >= r0.best_loss - 1e-6


class TestPredictTarget:
    def test_sklearn_path(self, synthetic_data, trained_logreg):
        _, _, X_test, _ = synthetic_data
        x = X_test[0]
        pred = _predict_target(("logreg", trained_logreg), x)
        expected = int(trained_logreg.predict(x.cpu().numpy().reshape(1, -1))[0])
        assert pred == expected

    @pytest.mark.slow
    def test_tabpfn_path(self, tiny_data, trained_tabpfn):
        _, _, X_test, _ = tiny_data
        pred = _predict_target(("TabPFN", trained_tabpfn), X_test[0])
        assert pred in (0, 1)

    @pytest.mark.slow
    def test_train_models_dispatches_correctly(self, tiny_data):
        X_train, y_train, _, _ = tiny_data
        tabpfn = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2, differentiable_input=True
        )
        logreg = LogisticRegression(max_iter=500)
        models = [("TabPFN", tabpfn, None, None), ("logreg", logreg, None, None)]
        _train_models(models, X_train, y_train)
        # Both should be fitted and produce valid predictions.
        assert hasattr(tabpfn, "_raw_predict")
        assert _predict_target(("TabPFN", tabpfn), X_train[0]) in (0, 1)
        assert _predict_target(("logreg", logreg), X_train[0]) in (0, 1)


class TestTransferAttack:
    def test_eps_zero_returns_natural_prediction(self, synthetic_data, trained_logreg, trained_mlp):
        _, _, X_test, y_test = synthetic_data
        x, y = X_test[0], y_test[0]

        result = transfer_attack(
            x, y,
            eps=0.0,
            source_clf=trained_logreg,
            source_attack_fn=pgd_attack_sklearn,
            source_attack_args={"steps": 5, "restarts": 1, "seed": 0},
            target_clf=trained_mlp._model,
            target_name="mlp",
        )
        assert result.best_loss == 0.0
        assert result.pred_adv == result.pred_nat
        assert torch.equal(result.x_adv, x)

    def test_transfers_within_eps_ball(self, synthetic_data, trained_logreg, trained_mlp):
        _, _, X_test, y_test = synthetic_data
        x, y = X_test[0], y_test[0]

        result = transfer_attack(
            x, y,
            eps=EPS,
            source_clf=trained_logreg,
            source_attack_fn=pgd_attack_sklearn,
            source_attack_args={"steps": 10, "restarts": 3, "seed": 0},
            target_clf=trained_mlp._model,
            target_name="mlp",
        )
        x_adv = result.x_adv.squeeze(0).to(dtype=x.dtype)
        assert _linf(x_adv, x) <= EPS + 1e-6

    @pytest.mark.slow
    def test_transfer_logreg_to_tabpfn(self, tiny_data, trained_tabpfn):
        X_train, y_train, X_test, y_test = tiny_data
        logreg = LogisticRegression(max_iter=500)
        logreg.fit(X_train.cpu().numpy(), y_train.cpu().numpy())

        result = transfer_attack(
            X_test[0], y_test[0],
            eps=EPS,
            source_clf=logreg,
            source_attack_fn=pgd_attack_sklearn,
            source_attack_args={"steps": 5, "restarts": 1, "seed": 0},
            target_clf=trained_tabpfn,
            target_name="TabPFN",
        )
        x_adv = result.x_adv.squeeze(0).to(dtype=X_test.dtype)
        assert _linf(x_adv, X_test[0]) <= EPS + 1e-6
        assert result.pred_adv in (0, 1)
        assert result.pred_nat in (0, 1)
