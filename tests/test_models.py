"""Adapter, Ensemble and the model catalog."""
import importlib.util

import numpy as np
import pytest
from sklearn.datasets import make_classification

from mltest.core.factory import ModelFactory
from mltest.models.adapters import ModelAdapter
from mltest.models.ensemble import EnsemblePredictor, majority_vote

X, y = make_classification(n_samples=200, n_features=6, random_state=0)
HAS_TORCH = importlib.util.find_spec("torch") is not None


@pytest.mark.parametrize("model_type", ModelFactory.available())
def test_every_model_speaks_the_adapter_interface(model_type):
    if model_type == "torch_mlp" and not HAS_TORCH:
        pytest.skip("torch not installed")
    params = {"epochs": 5} if model_type == "torch_mlp" else {}
    model = ModelFactory.create_model(model_type, seed=0, **params).fit(X, y)
    assert isinstance(model, ModelAdapter)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(model.predict(X)) <= {0, 1}
    assert model.positive_scores(X).shape == (len(X),)


def test_svm_adapter_derives_probabilities_from_decision_scores():
    svm = ModelFactory.create_model("svm").fit(X, y)
    assert not hasattr(svm.estimator, "predict_proba")
    assert np.all((svm.predict_proba(X) >= 0) & (svm.predict_proba(X) <= 1))


@pytest.mark.parametrize("voting", ["soft", "hard"])
def test_ensemble_combines_members(voting):
    members = [ModelFactory.create_model(m) for m in ("decision_tree", "logistic_regression", "naive_bayes")]
    ensemble = EnsemblePredictor(members, voting=voting).fit(X, y)
    assert ensemble.predict(X).shape == (len(X),)
    assert np.allclose(ensemble.predict_proba(X).sum(axis=1), 1.0)
    assert (ensemble.predict(X) == y).mean() > 0.8


def test_hard_vote_follows_the_majority():
    ensemble = EnsemblePredictor([ModelFactory.create_model("knn")] * 3, voting="hard")
    combined = ensemble.combine_predictions([np.array([1, 0]), np.array([1, 1]), np.array([0, 0])])
    assert list(combined) == [1, 0]


def test_majority_vote_weights_and_ties():
    assert majority_vote(["a", "b", "b"]) == "b"
    assert majority_vote(["a", "b", "b"], [5, 1, 1]) == "a"
    assert majority_vote(["x", "y"]) == "x"
