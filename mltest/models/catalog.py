"""Built-in models registered with ModelFactory.

To add a model, register a builder that returns a ModelAdapter:

    @ModelFactory.register("xgboost")
    def _xgboost(seed=42, **params):
        return SklearnAdapter(XGBClassifier(random_state=seed, **params), name="XGBoost")
"""
from __future__ import annotations

from functools import partial

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from mltest.core.factory import ModelFactory
from mltest.models.adapters import PyTorchAdapter, SklearnAdapter, build_torch_mlp


@ModelFactory.register("decision_tree")
def _decision_tree(seed: int = 42, **params) -> SklearnAdapter:
    options = {"max_depth": 8, "min_samples_leaf": 2, "class_weight": "balanced", "random_state": seed}
    return SklearnAdapter(DecisionTreeClassifier(**{**options, **params}), name="DecisionTree")


@ModelFactory.register("svm")
def _svm(seed: int = 42, **params) -> SklearnAdapter:
    # probability=False on purpose: SklearnAdapter derives probabilities from decision scores.
    options = {"kernel": "rbf", "C": 1.0, "gamma": "scale", "class_weight": "balanced", "random_state": seed}
    return SklearnAdapter(SVC(**{**options, **params}), name="SVM")


@ModelFactory.register("random_forest")
def _random_forest(seed: int = 42, **params) -> SklearnAdapter:
    options = {"n_estimators": 200, "min_samples_leaf": 2, "class_weight": "balanced_subsample", "random_state": seed}
    return SklearnAdapter(RandomForestClassifier(**{**options, **params}), name="RandomForest")


@ModelFactory.register("logistic_regression")
def _logistic_regression(seed: int = 42, **params) -> SklearnAdapter:
    options = {"max_iter": 2000, "class_weight": "balanced", "random_state": seed}
    return SklearnAdapter(LogisticRegression(**{**options, **params}), name="LogisticRegression")


@ModelFactory.register("naive_bayes")
def _naive_bayes(seed: int = 42, **params) -> SklearnAdapter:
    return SklearnAdapter(GaussianNB(**params), name="NaiveBayes")


@ModelFactory.register("knn")
def _knn(seed: int = 42, **params) -> SklearnAdapter:
    return SklearnAdapter(KNeighborsClassifier(**{"n_neighbors": 7, **params}), name="KNN")


@ModelFactory.register("neural_network")
def _neural_network(seed: int = 42, **params) -> SklearnAdapter:
    options = {"hidden_layer_sizes": (64, 32), "max_iter": 600, "early_stopping": True, "random_state": seed}
    return SklearnAdapter(MLPClassifier(**{**options, **params}), name="NeuralNetwork")


@ModelFactory.register("torch_mlp")
def _torch_mlp(seed: int = 42, hidden=(64, 32), **params) -> PyTorchAdapter:
    return PyTorchAdapter(partial(build_torch_mlp, hidden=tuple(hidden)), seed=seed, name="TorchMLP", **params)
