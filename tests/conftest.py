import pytest

from mltest.core.config import AppConfig, ModelManager


@pytest.fixture
def config(tmp_path):
    """A fresh AppConfig singleton writing into a temporary folder, with small budgets."""
    AppConfig.reset_instance()
    ModelManager.reset_instance()
    cfg = AppConfig()
    cfg.update({
        "output_dir": str(tmp_path),
        "observers": ["report"],
        "test_generation": {"budget": 60, "programs": ["triangle", "insertion_sort"]},
        "defect_prediction": {"loader_params": {"n_samples": 400},
                              "models": ["decision_tree", "logistic_regression"]},
        "test_oracle": {"n_inputs": 30, "programs": ["triangle", "gcd"]},
    })
    yield cfg
    AppConfig.reset_instance()
    ModelManager.reset_instance()
