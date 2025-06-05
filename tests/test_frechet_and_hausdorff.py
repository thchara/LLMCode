import sys
import types
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

# Prepare lightweight llmcode package to avoid heavy dependencies
package = types.ModuleType("llmcode")
package.__path__ = [str(Path(__file__).resolve().parents[1] / "llmcode")]
sys.modules.setdefault("llmcode", package)
llms_mod = types.ModuleType("llmcode.llms")
llms_mod.embed = lambda texts, use_cache=True, model=None: np.zeros((len(texts), 3))
sys.modules.setdefault("llmcode.llms", llms_mod)
coding_mod = types.ModuleType("llmcode.coding")
coding_mod.parse_codes = lambda df: df
sys.modules.setdefault("llmcode.coding", coding_mod)

spec = importlib.util.spec_from_file_location(
    "llmcode.metrics",
    Path(__file__).resolve().parents[1] / "llmcode" / "metrics.py",
)
metrics = importlib.util.module_from_spec(spec)
sys.modules["llmcode.metrics"] = metrics
spec.loader.exec_module(metrics)

from llmcode.metrics import (
    frechet_embedding_distance,
    hausdorff_embedding_distance,
)


@pytest.mark.parametrize(
    "metric",
    [frechet_embedding_distance, hausdorff_embedding_distance],
)
def test_self_agreement(metric):
    np.random.seed(0)
    A = np.random.rand(4, 3)
    assert_allclose(metric(A.copy(), A.copy()), 0.0, atol=1e-8)


@pytest.mark.parametrize(
    "metric",
    [frechet_embedding_distance, hausdorff_embedding_distance],
)
def test_symmetry(metric):
    np.random.seed(0)
    A = np.random.rand(3, 3)
    B = np.random.rand(5, 3)
    assert_allclose(metric(A, B), metric(B, A), atol=1e-6)


@pytest.mark.parametrize(
    "metric,A,B,expected",
    [
        (
            frechet_embedding_distance,
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([[-1.0, 0.0], [0.0, -1.0]]),
            2.0,
        ),
        (
            hausdorff_embedding_distance,
            np.array([[1.0, 0.0]]),
            np.array([[0.0, 1.0]]),
            1.0,
        ),
    ],
)
def test_edge_cases(metric, A, B, expected):
    assert_allclose(metric(A, B), expected, atol=1e-6)
