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

from llmcode.metrics import KNN_precision_and_recall


@pytest.mark.parametrize("k", [2])
def test_self_agreement(k):
    np.random.seed(0)
    A = np.random.rand(5, 3)
    precision, recall = KNN_precision_and_recall(A, A, k=k)
    assert_allclose(precision, 1.0)
    assert_allclose(recall, 1.0)


@pytest.mark.parametrize("k", [2])
def test_symmetry(k):
    np.random.seed(0)
    A = np.random.rand(5, 3)
    B = np.random.rand(5, 3)
    precision1, recall1 = KNN_precision_and_recall(A, B, k=k)
    precision2, recall2 = KNN_precision_and_recall(B, A, k=k)
    assert_allclose(precision1, recall2, atol=1e-6)
    assert_allclose(recall1, precision2, atol=1e-6)


@pytest.mark.parametrize(
    "A,B,k,expected",
    [
        (np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]), 1, (0.0, 0.0)),
    ],
)
def test_edge_case(A, B, k, expected):
    result = KNN_precision_and_recall(A, B, k=k)
    assert_allclose(result, expected)
