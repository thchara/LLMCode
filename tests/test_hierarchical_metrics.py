import importlib.util
import sys
import types
from pathlib import Path

from numpy.testing import assert_allclose

# Prepare lightweight llmcode package
package = types.ModuleType("llmcode")
package.__path__ = [str(Path(__file__).resolve().parents[1] / "llmcode")]
sys.modules.setdefault("llmcode", package)

spec = importlib.util.spec_from_file_location(
    "llmcode.metrics", Path(__file__).resolve().parents[1] / "llmcode" / "metrics.py"
)
metrics = importlib.util.module_from_spec(spec)
sys.modules["llmcode.metrics"] = metrics
spec.loader.exec_module(metrics)

from llmcode.metrics import cohen_kappa_hier


def test_full_match():
    a = ["Parent > Child1", "Other > Sub"]
    b = ["Parent > Child1", "Other > Sub"]
    assert_allclose(cohen_kappa_hier(a, b), 1.0, rtol=1e-6)


def test_sibling_match():
    a = ["Parent > A1", "Parent > A1"]
    b = ["Parent > A2", "Parent > A2"]
    sibling_credit = 0.5
    assert_allclose(
        cohen_kappa_hier(a, b, sibling_credit=sibling_credit),
        sibling_credit,
        rtol=1e-6,
    )


def test_no_commonality():
    a = ["A > A1", "B > B1"]
    b = ["C > C1", "D > D1"]
    assert_allclose(cohen_kappa_hier(a, b), 0.0, rtol=1e-6)
