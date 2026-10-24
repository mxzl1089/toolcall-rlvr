import pytest

from toolrl.contract_cases import CASES, REFERENCE
from toolrl.eval_strict import evaluate_prediction


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_strict_contract(case):
    result = evaluate_prediction(case["prediction"], REFERENCE)
    assert result["call_multiset_exact"] is case["multiset_exact"]
    assert result["sequence_exact"] is case["sequence_exact"]
    assert result["prediction_error"] == case["error_code"]
