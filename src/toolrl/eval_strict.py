"""Independent strict evaluator for ToolRL-style tool calls.

The primary metric compares a multiset of calls: line order is ignored while
duplicate calls are retained. JSON values are compared with type information,
so 1, 1.0, true, and "1" are different values.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any


OPEN_TAG = "<tool_call>"
CLOSE_TAG = "</tool_call>"


class ToolCallParseError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ToolCallParseError('duplicate_json_key', f'duplicate key: {key}')
        result[key] = value
    return result


def _reject_constant(value):
    raise ToolCallParseError('invalid_json_constant', f'nonfinite number: {value}')


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        _reject_constant(value)
    return result


def _typed_json(value: Any) -> tuple:
    """Return a hashable, recursively type-sensitive representation."""
    if value is None:
        return ("null",)
    if type(value) is bool:
        return ("bool", value)
    if type(value) is int:
        return ("int", value)
    if type(value) is float:
        return ("float", value)
    if type(value) is str:
        return ("str", value)
    if type(value) is list:
        return ("list", tuple(_typed_json(item) for item in value))
    if type(value) is dict:
        return (
            "dict",
            tuple(sorted((key, _typed_json(item)) for key, item in value.items())),
        )
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}")


def _canonical_call(call: dict[str, Any]) -> tuple:
    return ("call", call["name"], _typed_json(call["parameters"]))


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse exactly one tool_call block containing one JSON object per line."""
    if not isinstance(text, str):
        raise ToolCallParseError("non_string_output", "model output must be a string")
    opens = text.count(OPEN_TAG)
    closes = text.count(CLOSE_TAG)
    if opens == 1 and closes == 0:
        raise ToolCallParseError("truncated_tool_call", "tool_call block is not closed")
    if opens != 1 or closes != 1:
        raise ToolCallParseError(
            "invalid_tool_call_count",
            f"expected one tool_call block, found open={opens}, close={closes}",
        )
    start = text.index(OPEN_TAG) + len(OPEN_TAG)
    end = text.index(CLOSE_TAG)
    if end < start:
        raise ToolCallParseError('invalid_tool_call_order', 'closing tag precedes opening tag')
    body = text[start:end].strip()
    if not body:
        raise ToolCallParseError("empty_tool_call", "tool_call block is empty")

    calls: list[dict[str, Any]] = []
    for line_number, line in enumerate(body.splitlines(), start=1):
        if not line.strip():
            raise ToolCallParseError("invalid_json", f"line {line_number} is blank")
        try:
            call = json.loads(line, object_pairs_hook=_unique_pairs,
                              parse_constant=_reject_constant, parse_float=_finite_float)
        except json.JSONDecodeError as exc:
            raise ToolCallParseError(
                "invalid_json", f"line {line_number}: {exc.msg}"
            ) from exc
        if type(call) is not dict or set(call) != {"name", "parameters"}:
            raise ToolCallParseError(
                "invalid_call_schema",
                f"line {line_number} must contain exactly name and parameters",
            )
        if type(call["name"]) is not str or type(call["parameters"]) is not dict:
            raise ToolCallParseError(
                "invalid_call_schema",
                f"line {line_number} has invalid name/parameters types",
            )
        calls.append(call)
    return calls


def evaluate_prediction(prediction: str, reference: str) -> dict[str, Any]:
    """Evaluate one prediction against a valid reference answer."""
    try:
        reference_calls = parse_tool_calls(reference)
    except ToolCallParseError as exc:
        raise ValueError(f"invalid reference ({exc.code}): {exc}") from exc

    try:
        prediction_calls = parse_tool_calls(prediction)
    except ToolCallParseError as exc:
        return {
            "call_multiset_exact": False,
            "sequence_exact": False,
            "prediction_error": exc.code,
            "prediction_error_detail": str(exc),
            "predicted_call_count": 0,
            "reference_call_count": len(reference_calls),
        }

    predicted = [_canonical_call(call) for call in prediction_calls]
    expected = [_canonical_call(call) for call in reference_calls]
    return {
        "call_multiset_exact": Counter(predicted) == Counter(expected),
        "sequence_exact": predicted == expected,
        "prediction_error": None,
        "prediction_error_detail": None,
        "predicted_call_count": len(predicted),
        "reference_call_count": len(expected),
    }


def evaluate_jsonl(input_path: Path, output_path: Path) -> dict[str, Any]:
    details = []
    for line_number, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        reference = row.get("reference", row.get("ground_truth"))
        if reference is None or "prediction" not in row:
            raise ValueError(f"line {line_number} needs prediction and reference/ground_truth")
        result = evaluate_prediction(row["prediction"], reference)
        details.append({"id": row.get("id", line_number), **result})
    count = len(details)
    summary = {
        "samples": count,
        "call_multiset_exact_rate": sum(x["call_multiset_exact"] for x in details) / count if count else 0.0,
        "sequence_exact_rate": sum(x["sequence_exact"] for x in details) / count if count else 0.0,
        "parse_failure_count": sum(x["prediction_error"] is not None for x in details),
        "details": details,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSONL with id, prediction, reference")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = evaluate_jsonl(args.input, args.output)
    print(json.dumps({k: v for k, v in summary.items() if k != "details"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
