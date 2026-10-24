"""Shared adversarial cases for strict evaluation and upstream reward audits."""

from __future__ import annotations

import json


def render(calls: list[dict], closed: bool = True) -> str:
    body = "\n".join(json.dumps(call, ensure_ascii=False) for call in calls)
    end = "\n</tool_call>" if closed else ""
    return f"<think>contract test</think>\n<tool_call>\n{body}{end}"


REFERENCE_CALLS = [
    {"name": "search", "parameters": {"query": "cats", "limit": 1}},
    {"name": "weather", "parameters": {"city": "Paris"}},
]
REFERENCE = render(REFERENCE_CALLS)

CASES = [
    {
        "name": "correct_call",
        "prediction": render(REFERENCE_CALLS),
        "multiset_exact": True,
        "sequence_exact": True,
        "error_code": None,
        "reward_expectation": "maximum",
    },
    {
        "name": "wrong_tool",
        "prediction": render(
            [
                {"name": "lookup", "parameters": {"query": "cats", "limit": 1}},
                REFERENCE_CALLS[1],
            ]
        ),
        "multiset_exact": False,
        "sequence_exact": False,
        "error_code": None,
        "reward_expectation": "below_maximum",
    },
    {
        "name": "missing_parameter",
        "prediction": render(
            [
                {"name": "search", "parameters": {"query": "cats"}},
                REFERENCE_CALLS[1],
            ]
        ),
        "multiset_exact": False,
        "sequence_exact": False,
        "error_code": None,
        "reward_expectation": "below_maximum",
    },
    {
        "name": "wrong_value",
        "prediction": render(
            [
                {"name": "search", "parameters": {"query": "dogs", "limit": 1}},
                REFERENCE_CALLS[1],
            ]
        ),
        "multiset_exact": False,
        "sequence_exact": False,
        "error_code": None,
        "reward_expectation": "below_maximum",
    },
    {
        "name": "wrong_value_type",
        "prediction": render(
            [
                {"name": "search", "parameters": {"query": "cats", "limit": "1"}},
                REFERENCE_CALLS[1],
            ]
        ),
        "multiset_exact": False,
        "sequence_exact": False,
        "error_code": None,
        "reward_expectation": "below_maximum",
    },
    {
        "name": "extra_parameter",
        "prediction": render(
            [
                {
                    "name": "search",
                    "parameters": {"query": "cats", "limit": 1, "unknown": "extra"},
                },
                REFERENCE_CALLS[1],
            ]
        ),
        "multiset_exact": False,
        "sequence_exact": False,
        "error_code": None,
        "reward_expectation": "below_maximum",
    },
    {
        "name": "extra_call",
        "prediction": render(
            REFERENCE_CALLS + [{"name": "clock", "parameters": {}}]
        ),
        "multiset_exact": False,
        "sequence_exact": False,
        "error_code": None,
        "reward_expectation": "below_maximum",
    },
    {
        "name": "duplicate_call",
        "prediction": render(REFERENCE_CALLS + [REFERENCE_CALLS[0]]),
        "multiset_exact": False,
        "sequence_exact": False,
        "error_code": None,
        "reward_expectation": "below_maximum",
    },
    {
        "name": "parallel_order_swap",
        "prediction": render(list(reversed(REFERENCE_CALLS))),
        "multiset_exact": True,
        "sequence_exact": False,
        "error_code": None,
        "reward_expectation": "maximum",
    },
    {
        "name": "invalid_json",
        "prediction": '<think>x</think>\n<tool_call>\n{"name":"search","parameters":}\n</tool_call>',
        "multiset_exact": False,
        "sequence_exact": False,
        "error_code": "invalid_json",
        "reward_expectation": "correctness_minimum",
    },
    {
        "name": "truncated_output",
        "prediction": render(REFERENCE_CALLS, closed=False),
        "multiset_exact": False,
        "sequence_exact": False,
        "error_code": "truncated_tool_call",
        "reward_expectation": "correctness_minimum",
    },
    {
        "name": "two_tool_call_blocks",
        "prediction": render(REFERENCE_CALLS)
        + "\n"
        + render([{"name": "clock", "parameters": {}}]),
        "multiset_exact": False,
        "sequence_exact": False,
        "error_code": "invalid_tool_call_count",
        "reward_expectation": "format_invalid_correctness_maximum",
    },
]
