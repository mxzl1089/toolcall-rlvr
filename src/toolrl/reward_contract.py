"""Side-effect-contained adapter for the read-only upstream ToolRL reward."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
from pathlib import Path
from typing import Any


DEFAULT_REWARD_PATH = Path(
    r"data/ToolRL/verl/utils/reward_score/rlla.py"
)
REWARD_FLAGS = (
    "WITHLENGTH",
    "SCHEDULELENGTH",
    "CORRECTMAX1",
    "MAX1STEP30MAX3",
    "SCHEDULEREWARD",
    "REFINEDREWARD",
    "INTERMEDIATEREWARD",
    "COARSEREWARD",
)


def _load_reward_module(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("toolrl_upstream_reward_contract", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reward module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_upstream_reward(
    prediction: str,
    reference: str,
    reward_path: Path | str = DEFAULT_REWARD_PATH,
) -> dict[str, float]:
    """Score with the unmodified default upstream reward and restore env state."""
    keys = ("EXPERIMENT_NAME", *REWARD_FLAGS)
    old = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["EXPERIMENT_NAME"] = "qwen_gate0_contract"
        for key in REWARD_FLAGS:
            os.environ.pop(key, None)
        module = _load_reward_module(Path(reward_path))
        with contextlib.redirect_stdout(io.StringIO()):
            total, format_score, correctness, length = module.compute_score(
                prediction, reference, step=0
            )
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return {
        "total": float(total),
        "format": float(format_score),
        "correctness": float(correctness),
        "length": float(length),
    }
