"""Verify the frozen ToolRL split without modifying source data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata
from typing import Any

import pyarrow.parquet as pq


DEFAULT_TRAIN = Path(r"data/ToolRL/dataset/rlla_4k/train.parquet")
DEFAULT_TEST = Path(r"data/ToolRL/dataset/rlla_4k/test.parquet")
DEFAULT_MANIFEST = Path(r"docs/prospective_manifest_v3.json")
TOOL_NAME = re.compile(r"^\d+\.\s+Name:\s*(.+?)\s*$", re.MULTILINE)
TOOL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_tool_name(name: str) -> str:
    return unicodedata.normalize("NFKC", name).strip().casefold()


def schema_tool_names(instruction: str) -> set[str]:
    return {normalize_tool_name(name) for name in TOOL_NAME.findall(instruction)}


def parse_label_calls(label: str) -> tuple[list[dict[str, Any]], str | None]:
    open_count = label.count("<tool_call>")
    close_count = label.count("</tool_call>")
    if open_count != close_count or open_count == 0:
        return [], f"tool_call tag mismatch/open_count={open_count}/close_count={close_count}"
    blocks = TOOL_BLOCK.findall(label)
    if len(blocks) != open_count:
        return [], "tool_call blocks are malformed or nested"
    calls: list[dict[str, Any]] = []
    for block_number, block in enumerate(blocks, start=1):
        body = block.strip()
        if not body:
            return [], f"tool_call block {block_number} is empty"
        for line_number, line in enumerate(body.splitlines(), start=1):
            if not line.strip():
                return [], f"block {block_number} line {line_number} is blank"
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                return [], f"block {block_number} line {line_number}: {exc.msg}"
            if type(value) is not dict:
                return [], f"block {block_number} line {line_number} is not an object"
            calls.append(value)
    return calls, None


def _row_id(row: dict[str, Any]) -> int:
    return int(row["extra_info"]["index"])


def verify(
    train_path: Path,
    test_path: Path,
    manifest_path: Path,
    parent_manifest_path: Path | None = None,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lineage = None
    lineage_failures: list[str] = []
    if parent_manifest_path is not None:
        parent = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
        removed = {entry["id"] for entry in manifest.get("removed_ids", [])}
        expected_train = [
            sample_id for sample_id in parent["indices"]["train"] if sample_id not in removed
        ]
        checks = {
            "parent_sha256": manifest.get("parent_sha256") == sha256_file(parent_manifest_path),
            "parent_version": manifest.get("parent_version") == "v1",
            "fixed_before_testing": manifest.get("removal_rationale_fixed_before_testing") is True,
            "seed_unchanged": manifest.get("seed") == parent.get("seed") == 20261013,
            "source_unchanged": manifest.get("source") == parent.get("source"),
            "train_is_parent_minus_removed": manifest.get("indices", {}).get("train") == expected_train,
            "val_byte_for_byte_list_equal": manifest.get("indices", {}).get("val") == parent["indices"]["val"],
            "test_byte_for_byte_list_equal": manifest.get("indices", {}).get("test") == parent["indices"]["test"],
        }
        if manifest.get('version')=='v3':
            checks={
                'parent_sha256':manifest.get('parent_sha256')==sha256_file(parent_manifest_path),
                'parent_version':manifest.get('parent_version')==parent.get('version')=='v2',
                'authorized':manifest.get('cleaning_authorized_by_user') is True,
                'seed_unchanged':manifest.get('seed')==parent.get('seed')==20261013,
                'source_unchanged':manifest.get('source')==parent.get('source'),
            }
            for split in ('train','val','test'):
                excluded={r['id'] for r in manifest['removed_from_v2'][split]}
                checks[f'{split}_ordered_subsequence']=manifest['indices'][split]==[i for i in parent['indices'][split] if i not in excluded]
            excluded={r['id'] for r in manifest['removed_from_v2']['official']}
            checks['official_ordered_subsequence']=manifest['official_short_tool_test_indices']==[i for i in parent['official_short_tool_test_indices'] if i not in excluded]
        lineage_failures = [name for name, passed in checks.items() if not passed]
        lineage = {
            "parent_path": str(parent_manifest_path),
            "parent_sha256_actual": sha256_file(parent_manifest_path),
            "removed_ids": sorted(removed),
            "checks": checks,
        }
        if manifest.get('version')=='v3':
            lineage.pop('removed_ids')
            lineage['removed_ids_by_split']={s:[r['id'] for r in entries] for s,entries in manifest['removed_from_v2'].items()}
    actual_source_hashes = {
        "train_parquet_sha256": sha256_file(train_path),
        "test_parquet_sha256": sha256_file(test_path),
    }
    source_hashes = {}
    hash_failures = []
    for key, actual in actual_source_hashes.items():
        expected = manifest.get("source", {}).get(key)
        matched = expected == actual
        source_hashes[key] = {"expected": expected, "actual": actual, "matched": matched}
        if not matched:
            hash_failures.append(key)

    train_rows = pq.read_table(train_path).to_pylist()
    test_rows = pq.read_table(test_path).to_pylist()
    row_map: dict[int, dict[str, Any]] = {}
    duplicate_source_ids = []
    for row in train_rows:
        index = _row_id(row)
        if index in row_map:
            duplicate_source_ids.append(index)
        row_map[index] = row

    split_names = ("train", "val", "test")
    indices = {name: list(manifest.get("indices", {}).get(name, [])) for name in split_names}
    split_summary: dict[str, Any] = {}
    missing_ids: dict[str, list[int]] = {}
    count_mismatches = []
    tool_sets: dict[str, set[str]] = {}
    invalid_json_ids: dict[str, list[int]] = {}
    missing_schema_tool_ids: dict[str, list[int]] = {}
    empty_schema_ids: dict[str, list[int]] = {}

    for split in split_names:
        split_ids = indices[split]
        missing = [index for index in split_ids if index not in row_map]
        missing_ids[split] = missing
        expected_count = manifest.get("counts", {}).get(split)
        if expected_count != len(split_ids):
            count_mismatches.append(split)
        rows = [row_map[index] for index in split_ids if index in row_map]
        tools: set[str] = set()
        json_failures = []
        schema_failures = []
        empty_schemas = []
        for row in rows:
            sample_id = _row_id(row)
            schema_names = schema_tool_names(row["extra_info"]["instruction"])
            tools.update(schema_names)
            if not schema_names:
                empty_schemas.append(sample_id)
            calls, error = parse_label_calls(row["reward_model"]["ground_truth"])
            if error is not None:
                json_failures.append(sample_id)
                continue
            for call in calls:
                name = call.get("name")
                if type(name) is not str or normalize_tool_name(name) not in schema_names:
                    schema_failures.append(sample_id)
                    break
        tool_sets[split] = tools
        invalid_json_ids[split] = sorted(set(json_failures))
        missing_schema_tool_ids[split] = sorted(set(schema_failures))
        empty_schema_ids[split] = sorted(set(empty_schemas))
        split_summary[split] = {
            "length": len(split_ids),
            "first_5_ids": split_ids[:5],
            "index_list_sha256": sha256_json(split_ids),
            "selected_records_sha256": sha256_json(rows),
            "tool_name_count_normalized": len(tools),
        }

    index_overlaps = {}
    tool_overlaps = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        key = f"{left}__{right}"
        index_overlaps[key] = sorted(set(indices[left]) & set(indices[right]))
        tool_overlaps[key] = sorted(tool_sets[left] & tool_sets[right])

    failures = {
        "lineage_checks": lineage_failures,
        "source_hash_keys": hash_failures,
        "duplicate_source_ids": sorted(set(duplicate_source_ids)),
        "count_mismatch_splits": count_mismatches,
        "missing_ids": missing_ids,
        "index_overlaps": index_overlaps,
        "tool_name_overlaps": tool_overlaps,
        "invalid_json_ids": invalid_json_ids,
        "label_tool_not_in_schema_ids": missing_schema_tool_ids,
        "empty_schema_ids": empty_schema_ids,
    }
    if manifest.get('version')=='v3':
        from toolrl.clean_manifest import violations
        failures['schema_contract_ids']={split:[i for i in indices[split] if i in row_map and violations({'prompt':row_map[i]['prompt'],'ground_truth':row_map[i]['reward_model']['ground_truth']})] for split in split_names}
        failures['within_split_duplicate_ids']={split:sorted({i for i in indices[split] if indices[split].count(i)>1}) for split in split_names}
    passed = not any(
        value
        for group in failures.values()
        for value in (group.values() if isinstance(group, dict) else [group])
    )
    return {
        "passed": passed,
        "actual_input_path": str(manifest_path),
        "manifest_seed": manifest.get("seed"),
        "lineage": lineage,
        "source_row_counts": {"train_parquet": len(train_rows), "test_parquet": len(test_rows)},
        "source_sha256": source_hashes,
        "splits": split_summary,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--parent-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.train, args.test, args.manifest, args.parent_manifest)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
