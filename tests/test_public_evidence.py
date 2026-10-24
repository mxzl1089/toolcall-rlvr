"""The lightweight public records must reproduce the reported primary comparison."""
import importlib.util
from pathlib import Path
import numpy as np
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]

def test_published_primary_comparison():
    spec=importlib.util.spec_from_file_location('reproduce',ROOT/'scripts/reproduce_statistics.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    result=mod.analyze(ROOT/'results/per_prediction.csv')
    assert [result['accuracy'][m]['correct'] for m in ['continue','sparse','dense']]==[211,204,200]
    assert all(v['n']==477 for v in result['accuracy'].values())
    np.testing.assert_allclose(result['dense_minus_sparse_pp'],-4/477*100,atol=1e-10)
    np.testing.assert_allclose(result['paired_bootstrap95_pp'],[-2.935010482180293,1.0482180293501049],atol=1e-10)
    assert sum(result['first_failure_counts']['dense'].values())==277

def test_frozen_manifest_bytes():
    config=json.loads((ROOT/'configs/protocol.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((ROOT/config['manifest']).read_bytes()).hexdigest()==config['manifest_sha256']

def test_public_result_checksums():
    expected=json.loads((ROOT/'results/SHA256SUMS.json').read_text(encoding='utf-8'))
    for name,digest in expected.items():
        assert hashlib.sha256((ROOT/'results'/name).read_bytes()).hexdigest()==digest,name
