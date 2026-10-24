"""Local paths, immutable protocol, and crash-safe bookkeeping (CPU only)."""
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(ROOT / 'artifacts' / 'torch_cache'))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for attempt in range(80):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 79:
                raise
            time.sleep(.25)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def protocol():
    return read(ROOT / 'config/protocol.json')


def freeze_check():
    lock = read(ROOT / 'config/freeze.json')
    for p, expected in lock['files'].items():
        target = Path(p) if Path(p).is_absolute() else ROOT / p
        if sha(target) != expected:
            raise RuntimeError(f'Frozen input changed: {target}')
    return lock


def exposure_order(ids, seed, epochs):
    result = []
    for epoch in range(epochs):
        group = sorted(ids)
        random.Random(seed + epoch).shuffle(group)
        result.extend(group)
    return result


def checkpoint_valid(path):
    path = Path(path)
    marker = path / 'committed.json'
    if not marker.exists():
        return False
    m = read(marker)
    return all((path / p).is_file() and sha(path / p) == h for p, h in m['files'].items())


def latest_checkpoint(folder):
    candidates = sorted(Path(folder).glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]), reverse=True)
    return next((p for p in candidates if checkpoint_valid(p)), None)
