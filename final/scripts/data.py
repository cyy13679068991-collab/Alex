"""Exact input preparation for the frozen CAHC data version (stdlib only)."""
from __future__ import annotations
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = 'e6975ea33b0734255137e7592f83f7fcd325405d'
URL = 'https://github.com/nilics/CAHC.git'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for buf in iter(lambda: f.read(1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()


def specifications():
    return json.loads((ROOT / 'configs/datasets.json').read_text(encoding='utf8'))['data_hashes']


def check_dataset(data_root, dataset, *, labels_required=True):
    base = Path(data_root) / dataset
    issues = []
    for name, expected in specifications()[dataset].items():
        file = base / name
        if not file.is_file():
            issues.append(f'MISSING {file}')
        else:
            actual = sha256(file)
            if actual != expected:
                issues.append(f'HASH MISMATCH {file}: expected {expected}, got {actual}')
    if labels_required and not (base / 'labels.pickle').is_file() and not (base / 'labels.npy').is_file():
        issues.append(f'MISSING {base}/labels.pickle (needed for evaluation only)')
    return issues


def _cmd(argv, *, cwd=None):
    subprocess.run(argv, cwd=cwd, check=True)


def download_exact_data(data_root, names):
    root = Path(data_root).expanduser().resolve()
    if all(not check_dataset(root, n) for n in names):
        return
    checkout = root.parent / '.rwot_download' / 'CAHC'
    if not (checkout / '.git').exists():
        checkout.mkdir(parents=True, exist_ok=True)
        _cmd(['git', 'init', '-q'], cwd=checkout)
        _cmd(['git', 'remote', 'add', 'origin', URL], cwd=checkout)
        _cmd(['git', 'sparse-checkout', 'init', '--cone'], cwd=checkout)
        _cmd(['git', 'sparse-checkout', 'set', 'data/dataset'], cwd=checkout)
    _cmd(['git', '-c', 'advice.detachedHead=false', 'fetch', '--depth', '1', 'origin', COMMIT], cwd=checkout)
    _cmd(['git', 'checkout', '--detach', '--force', 'FETCH_HEAD'], cwd=checkout)
    src_root = checkout / 'data' / 'dataset'
    if not src_root.exists():
        raise FileNotFoundError(f'Pinned source {URL}#{COMMIT} has no data/dataset directory')
    for dataset in names:
        target = root / dataset
        target.mkdir(parents=True, exist_ok=True)
        source = src_root / dataset
        if not source.exists():
            raise FileNotFoundError(f'Expected source folder not found: {source}; do not substitute alternate preprocessing')
        for filename in ['features.pickle', 'hypergraph.pickle', 'labels.pickle']:
            src = source / filename
            dst = target / filename
            if not src.is_file():
                if filename == 'labels.pickle' and (target / 'labels.npy').is_file():
                    continue
                raise FileNotFoundError(f'Missing frozen dataset file: {src}')
            if not dst.exists():
                shutil.copy2(src, dst)
    errors = [i for name in names for i in check_dataset(root, name)]
    if errors:
        raise RuntimeError('Frozen inputs are unavailable/mismatched:\n' + '\n'.join(errors))
