"""Collect fresh evaluation JSONs. Reference scores never substitute for absent new runs."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
METRICS = ['ACC', 'NMI', 'ARI', 'Macro-F1']
JSON_KEYS = dict(zip(METRICS, ['acc', 'nmi', 'ari', 'f1']))


def load_ref(path):
    with open(path, newline='', encoding='utf8') as f:
        rows = list(csv.DictReader(f))
    return {(r['dataset'], int(r['seed'])): {m: float(r[m]) for m in METRICS} for r in rows}


def write_csv(path, rows, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def collect(run_root, names, seeds, *, mean_tol=0.01, std_tol=0.015, seed_tol=0.03, require_complete=True):
    config=json.loads((ROOT/'configs/final.json').read_text(encoding='utf8'))
    seeds = {n: config.get('dataset_seeds',{}).get(n,config['seeds']) for n in names} if seeds is None else seeds
    seeds = seeds if isinstance(seeds,dict) else {n:seeds for n in names}
    run_root = Path(run_root)
    ref = load_ref(ROOT / 'results/per_seed.csv')
    records, missing, pairs, summary = [], [], [], []
    groups = defaultdict(list)
    for name in names:
        for seed in seeds[name]:
            key = (name, seed)
            path = run_root / 'evaluations' / name.replace('/', '__') / f'seed{seed}' / 'metrics.json'
            if not path.is_file():
                missing.append(f'{name} seed{seed}: {path}')
                continue
            m = json.loads(path.read_text(encoding='utf8'))
            if m.get('dataset') != name or int(m.get('seed', -1)) != seed:
                raise ValueError(f'Identity mismatch in {path}')
            r = {'dataset': name, 'seed': seed}
            for metric in METRICS:
                value = float(m[JSON_KEYS[metric]])
                if not 0 <= value <= 1:
                    raise ValueError(f'Out-of-range {metric} in {path}: {value}')
                r[metric] = value
                expected = ref[key][metric]
                pairs.append(dict(dataset=name, seed=seed, metric=metric, reference=expected,
                                  reproduced=value, absolute_difference=abs(value-expected),
                                  within_seed_tolerance=abs(value-expected) <= seed_tol))
            groups[name].append(r)
            records.append(r)
    write_csv(run_root / 'per_seed.csv', records, ['dataset', 'seed'] + METRICS)
    for name in names:
        rs = groups[name]
        if not rs:
            continue
        ref_rs = [ref[(name, seed)] for seed in seeds[name]]
        for metric in METRICS:
            vals = [r[metric] for r in rs]
            refs = [r[metric] for r in ref_rs]
            m, s = mean(vals), pstdev(vals)
            # The mean/std comparison is only valid on the complete seed set.
            complete = len(rs) == len(seeds[name])
            mean_delta = abs(m-mean(refs)) if complete else None
            std_delta = abs(s-pstdev(refs)) if complete else None
            summary.append(dict(dataset=name, metric=metric, runs=len(rs), mean=m, std=s,
                                reference_mean=mean(refs), reference_std=pstdev(refs),
                                mean_difference=mean_delta, std_difference=std_delta,
                                within_tolerance=(complete and mean_delta <= mean_tol and std_delta <= std_tol)))
    write_csv(run_root / 'summary.csv', summary, ['dataset','metric','runs','mean','std','reference_mean','reference_std','mean_difference','std_difference','within_tolerance'])
    write_csv(run_root / 'comparison.csv', pairs, ['dataset','seed','metric','reference','reproduced','absolute_difference','within_seed_tolerance'])
    accepted = not missing and bool(summary) and all(r['within_tolerance'] for r in summary) and all(p['within_seed_tolerance'] for p in pairs)
    verdict = dict(complete=not missing, matches_reference=accepted, expected_runs=sum(len(seeds[n]) for n in names),
                   completed_runs=len(records), missing=missing, tolerances=dict(mean=mean_tol,std=std_tol,per_seed=seed_tol),
                   values_are_new_runs_only=True)
    (run_root / 'verification.json').write_text(json.dumps(verdict,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in verdict.items() if k!='missing'},indent=2), flush=True)
    if require_complete and missing:
        raise RuntimeError(f'{len(missing)} fresh evaluations missing, see {run_root}/verification.json')
    return verdict


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-root',type=Path,required=True)
    ap.add_argument('--dataset',action='append',dest='datasets')
    ap.add_argument('--seeds',type=int,nargs='+',default=None)
    ap.add_argument('--mean-tol',type=float,default=0.01)
    ap.add_argument('--std-tol',type=float,default=0.015)
    ap.add_argument('--seed-tol',type=float,default=0.03)
    ap.add_argument('--allow-partial',action='store_true')
    args=ap.parse_args()
    names=args.datasets or list(json.loads((ROOT/'configs/final.json').read_text())['datasets'])
    verdict=collect(args.run_root,names,args.seeds,mean_tol=args.mean_tol,std_tol=args.std_tol,
                    seed_tol=args.seed_tol,require_complete=not args.allow_partial)
    if not verdict['matches_reference'] and not args.allow_partial:
        raise SystemExit(2)

if __name__ == '__main__': main()
