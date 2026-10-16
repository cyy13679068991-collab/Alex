"""50 independent processes (ten datasets x five seeds) with isolated outputs and strict evaluation."""
from __future__ import annotations
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from data import specifications, check_dataset, download_exact_data
from collect_results import collect

ROOT=Path(__file__).resolve().parents[1]


def command(argv, logfile):
    logfile.parent.mkdir(parents=True,exist_ok=True)
    with open(logfile,'w',encoding='utf8') as log:
        log.write('COMMAND: '+' '.join(map(str,argv))+'\n')
        log.flush()
        process=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
    if process.returncode:
        raise RuntimeError(f'Command failed ({process.returncode}), inspect {logfile}')


def run(args):
    names=args.datasets or list(specifications())
    config=json.loads((ROOT/'configs/final.json').read_text(encoding='utf8'))
    seeds={n: args.seeds if args.seeds is not None else config.get('dataset_seeds',{}).get(n,config['seeds']) for n in names}
    if not names or any(not v or len(v)!=len(set(v)) or any(x<0 for x in v) for v in seeds.values()):
        raise ValueError('Supply unique nonnegative seeds')
    data_root=args.data_root.resolve()
    output=args.output.resolve()
    jobs=[(name,seed) for name in names for seed in seeds[name]]
    print(f'{len(jobs)} planned runs: '+', '.join(f'{n}[{s}]' for n,s in jobs),flush=True)
    if args.dry_run:
        print('Dry run: no dataset fetch, GPU training or result-writing performed.',flush=True)
        return
    if args.download:
        download_exact_data(data_root,names)
    errors=[e for n in names for e in check_dataset(data_root,n)]
    if errors:
        raise RuntimeError('DATA PREFLIGHT FAILED:\n'+'\n'.join(errors))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable: training requires CUDA')
    if torch.__version__.split('+')[0] != '2.5.1':
        raise RuntimeError(f'Expected PyTorch 2.5.1, found {torch.__version__}')
    import numpy, scipy, sklearn
    manifest=dict(python=sys.version,platform=platform.platform(),cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0),torch=torch.__version__,numpy=numpy.__version__,
                  scipy=scipy.__version__,sklearn=sklearn.__version__,dataset_names=names,seeds=seeds,
                  config=json.loads((ROOT/'configs/final.json').read_text(encoding='utf8')),
                  dataset_files={n:{k:__import__('hashlib').sha256((data_root/n/k).read_bytes()).hexdigest() for k in specifications()[n]} for n in names})
    output.mkdir(parents=True,exist_ok=True)
    (output/'environment.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    failures=[]
    for index,(name,seed) in enumerate(jobs,1):
        slug=name.replace('/','__')
        folder=output/'jobs'/slug/f'seed{seed}'
        ckpt=folder/slug/f'seed{seed}'/'final'/'checkpoint.pt'
        ev=output/'evaluations'/slug/f'seed{seed}'
        if ev.joinpath('metrics.json').exists() and ckpt.exists():
            if not args.resume:
                raise FileExistsError(f'Existing result; use --resume: {ev}')
            print(f'[{index}/{len(jobs)}] reuse completed {name} seed={seed}',flush=True)
            continue
        if folder.exists() and not ckpt.exists():
            raise RuntimeError(f'Incomplete run at {folder}. Move it aside manually before retrying (no silent overwrites).')
        print(f'[{index}/{len(jobs)}] train {name} seed={seed}',flush=True)
        try:
            if not ckpt.exists():
                command([sys.executable,str(ROOT/'scripts/train.py'),'--data-root',str(data_root),
                         '--dataset',name,'--seed',str(seed),'--output',str(folder),
                         '--cache-dir',str(output/'shared_cache')],
                        output/'logs'/f'{slug}_seed{seed}_train.log')
            if ev.exists():
                raise FileExistsError(f'Partial evaluation directory; move manually: {ev}')
            command([sys.executable,str(ROOT/'scripts/evaluate.py'),'--data-root',str(data_root),
                     '--checkpoint',str(ckpt),'--device','cuda','--output',str(ev)],
                    output/'logs'/f'{slug}_seed{seed}_eval.log')
        except Exception as ex:
            failures.append(f'{name} seed{seed}: {ex}')
            print(f'ERROR: {failures[-1]}',flush=True)
            if not args.continue_on_error: break
    verdict=collect(output,names,seeds,mean_tol=args.mean_tol,std_tol=args.std_tol,
                    seed_tol=args.seed_tol,require_complete=False)
    if failures or not verdict['complete']:
        raise RuntimeError('Some runs did not complete. Check logs and verification.json.\n'+'\n'.join(failures))
    if not verdict['matches_reference']:
        raise RuntimeError('Fresh 50-run metrics differ beyond the published tolerances; inspect comparison.csv')
    print('Full reproduction within configured tolerances.',flush=True)


def main():
    ap=argparse.ArgumentParser(description='Reproduce original five-seed RWOT scores from scratch.')
    ap.add_argument('--data-root',type=Path,default=ROOT/'data')
    ap.add_argument('--output',type=Path,default=ROOT/'runs/reproduction')
    ap.add_argument('--dataset',dest='datasets',choices=list(specifications()),action='append')
    ap.add_argument('--seeds',type=int,nargs='+',default=None)
    ap.add_argument('--download',action='store_true',help='Fetch pinned CAHC dataset files if missing')
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--continue-on-error',action='store_true')
    ap.add_argument('--mean-tol',type=float,default=0.01)
    ap.add_argument('--std-tol',type=float,default=0.015)
    ap.add_argument('--seed-tol',type=float,default=0.03)
    run(ap.parse_args())

if __name__ == '__main__': main()
