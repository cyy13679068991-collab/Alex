"""Training, prediction and evaluation interfaces."""
from pathlib import Path
import argparse, json, gc
from . import core
from .joint import initialize_linear_head, verify_head_initialization, make_reference, refine, save_outputs, outputs, no_kmeans_calls
import numpy as np
import torch
import scipy.sparse as sp
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]

def configure(args):
    config = Path(args.config).resolve()
    document = json.loads(config.read_text(encoding='utf8'))
    core.CONFIGS = document['datasets']
    core.SEEDS = document['seeds']
    core.DATA_ROOT = Path(args.data_root).resolve()
    return document

def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/final.json')
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--dataset', choices=list(core.CONFIGS))
    p.add_argument('--seed', type=int)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache-dir', type=Path, help='Optional shared deterministic input cache for repeated seeds')
    return p

def inference_inputs(core, name, cfg, device):
    x, h, paths = core.load_features_graph(name)
    hashes = {p.name: core.sha256(p) for p in paths}
    if hashes != core.DATA_HASHES[name]:
        raise RuntimeError('Data version/order differs from the frozen inputs')
    x = torch.as_tensor(x.toarray() if sp.issparse(x) else np.asarray(x), dtype=torch.float32, device=device)
    if cfg['input_norm'] != 'none':
        x = F.normalize(x, p=1 if cfg['input_norm'] == 'l1' else 2, dim=1)
    dv, de = (np.asarray(h.sum(1)).ravel(), np.asarray(h.sum(0)).ravel())
    return dict(x=x, H=core.sparse(h, device), dv=torch.tensor((dv + 1e-12) ** (-0.5), device=device), de=torch.tensor((de + 1e-12) ** (-1), device=device))

def predict(args):
    configure(args)
    checkpoint = args.checkpoint
    if checkpoint is None:
        if args.dataset is None or args.seed is None:
            raise ValueError('Supply --checkpoint or both --dataset and --seed')
        checkpoint = ROOT / 'checkpoints' / args.dataset.replace('/', '__') / f'seed{args.seed}.pt'
    checkpoint = Path(checkpoint).resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    with core.prohibit_label_io(), no_kmeans_calls(core):
        ck = torch.load(checkpoint, map_location=args.device, weights_only=False)
        if ck.get('phase') != 'joint_refined' or ck.get('refinement_steps') != 10:
            raise ValueError('Expected a ten-step joint checkpoint')
        cfg = ck['config']
        name = cfg['dataset']
        if args.dataset is not None and args.dataset != name:
            raise ValueError('Dataset/checkpoint mismatch')
        if args.seed is not None and ck['seed'] != args.seed:
            raise ValueError('Seed/checkpoint mismatch')
        if cfg != core.CONFIGS[name]:
            raise ValueError('Config differs from the saved model; use its exact training config')
        inp = inference_inputs(core, name, cfg, torch.device(args.device))
        model = core.RWOTModel(inp, core.CLUSTERS[name], cfg).to(args.device)
        model.load_state_dict(ck['state_dict'], strict=True)
        _, s, pred = outputs(model, inp, cfg, ck['epoch'])
        pred = pred.cpu().numpy()
        np.save(args.output / 'prediction.npy', pred)
    return (name, ck['seed'], pred)

def train(args):
    document = configure(args)
    torch.set_num_threads(4)
    names = [args.dataset] if args.dataset else list(core.CONFIGS)
    seeds = [args.seed] if args.seed is not None else core.SEEDS
    if args.pretrained and (len(names) != 1 or len(seeds) != 1):
        raise ValueError('--pretrained requires one dataset and seed')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    core.CACHE_DIR = Path(args.cache_dir).resolve() if args.cache_dir else args.output / 'cache'
    protocol = json.loads((ROOT / 'configs/joint.json').read_text(encoding='utf8'))
    assert protocol['steps'] == 10 and protocol['lr_mult'] == 0.1 and (protocol['seed_offset'] == 100000)
    for name in names:
        run_seeds = [args.seed] if args.seed is not None else document.get('dataset_seeds',{}).get(name,core.SEEDS)
        for seed in run_seeds:
            folder = args.output / name.replace('/', '__') / f'seed{seed}'
            folder.mkdir(parents=True)
            cfg = dict(core.CONFIGS[name])
            if args.pretrained:
                path = args.pretrained
            else:
                core.train_one(name, seed, folder / 'pretrain')
                path = folder / 'pretrain' / f"epoch_{cfg['epochs']}" / 'checkpoint.pt'
            torch.cuda.empty_cache()
            with core.prohibit_label_io():
                ck = torch.load(path, map_location='cpu', weights_only=False)
                if ck['config'] != cfg or ck['epoch'] != cfg['epochs']:
                    raise ValueError('Pretraining config/epoch mismatch')
                inp, raw_ref, *_ = core.prepare(name, cfg, torch.device('cuda'))
                core.seed_all(seed)
                model = core.RWOTModel(inp, core.CLUSTERS[name], cfg).cuda()
                model.load_state_dict(ck['state_dict'], strict=True)
                z, _, _ = outputs(model, inp, cfg, ck['epoch'])
                centers = torch.as_tensor(ck['centers'], dtype=z.dtype, device=z.device)
                initialize_linear_head(model.cluster_output, centers)
                if not verify_head_initialization(model.cluster_output, z, centers):
                    raise RuntimeError('Analytical initialization mismatch')
                del ck, centers, z
                ref = make_reference(core, inp, raw_ref, cfg)
                refine(core, model, inp, ref, cfg, cfg['epochs'], protocol['steps'], protocol['lr_mult'], seed)
                with no_kmeans_calls(core):
                    save_outputs(folder / 'final', model, inp, cfg, cfg['epochs'] + 10, seed)
            del model, inp, raw_ref, ref
            gc.collect()
            torch.cuda.empty_cache()
