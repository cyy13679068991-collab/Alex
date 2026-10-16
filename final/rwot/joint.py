from __future__ import annotations
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
import torch
from torch import nn

def temperature(cfg, absolute_epoch):
    return max(float(cfg['temp_min']), float(cfg['temp_start']) * float(cfg['temp_decay']) ** (absolute_epoch - 1))


@torch.no_grad()
def initialize_linear_head(head: nn.Linear, centers: torch.Tensor) -> None:
    """Install W_k=2*c_k, b_k=-||c_k||² without replacing Parameter objects."""
    if not isinstance(head, nn.Linear) or head.bias is None:
        raise TypeError("Expected the original nn.Linear(D,K,bias=True)")
    c = torch.as_tensor(centers, device=head.weight.device, dtype=head.weight.dtype)
    if c.ndim != 2 or tuple(c.shape) != tuple(head.weight.shape):
        raise ValueError(f"centers must have shape {tuple(head.weight.shape)}")
    if not torch.isfinite(c).all():
        raise ValueError("centers contain non-finite values")
    # Do NOT normalize centers, remove the bias, or multiply by temperature.
    head.weight.copy_(2.0 * c)
    head.bias.copy_(-c.square().sum(dim=1))


@torch.no_grad()
def nearest_center(z: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Reference Euclidean nearest center."""
    if z.ndim != 2 or centers.ndim != 2 or z.shape[1] != centers.shape[1]:
        raise ValueError("Expected z[N,D], centers[K,D]")
    zz, cc = z.double(), centers.to(z.device).double()
    d2 = zz.square().sum(1, keepdim=True) + cc.square().sum(1)[None] - 2 * zz @ cc.T
    return d2.argmin(dim=1)


@torch.no_grad()
def verify_head_initialization(head: nn.Linear, z: torch.Tensor, centers: torch.Tensor) -> bool:
    return torch.equal(head(z).argmax(1), nearest_center(z, centers))

@contextmanager
def no_kmeans_calls(core):
    """Reject both source.snapshot() and accidental fitting during refinement/export."""
    with patch.object(core.KMeans, 'fit', side_effect=RuntimeError('KMeans forbidden after initialization')):
        yield


def make_reference(core, inp, raw_ref, cfg):
    """Exact copy of the original reference conditioning; never simplify it."""
    raw = raw_ref['q'].clone()
    ref = dict(raw_ref, q=core.sharpen_distribution(raw, cfg['beta']))
    if cfg['soft_family'] != 'none':
        nbr = ref['neighbors']
        rec = (nbr[nbr] == torch.arange(len(nbr), device=nbr.device)[:, None, None]).any(-1)
        xc = core.candidate_cosine(inp['x'], nbr)
        rel = (core.reliability_scores(raw, xc, 'structure_feature_geomean_reciprocal', reciprocal=rec) / 1.25).clamp(0, 1).detach()
        qs = core.sharpen_q(raw, cfg['beta'])
        qr = core.soft_reweight_linear(qs, rel, cfg['intensity']) if cfg['soft_family'] == 'linear' else core.soft_reweight_exponential(qs, rel, cfg['intensity'])
        ref = dict(ref, q=qr)
    return ref


@torch.no_grad()
def outputs(model, inp, cfg, epoch):
    model.eval()
    o = model(inp)
    s = (o['logits'] / temperature(cfg, epoch)).softmax(1)
    if not torch.isfinite(o['z']).all() or not torch.isfinite(s).all():
        raise FloatingPointError('Non-finite output')
    pred = s.argmax(1)
    if not torch.equal(pred, o['logits'].argmax(1)):
        raise RuntimeError('Softmax rounding changed argmax: inspect precision before proceeding')
    return o['z'].detach(), s.detach(), pred.detach()


def refine(core, model, inp, ref, cfg, start_epoch, steps, lr_mult, seed):
    """Joint refinement with the same OT, NCut, InfoNCE and spread objectives."""
    if steps < 1 or not 0 < lr_mult <= 1:
        raise ValueError('steps >= 1 and 0 < lr_mult <= 1 required')
    for p in model.parameters():
        p.requires_grad_(True)
    core.seed_all(seed + 100000)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg['lr']) * lr_mult,
                           weight_decay=cfg['weight_decay'])
    engine = core.FastOTEngine(inp['x'], ref, cfg, batch_size=4096)
    with no_kmeans_calls(core):
        for step in range(1, steps + 1):
            epoch = start_epoch + step
            model.train()
            opt.zero_grad(set_to_none=True)
            out = model(inp)
            z = out['z']
            s = (out['logits'] / temperature(cfg, epoch)).softmax(1)
            nc = core.relaxed_ncut_loss(s, inp['graph'], epsilon=cfg['ncut_eps'])
            spread = core.optional_spread(z, cfg['lambda_spread'], core.spread_loss)
            value, og = engine.compute(z, ref)
            base = core.compose_base(nc, spread, cfg['lambda_ncut'], cfg['lambda_spread'])
            if cfg['lambda_disc']:
                v1 = model(core.view(inp, 0.15, 0.1))['z']
                v2 = model(core.view(inp, 0.15, 0.1))['z']
                ids = torch.randperm(len(z), device=z.device)[:4096] if len(z) > 5000 else None
                base = base + cfg['lambda_disc'] * core.symmetric_infonce(v1, v2, 0.5, ids)
            if not (torch.isfinite(base) and torch.isfinite(og).all() and np.isfinite(value)):
                raise FloatingPointError('Non-finite loss/gradient')
            torch.autograd.backward([base, z], [None, cfg['lambda_ot'] * og])
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
            opt.step()


def save_outputs(folder, model, inp, cfg, epoch, seed):
    folder.mkdir(parents=True, exist_ok=False)
    _, _, pred = outputs(model, inp, cfg, epoch)
    np.save(folder / 'prediction.npy', pred.cpu().numpy())
    torch.save(dict(state_dict={n: p.detach().cpu().clone() for n, p in model.state_dict().items()},
                    config=cfg, epoch=epoch, phase='joint_refined', refinement_steps=10, seed=seed),
               folder / 'checkpoint.pt')
