"""Train the full-spread RWOT model with frozen dataset configurations.

python train.py --data-root /path/to/data --output runs/five_seed
See README.md for dependencies, data formats and evaluation.
"""
from __future__ import annotations
import os
os.environ.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8')
import argparse, builtins, csv, gc, hashlib, io, json, pickle, random, statistics, time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score, pairwise_distances_argmin
DATA_ROOT = Path('data')
CACHE_DIR = Path('rwot_cache')
RUN_CONFIG = json.loads((Path(__file__).resolve().parents[1] / 'configs/final.json').read_text(encoding='utf8'))
CONFIGS = RUN_CONFIG['datasets']
SEEDS = RUN_CONFIG['seeds']
CLUSTERS = json.loads((Path(__file__).resolve().parents[1] / 'configs/datasets.json').read_text(encoding='utf8'))['clusters']
DATA_HASHES = json.loads((Path(__file__).resolve().parents[1] / 'configs/datasets.json').read_text(encoding='utf8'))['data_hashes']
INPUT_SOURCE_HASHES = {'training_base.py': '4409f6f89ef99f3bfa7ae1929a79429002239653be45b574596f6440e1d68179',
 'structural_reference.py': 'bf0fa2b523b78ee0251570ec64f69ca2b3730164e546384b98adfc5ef535ddad',
 'data_unlabeled.py': 'fa7ee9c933ee9c2918462bcb7174df79b400d2c1756fa3f4f8c5fa32d5f54a93'}

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def write_json(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, indent=2, allow_nan=False), encoding='utf-8')

@contextmanager
def prohibit_label_io(enabled=True):
    if not enabled:
        yield
        return
    original = builtins.open
    io_original = io.open

    def check(p):
        if isinstance(p, (str, bytes, os.PathLike)):
            n = os.fsdecode(p).replace('\\', '/').split('/')[-1].lower()
            if n.startswith('label') and (not n.endswith(('.py', '.pyc'))):
                raise RuntimeError('Unsupervised training attempted label I/O: ' + n)

    def guarded(p, *a, **k):
        check(p)
        return original(p, *a, **k)

    def io_guarded(p, *a, **k):
        check(p)
        return io_original(p, *a, **k)
    builtins.open = guarded
    io.open = io_guarded
    try:
        yield
    finally:
        builtins.open = original
        io.open = io_original

def load_features_graph(name):
    folder = DATA_ROOT / name
    if (folder / 'hypergraph.npz').exists():
        paths = [folder / 'features.npz', folder / 'hypergraph.npz']
        features = sp.load_npz(paths[0])
        h = sp.load_npz(paths[1])
        np.clip(h.data, 0, 1, out=h.data)
    else:
        paths = [folder / 'features.pickle', folder / 'hypergraph.pickle']
        with paths[0].open('rb') as f:
            features = pickle.load(f)
        with paths[1].open('rb') as f:
            hypergraph = pickle.load(f)
        rows = []
        cols = []
        for i, edge in enumerate(hypergraph.values()):
            nodes = sorted(set(edge))
            rows.extend([i] * len(nodes))
            cols.extend(nodes)
        h = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(hypergraph), features.shape[0]))
    if h.shape[0] != features.shape[0]:
        if h.shape[1] != features.shape[0]:
            raise ValueError('Incidence shape does not match features')
        h = h.T
    return (features, h.astype(np.float32).tocsc(), paths)

def load_labels(name):
    p = DATA_ROOT / name
    y = np.load(p / 'labels.npy') if (p / 'labels.npy').exists() else np.asarray(pickle.load((p / 'labels.pickle').open('rb')))
    if y.ndim == 2:
        y = y.argmax(1)
    return np.unique(y, return_inverse=True)[1]

def transition(incidence, edge_weight=None):
    h = sp.csr_matrix(incidence, dtype=np.float64)
    if h.ndim != 2 or h.shape[0] == 0 or h.shape[1] == 0:
        raise ValueError('incidence must be a nonempty node by edge matrix')
    if np.any(h.data < 0) or not np.isfinite(h.data).all():
        raise ValueError('incidence must be finite and nonnegative')
    w = np.ones(h.shape[1]) if edge_weight is None else np.asarray(edge_weight, dtype=np.float64)
    if w.shape != (h.shape[1],) or np.any(w < 0) or (not np.isfinite(w).all()):
        raise ValueError('invalid fixed edge weights')
    de = np.asarray(h.sum(axis=0)).ravel()
    dv = np.asarray(h @ w).ravel()
    if np.any(dv <= 0):
        raise ValueError('isolated nodes make P undefined')
    p = sp.diags(1.0 / dv) @ h @ sp.diags(np.divide(w, de, out=np.zeros_like(w), where=de > 0)) @ h.T
    return p.tocsr()

def random_walk_restart(incidence, edge_weight=None, alpha=0.2, tol=1e-08, max_iter=120):
    if not 0 < alpha <= 1:
        raise ValueError('alpha must lie in (0,1]')
    p = transition(incidence, edge_weight)
    n = p.shape[0]
    identity = np.eye(n, dtype=np.float64)
    r = identity.copy()
    residual = float('inf')
    for iteration in range(1, max_iter + 1):
        next_r = alpha * identity + (1.0 - alpha) * (p @ r)
        residual = float(np.max(np.abs(next_r - r)))
        r = next_r
        if residual < tol:
            break
    if residual >= tol:
        raise RuntimeError('RWR failed to converge')
    row_error = float(np.max(np.abs(r.sum(axis=1) - 1)))
    return (r, dict(iterations=iteration, residual=residual, converged=residual < tol, row_error=row_error))

def _build_reference_unlocked(h, alpha=0.2, topk=20):
    hh = hashlib.sha256(h.data.tobytes() + h.indices.tobytes() + h.indptr.tobytes() + str(h.shape).encode()).hexdigest()[:20]
    cache = CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    p = cache / f'rwr_{hh}_{alpha}.npz'
    if p.exists():
        a = np.load(p)
        r = a['r'].copy()
        diagnostic = dict(iterations=int(a['iterations']), residual=float(a['residual']), row_error=float(a['row_error']), converged=True)
    else:
        r, diagnostic = random_walk_restart(h, alpha=alpha, max_iter=1000)
        tmp = p.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, r=r, **{k: diagnostic[k] for k in ['iterations', 'residual', 'row_error']})
        tmp.replace(p)
    n = len(r)
    np.fill_diagonal(r, 0)
    np.fill_diagonal(r, -np.inf)
    neighbors = np.argsort(-r, axis=1, kind='stable')[:, :min(topk, n - 1)]
    np.fill_diagonal(r, 0)
    rows = np.repeat(np.arange(n), neighbors.shape[1])
    values = r[rows, neighbors.ravel()].reshape(neighbors.shape)
    if np.any(values.sum(1) <= 0):
        raise ValueError('No positive structural neighbors')
    retained = values.sum(1) / r.sum(1)
    q = values / values.sum(1, keepdims=True)
    a = sp.csr_matrix((values.ravel(), (rows, neighbors.ravel())), shape=(n, n))
    a = (a + a.T) * 0.5
    a.eliminate_zeros()
    degree = np.asarray(a.sum(1)).ravel()
    inv = np.divide(1, np.sqrt(degree), out=np.zeros(n), where=degree > 0)
    ahat = (sp.diags(inv) @ a @ sp.diags(inv)).tocsr()
    cost = np.empty((n, neighbors.shape[1], neighbors.shape[1]), dtype=np.float32)
    for i, idx in enumerate(neighbors):
        cost[i] = 1 - ahat[idx][:, idx].toarray()
        np.fill_diagonal(cost[i], 0)
    if np.any(cost < 0) or not np.isfinite(cost).all():
        raise ValueError('Invalid structural ground cost')
    diagnostic.update(retained_RWR_mass_mean=float(retained.mean()), alpha=alpha, topk=topk, q_zero_fraction=float(np.mean(q == 0)), cost_min=float(cost.min()), cost_max=float(cost.max()), q_entropy=float(-(q * np.log(q + 1e-30)).sum(1).mean()), local_non_diagonal_cost_mean=float(cost[:, ~np.eye(neighbors.shape[1], dtype=bool)].mean()))
    return (neighbors, q.astype(np.float32), cost, a, diagnostic)

@contextmanager
def cache_lock(path):
    with path.open('a+b') as f:
        if os.name == 'nt':
            import msvcrt
            if f.seek(0, 2) == 0:
                f.write(b'0')
                f.flush()
            f.seek(0)
            while True:
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == 'nt':
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def build_reference(h, alpha=0.2, topk=20):
    hh = hashlib.sha256(h.data.tobytes() + h.indices.tobytes() + h.indptr.tobytes() + str(h.shape).encode()).hexdigest()[:20]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with cache_lock(CACHE_DIR / f'rwr_{hh}_{alpha}.lock'):
        return _build_reference_unlocked(h, alpha, topk)

def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(False)

def sparse(m, device):
    c = m.tocoo()
    return torch.sparse_coo_tensor(np.stack([c.row, c.col]), torch.tensor(c.data, dtype=torch.float32), c.shape).coalesce().to(device)

def prepare_raw(name, cfg, device):
    x, h, paths = load_features_graph(name)
    x = x.toarray() if sp.issparse(x) else np.asarray(x)
    x = torch.tensor(x, dtype=torch.float32, device=device)
    if cfg['input_norm'] != 'none':
        x = F.normalize(x, p=1 if cfg['input_norm'] == 'l1' else 2, dim=1)
    nbr, q, cost, ar, rd = build_reference(h, cfg.get('alpha', 0.2), cfg.get('topk', 20))
    dv = np.asarray(h.sum(1)).ravel()
    de = np.asarray(h.sum(0)).ravel()
    ah = (h @ sp.diags((de + 1e-12) ** (-1)) @ h.T).tocsr()
    before_nnz = int(ah.nnz)
    before_diag = float(abs(ah.diagonal()).max())
    if cfg.get('ncut_remove_self_loop', False):
        ah.setdiag(0.0)
        ah.eliminate_zeros()
    rd.update(ncut_nnz_before=before_nnz, ncut_nnz_after=int(ah.nnz), ncut_diagonal_max_before=before_diag, ncut_diagonal_max_after=float(abs(ah.diagonal()).max()), ncut_remove_self_loop=bool(cfg.get('ncut_remove_self_loop', False)))
    inputs = dict(x=x, H=sparse(h, device), dv=torch.tensor((dv + 1e-12) ** (-0.5), device=device), de=torch.tensor((de + 1e-12) ** (-1), device=device), graph=sparse(ah, device))
    structural = dict(neighbors=torch.tensor(nbr, device=device), q=torch.tensor(q, device=device), cost=torch.tensor(cost, device=device))
    return (inputs, structural, ar, ah, rd, {f.name: sha256(f) for f in paths})

def representation(z, s, seed, k):
    singular = np.linalg.svd(z - z.mean(0), compute_uv=False)
    prob = singular / max(singular.sum(), 1e-30)
    return dict(effective_rank=float(np.exp(-(prob * np.log(prob + 1e-30)).sum())), feature_variance=float(z.var(0).mean()), per_dimension_variance=z.var(0).tolist(), mean_pairwise_cosine=float((z * z[np.random.default_rng(seed).permutation(len(z))]).sum(1).mean()), empty_clusters=int(k - len(np.unique(s.argmax(1)))), assignment_entropy=float(-(s * np.log(s + 1e-30)).sum(1).mean()), margin=float(np.diff(np.sort(s, axis=1)[:, -2:], axis=1).mean()))

def same(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype and (a.layout == b.layout)
    if a.is_sparse:
        assert torch.equal(a.indices(), b.indices()) and torch.equal(a.values(), b.values())
    else:
        assert torch.equal(a, b)

def prepare_locked(name, cfg, device, signature, key, p, meta, dh):
    cache = p.parent
    if p.exists() and meta.exists():
        m = json.loads(meta.read_text())
        assert m['signature'] == signature and sha256(p) == m['sha256'] and m['bitwise_roundtrip_verified']
        t = torch.load(p, map_location='cpu', weights_only=False)
        assert t['dh'] == dh
        return ({k: v.to(device) for k, v in t['inp'].items()}, {k: v.to(device) for k, v in t['ref'].items()}, None, None, t['rd'], dh)
    inp, ref, ar, ah, rd, actual = prepare_raw(name, cfg, device)
    assert actual == dh
    t = dict(inp={k: v.detach().cpu() for k, v in inp.items()}, ref={k: v.detach().cpu() for k, v in ref.items()}, rd=rd, dh=dh)
    temp = cache / (p.name + f'.tmp{os.getpid()}')
    torch.save(t, temp)
    reloaded = torch.load(temp, map_location='cpu', weights_only=False)
    for group in ['inp', 'ref']:
        for k, v in t[group].items():
            same(v, reloaded[group][k])
    assert reloaded['rd'] == rd and reloaded['dh'] == dh
    os.replace(temp, p)
    m = dict(signature=signature, sha256=sha256(p), bitwise_roundtrip_verified=True, contains_labels=False, tensor_shapes={group: {k: list(v.shape) for k, v in t[group].items()} for group in ['inp', 'ref']})
    meta.write_text(json.dumps(m, indent=2))
    proof = CACHE_DIR / 'input_cache_proofs'
    proof.mkdir(exist_ok=True)
    (proof / (key + '.json')).write_text(json.dumps(m, indent=2))
    print('EXACT_INPUT_CACHE_CREATED', name, flush=True)
    return (inp, ref, ar, ah, rd, actual)

def prepare(name, cfg, device):
    folder = DATA_ROOT / name
    names = ['features.npz', 'hypergraph.npz'] if (folder / 'hypergraph.npz').exists() else ['features.pickle', 'hypergraph.pickle']
    dh = {n: sha256(folder / n) for n in names}
    if dh != DATA_HASHES[name]:
        raise ValueError(f'{name}: 数据版本与固定实验不一致。需要 DATA_HASHES 中记录的原始输入文件。')
    signature = dict(dataset=name, data_sha256=dh, config={k: cfg[k] for k in ['input_norm', 'alpha', 'topk', 'ncut_remove_self_loop']}, source_sha256=INPUT_SOURCE_HASHES)
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:24]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / ('prepared_' + key + '.pt')
    meta = p.with_suffix('.json')
    with cache_lock(CACHE_DIR / (p.name + '.lock')):
        return prepare_locked(name, cfg, device, signature, key, p, meta, dh)

class HGNNLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, H, dv_inv_sqrt, de_inv):
        dv = dv_inv_sqrt.view(-1, 1) if dv_inv_sqrt.dim() == 1 else dv_inv_sqrt
        de = de_inv.view(-1, 1) if de_inv.dim() == 1 else de_inv
        x = torch.matmul(x, self.weight) * dv
        x = torch.sparse.mm(H.t(), x) * de
        x = torch.sparse.mm(H, x) * dv
        return x

class HGNNEncoder(nn.Module):

    def __init__(self, in_channels, hidden_channels, dropout=0.5, extra_residual=False):
        super().__init__()
        self.dropout = dropout
        self.extra_residual = extra_residual
        self.hgc1 = HGNNLayer(in_channels, hidden_channels)
        self.hgc2 = HGNNLayer(hidden_channels, hidden_channels)
        self.ln1 = nn.LayerNorm(hidden_channels)
        self.ln2 = nn.LayerNorm(hidden_channels)
        self.res_proj = nn.Linear(in_channels, hidden_channels) if in_channels != hidden_channels else nn.Identity()

    def forward(self, x, H, dv_inv_sqrt, de_inv):
        residual = self.res_proj(x)
        h1 = self.hgc1(x, H, dv_inv_sqrt, de_inv)
        h1 = F.leaky_relu(self.ln1(h1), 0.2)
        h1 = F.dropout(h1, self.dropout, training=self.training)
        h1 = h1 + residual
        h2 = self.hgc2(h1, H, dv_inv_sqrt, de_inv)
        h2 = F.leaky_relu(self.ln2(h2), 0.2)
        if self.extra_residual:
            return h2 + h1 + residual
        return h2 + h1

class RWOTModel(nn.Module):

    def __init__(self, inputs, k, cfg):
        super().__init__()
        d = cfg['embedding_dim']
        self.encoder = HGNNEncoder(inputs['x'].shape[1], cfg['hidden'], cfg['dropout'], extra_residual=bool(cfg.get('extra_residual', False)))
        self.topo_proj = nn.Sequential(nn.Linear(cfg['hidden'], d), nn.LayerNorm(d))
        self.cluster_output = nn.Linear(d, k)
        nn.init.orthogonal_(self.cluster_output.weight)
        nn.init.zeros_(self.cluster_output.bias)

    def forward(self, inputs):
        hidden = self.encoder(inputs['x'], inputs['H'], inputs['dv'], inputs['de'])
        z = F.normalize(self.topo_proj(hidden), dim=1)
        return dict(z=z, logits=self.cluster_output(z))

def cluster_metrics(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    size = max(y_pred.max(), y_true.max()) + 1
    weight = np.zeros((size, size), dtype=np.int64)
    for i in range(y_pred.size):
        weight[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(weight.max() - weight)
    mapping = {row: col for row, col in zip(row_ind, col_ind)}
    y_aligned = np.array([mapping.get(x, x) for x in y_pred])
    acc = weight[row_ind, col_ind].sum() / y_pred.size
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    f1 = f1_score(y_true, y_aligned, average='macro')
    return (acc, f1, nmi, ari)

def _ncut(S, A, epsilon=0.0001, average=True):
    k = S.size(1)
    if A.is_sparse:
        degree = torch.sparse.sum(A, dim=1).to_dense()
        AS = torch.sparse.mm(A, S)
    else:
        degree = A.sum(dim=1)
        AS = torch.mm(A, S)
    degree = degree.to(device=S.device, dtype=S.dtype)
    degree = degree + 0.0001 * degree.mean().clamp_min(1e-12)
    DS = S * degree.unsqueeze(1)
    volume = torch.mm(S.t(), DS)
    assoc = torch.mm(S.t(), AS)
    cut = volume - assoc
    eye = torch.eye(k, device=S.device, dtype=S.dtype)
    loss = torch.trace(torch.linalg.pinv(volume + epsilon * eye).mm(cut))
    return loss / k if average else loss

def relaxed_ncut_loss(s, graph, epsilon=0.0001):
    return _ncut(s.double(), graph.double(), epsilon=epsilon)

def sharpen_distribution(q: torch.Tensor, beta: float, eps: float=1e-12) -> torch.Tensor:
    """Sharpen an existing row-stochastic RWR distribution without labels."""
    if beta <= 0:
        raise ValueError('beta must be positive')
    if beta == 1:
        return q.clone()
    x = (q + eps).pow(float(beta))
    return x / x.sum(dim=-1, keepdim=True).clamp_min(eps)

def spread_loss(z: torch.Tensor, eps: float=1e-12) -> torch.Tensor:
    """
    L_spread = || mean_i normalize(z_i) ||^2.
    Small when normalized embeddings do not collapse to one common direction.
    """
    zn = F.normalize(z, p=2, dim=-1, eps=eps)
    mean_dir = zn.mean(dim=0)
    return mean_dir.pow(2).sum()

def row_rank_percentile(scores: torch.Tensor) -> torch.Tensor:
    """
    Convert [N,K] scores to per-row descending rank percentiles in [0,1].
    Best item -> 1, worst item -> 0.
    Stable ties preserve the supplied original RWR support-column ordering.
    """
    if scores.ndim != 2:
        raise ValueError('scores must be [N,K]')
    n, k = scores.shape
    if k == 1:
        return torch.ones_like(scores)
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    base = torch.arange(k, device=scores.device).view(1, -1).expand(n, -1)
    ranks.scatter_(1, order, base)
    return 1.0 - ranks.to(scores.dtype) / float(k - 1)

def candidate_cosine(x: torch.Tensor, neighbor_index: torch.Tensor, eps: float=1e-12) -> torch.Tensor:
    """
    Dense implementation for pairwise cosine only on supplied candidate indices.
    x: [N,D], neighbor_index: [N,K].
    For sparse X, implement the same operation with sparse row normalization
    without densifying the full matrix.
    """
    xn = F.normalize(x, p=2, dim=1, eps=eps)
    center = xn.unsqueeze(1)
    neigh = xn[neighbor_index]
    return (center * neigh).sum(dim=-1)

def reliability_scores(q: torch.Tensor, x_cos: torch.Tensor, mode: str, reciprocal: torch.Tensor | None=None, reciprocal_bonus: float=0.25, eps: float=1e-12) -> torch.Tensor:
    u_r = row_rank_percentile(q)
    u_x = row_rank_percentile(x_cos)
    if mode == 'feature_rank':
        return u_x
    agreement = torch.sqrt((u_r + eps) * (u_x + eps))
    if mode == 'structure_feature_geomean':
        return agreement
    if mode == 'structure_feature_geomean_reciprocal':
        if reciprocal is None:
            raise ValueError('reciprocal mask required')
        return agreement * (1.0 + reciprocal_bonus * reciprocal.to(agreement.dtype))
    raise ValueError(f'unknown mode: {mode}')

def sharpen_q(q: torch.Tensor, beta: float=4.0, eps: float=1e-12) -> torch.Tensor:
    x = (q.detach().double() + eps).pow(beta)
    return (x / x.sum(dim=1, keepdim=True)).to(q.dtype).clamp_min(torch.finfo(q.dtype).tiny).detach()

def soft_reweight_linear(q_sharp: torch.Tensor, reliability: torch.Tensor, lambda_r: float, eps: float=1e-12) -> torch.Tensor:
    if not 0.0 <= lambda_r < 1.0:
        raise ValueError('lambda_r must be in [0,1)')
    if q_sharp.shape != reliability.shape:
        raise ValueError('shape mismatch')
    w = 1.0 + lambda_r * (2.0 * reliability - 1.0)
    if torch.any(w <= 0):
        raise RuntimeError('linear reliability weight must remain positive')
    mass = q_sharp.detach().double() * w.detach().double()
    return (mass / mass.sum(dim=1, keepdim=True)).to(q_sharp.dtype).clamp_min(torch.finfo(q_sharp.dtype).tiny).detach()

def soft_reweight_exponential(q_sharp: torch.Tensor, reliability: torch.Tensor, kappa: float, eps: float=1e-12) -> torch.Tensor:
    if kappa < 0:
        raise ValueError('kappa must be non-negative')
    if q_sharp.shape != reliability.shape:
        raise ValueError('shape mismatch')
    centered = reliability - reliability.mean(dim=1, keepdim=True)
    w = torch.exp(kappa * centered)
    mass = q_sharp.detach().double() * w.detach().double()
    return (mass / mass.sum(dim=1, keepdim=True)).to(q_sharp.dtype).clamp_min(torch.finfo(q_sharp.dtype).tiny).detach()

def feature_mask_view(x: torch.Tensor, rate: float) -> torch.Tensor:
    if rate <= 0:
        return x
    keep = torch.rand_like(x) >= float(rate)
    return x * keep.to(x.dtype)

def membership_dropout(hyperedge_index: torch.Tensor, rate: float) -> torch.Tensor:
    """
    Drop existing node-hyperedge memberships only.
    Does not add synthetic memberships.
    """
    if rate <= 0:
        return hyperedge_index
    e = hyperedge_index.size(1)
    keep = torch.rand(e, device=hyperedge_index.device) >= float(rate)
    out = hyperedge_index[:, keep]
    if out.size(1) == 0:
        return hyperedge_index
    return out

def symmetric_infonce(z1: torch.Tensor, z2: torch.Tensor, tau: float=0.5, node_ids: torch.Tensor | None=None) -> torch.Tensor:
    """
    Full or sampled-node symmetric InfoNCE.
    z1/z2 are complete encoder outputs.
    """
    if node_ids is not None:
        z1 = z1[node_ids]
        z2 = z2[node_ids]
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    logits = z1 @ z2.t()
    logits = logits / float(tau)
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

def view(inp, pf=0.4, pm=0.4):
    h = inp['H'].coalesce()
    ix = membership_dropout(h.indices(), pm)
    assert bool((h.values() == 1).all())
    hh = torch.sparse_coo_tensor(ix, torch.ones(ix.shape[1], device=h.device), h.shape).coalesce()
    dv = torch.sparse.sum(hh, dim=1).to_dense()
    de = torch.sparse.sum(hh, dim=0).to_dense()
    return dict(inp, x=feature_mask_view(inp['x'], pf), H=hh, dv=torch.where(dv > 0, dv.clamp_min(1e-12).rsqrt(), 0.0), de=torch.where(de > 0, de.clamp_min(1e-12).reciprocal(), 0.0))

def ot_value(a, b, cost, logk, iterations):
    a = a.clamp_min(1e-12)
    a = a / a.sum(-1, keepdim=True).clamp_min(1e-12)
    b = b.clamp_min(1e-12)
    b = b / b.sum(-1, keepdim=True).clamp_min(1e-12)
    la, lb = (a.log(), b.log())
    u, v = (torch.zeros_like(a), torch.zeros_like(b))
    for _ in range(iterations):
        u = la - torch.logsumexp(logk + v[:, None, :], 2)
        v = lb - torch.logsumexp(logk + u[:, :, None], 1)
    lp = u[:, :, None] + logk + v[:, None, :]
    pi = lp.exp()
    transport = (pi * cost).sum((1, 2))
    kl = (pi * (lp - (la[:, :, None] + lb[:, None, :]))).sum((1, 2))
    return (transport, kl, (pi.sum(2) - a).abs().amax(1).detach(), (pi.sum(1) - b).abs().amax(1).detach())

class FastOTEngine:

    def __init__(self, z, ref, cfg, batch_size=2048, transport_diagnostic=True):
        assert cfg['solver'] == 'fixed30' and 'active' not in ref
        self.cfg = cfg
        self.iters = int(cfg.get('sinkhorn_iters', 30))
        self.batch_size = batch_size
        self.transport_diagnostic = transport_diagnostic
        self.ref = ref
        self.signature = tuple(((ref[k].data_ptr(), ref[k]._version) for k in ['neighbors', 'q', 'cost']))
        self.logk = -ref['cost'].detach() / cfg['epsilon_ot']
        qq, rr, cc = ([], [], [])
        with torch.no_grad():
            for start in range(0, len(z), batch_size):
                s = slice(start, start + batch_size)
                q = ref['q'][s].detach()
                t, h, r, c = ot_value(q, q, ref['cost'][s], self.logk[s], self.iters)
                qq.append(t + cfg['epsilon_ot'] * h)
                rr.append(r)
                cc.append(c)
        self.qq, self.qr, self.qc = map(torch.cat, (qq, rr, cc))

    def compute(self, z, ref, gradient=True):
        assert tuple(((ref[k].data_ptr(), ref[k]._version) for k in ['neighbors', 'q', 'cost'])) == self.signature, 'Reference mutated; build a new engine'
        eps = self.cfg['epsilon_ot']
        n = len(z)
        with torch.set_grad_enabled(gradient):
            proxy = z.detach().requires_grad_(gradient)
            zn = F.normalize(proxy, dim=1)
            p = ((zn[:, None, :] * zn[ref['neighbors']]).sum(-1) / self.cfg['tau_sem']).softmax(1)
            gp = torch.empty_like(p) if gradient else None
            values, rows, cols, transports = ([], [], [], [])
            for start in range(0, n, self.batch_size):
                s = slice(start, start + self.batch_size)
                pb = p[s].detach().requires_grad_(gradient)
                q = ref['q'][s].detach()
                c = ref['cost'][s].detach()
                tq, hq, rq, cq = ot_value(q, pb, c, self.logk[s], self.iters)
                tp, hp, rp, cp = ot_value(pb, pb, c, self.logk[s], self.iters)
                div = tq + eps * hq - 0.5 * self.qq[s] - 0.5 * (tp + eps * hp)
                if gradient:
                    gp[s] = torch.autograd.grad(div.sum() / n, pb)[0]
                values.append(div.detach())
                rows.append(torch.stack([rq, self.qr[s], rp]).amax(0))
                cols.append(torch.stack([cq, self.qc[s], cp]).amax(0))
                if self.transport_diagnostic:
                    with torch.no_grad():
                        td, _, _, _ = ot_value(q.double(), pb.detach().double(), c.double(), -c.double() / eps, self.iters)
                        transports.append(td)
            g = torch.autograd.grad(p, proxy, gp)[0] if gradient else None
        div = torch.cat(values)
        rr = torch.maximum(torch.cat(rows), torch.cat(cols))
        total = float(div.sum() / n)
        pe = -(p.detach().clamp_min(1e-12) * p.detach().clamp_min(1e-12).log()).sum() / n
        q = ref['q'].detach()
        qe = -(q.clamp_min(1e-12) * q.clamp_min(1e-12).log()).sum() / n
        return (total, g, dict(true_transport_cost=float(torch.cat(transports).sum() / n) if transports else None, entropy_regularized_objective=total, optimization_surrogate=total, max_residual=float(rr.max()), mean_residual=float(rr.mean()), residual_gt_threshold_fraction=float((rr >= 0.0001).double().mean()), mean_iterations=float(self.iters), p90_iterations=float(self.iters), max_iterations=self.iters, max_iter_limit=self.iters, max_iter_hit_rate=None, initial_batch_max_iter_hit_rate_max=None, max_iter_escalations=0, q_entropy=float(qe), p_entropy=float(pe), gradient_norm_ot_to_z=float(g.norm()) if g is not None else None, all_converged=bool((rr < 0.0001).all()), negative_divergence_fraction=float((div < -1e-06).float().mean()), objective_kind='debiased_Sinkhorn_divergence', anchor_batch_size=self.batch_size))

def optional_spread(z, weight, spread_fn):
    return spread_fn(z) if float(weight) != 0.0 else z.new_zeros(())

def compose_base(ncut, spread, ncut_weight, spread_weight):
    base = ncut_weight * ncut
    if float(spread_weight) != 0.0:
        base = base + spread_weight * spread
    return base

def metrics(y, p):
    a, f, n, r = cluster_metrics(y, p)
    return dict(acc=float(a), nmi=float(n), ari=float(r), f1=float(f))

def save(folder, model, cfg, epoch, z, s, pred, centers=None, selected=None):
    for n, a in [('Z', z), ('S', s), ('prediction', pred)]:
        np.save(folder / (n + '.npy'), a)
    torch.save(dict(state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, config=cfg, epoch=epoch, centers=centers, selected_metrics=selected), folder / 'checkpoint.pt')
    if selected is not None:
        write_json(folder / 'best.json', selected)

def snapshot(p, model, inp, cfg, seed, epoch):
    p.mkdir()
    model.eval()
    with torch.no_grad():
        o = model(inp)
        z = o['z'].cpu().numpy()
        s = (o['logits'] / max(cfg['temp_min'], cfg['temp_start'] * cfg['temp_decay'] ** (epoch - 1))).softmax(1).cpu().numpy()
    km = KMeans(n_clusters=CLUSTERS[cfg['dataset']], n_init=100, random_state=seed).fit(z)
    save(p, model, cfg, epoch, z, s, km.labels_, km.cluster_centers_)
    np.save(p / 'head_prediction.npy', s.argmax(1))
    write_json(p / 'representation.json', representation(z, s, seed, CLUSTERS[cfg['dataset']]))
    ck = torch.load(p / 'checkpoint.pt', map_location='cuda', weights_only=False)
    model.load_state_dict(ck['state_dict'])
    with torch.no_grad():
        zz = model(inp)['z'].cpu().numpy()
    from sklearn.metrics import pairwise_distances_argmin
    assert np.allclose(zz, z, rtol=5e-05, atol=5e-05)
    assert np.array_equal(pairwise_distances_argmin(zz, ck['centers']), km.labels_)
    write_json(p / 'reload_verification.json', dict(actual_reload=True, checkpoint_sha256=sha256(p / 'checkpoint.pt')))

def train_one(name, seed, output, epochs_override=None):
    cfg = dict(CONFIGS[name])
    if epochs_override is not None:
        cfg.update(epochs=int(epochs_override), final_epoch=int(epochs_override))
    args = SimpleNamespace(dataset=name, seed=seed, ssl=cfg['lambda_disc'], epochs=cfg['epochs'], smoke=epochs_override is not None)
    p = Path(output).resolve()
    if p.exists():
        raise FileExistsError(f'不覆盖已有输出: {p}')
    p.mkdir(parents=True)
    endpoints = {args.epochs}
    write_json(p / 'config.json', dict(model=cfg, seed=seed, lambda_disc=args.ssl, view_feature_mask=0.15, view_membership_drop=0.1, tau=0.5, endpoints=sorted(endpoints)))
    torch.set_num_threads(4)
    started = time.time()
    history = []
    torch.cuda.reset_peak_memory_stats()
    with prohibit_label_io():
        inp, ref, _, _, rd, dh = prepare(args.dataset, cfg, torch.device('cuda'))
        seed_all(args.seed)
        model = RWOTModel(inp, CLUSTERS[args.dataset], cfg).cuda()
        raw = ref['q'].clone()
        ref = dict(ref, q=sharpen_distribution(raw, cfg['beta']))
        if cfg['soft_family'] != 'none':
            nbr = ref['neighbors']
            rec = (nbr[nbr] == torch.arange(len(nbr), device='cuda')[:, None, None]).any(-1)
            xc = candidate_cosine(inp['x'], nbr)
            rel = (reliability_scores(raw, xc, 'structure_feature_geomean_reciprocal', reciprocal=rec) / 1.25).clamp(0, 1).detach()
            qs = sharpen_q(raw, cfg['beta'])
            qr = soft_reweight_linear(qs, rel, cfg['intensity']) if cfg['soft_family'] == 'linear' else soft_reweight_exponential(qs, rel, cfg['intensity'])
            ref = dict(ref, q=qr)
        engine = FastOTEngine(inp['x'], ref, cfg, batch_size=4096)
        opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 400, eta_min=cfg['min_lr'])
        for epoch in range(1, args.epochs + 1):
            model.train()
            opt.zero_grad(set_to_none=True)
            o = model(inp)
            z = o['z']
            s = (o['logits'] / max(cfg['temp_min'], cfg['temp_start'] * cfg['temp_decay'] ** (epoch - 1))).softmax(1)
            nc = relaxed_ncut_loss(s, inp['graph'], epsilon=cfg['ncut_eps'])
            sp = optional_spread(z, cfg['lambda_spread'], spread_loss)
            value, og, diag = engine.compute(z, ref)
            base = compose_base(nc, sp, cfg['lambda_ncut'], cfg['lambda_spread'])
            ssl = z.sum() * 0
            if args.ssl:
                v1 = model(view(inp, 0.15, 0.1))['z']
                v2 = model(view(inp, 0.15, 0.1))['z']
                ids = torch.randperm(len(z), device=z.device)[:4096] if len(z) > 5000 else None
                ssl = symmetric_infonce(v1, v2, 0.5, ids)
                base = base + args.ssl * ssl
            assert torch.isfinite(base) and torch.isfinite(og).all() and np.isfinite(value)
            torch.autograd.backward([base, z], [None, cfg['lambda_ot'] * og])
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
            opt.step()
            sched.step()
            history.append(dict(epoch=epoch, rwot=value, ncut=float(nc.detach()), spread=float(sp.detach()), weighted_spread=cfg['lambda_spread'] * float(sp.detach()), ssl=float(ssl.detach()), grad_norm=float(gn), residual=diag['max_residual'], seconds=time.time() - started))
            if epoch in endpoints:
                snapshot(p / f'epoch_{epoch}', model, inp, cfg, args.seed, epoch)
            if epoch % 10 == 0 or epoch == args.epochs:
                write_json(p / 'history.json', history)
                write_json(p / 'progress.json', dict(epoch=epoch, total=args.epochs, seconds=time.time() - started))
                print(args.dataset, args.ssl, epoch, flush=True)
    files = {x.relative_to(p).as_posix(): sha256(x) for x in p.rglob('*') if x.is_file()}
    write_json(p / 'training.json', dict(status='complete', dataset=args.dataset, seed=args.seed, epochs=args.epochs, smoke=args.smoke, labels_read_during_training=False, data_sha256=dh, files=files, seconds=time.time() - started, peak_gpu_allocated_mb=torch.cuda.max_memory_allocated() / 2 ** 20, peak_gpu_reserved_mb=torch.cuda.max_memory_reserved() / 2 ** 20))

def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def evaluate_runs(runs, output):
    for name, seed, p in runs:
        record = json.loads((p / 'training.json').read_text())
        assert record['status'] == 'complete' and (not record['labels_read_during_training'])
        assert all((sha256(p / k) == v for k, v in record['files'].items()))
    rows = []
    labels = {}
    for name, seed, p in runs:
        if name not in labels:
            labels[name] = load_labels(name)
        epoch = CONFIGS[name]['epochs']
        snap = p / f'epoch_{epoch}'
        for method, file in [('KMeans(Z)', 'prediction.npy'), ('argmax(S)', 'head_prediction.npy')]:
            pred = np.load(snap / file)
            if len(pred) != len(labels[name]):
                raise ValueError(f'{name}: 标签与节点数不一致')
            rows.append(dict(dataset=name, seed=seed, epoch=epoch, method=method, **metrics(labels[name], pred)))
    write_csv(output / 'per_seed.csv', rows)
    summary = []
    for name in dict.fromkeys((x[0] for x in runs)):
        for method in ['KMeans(Z)', 'argmax(S)']:
            rr = [x for x in rows if x['dataset'] == name and x['method'] == method]
            summary.append(dict(dataset=name, method=method, epoch=rr[0]['epoch'], n_seeds=len(rr), **{k: statistics.mean((x[k] for x in rr)) for k in ['acc', 'nmi', 'ari', 'f1']}, **{k + '_std': statistics.pstdev((x[k] for x in rr)) for k in ['acc', 'nmi', 'ari', 'f1']}))
    write_csv(output / 'summary.csv', summary)
    for x in summary:
        if x['method'] == 'KMeans(Z)':
            print(x['dataset'], ' | '.join((f"{k.upper()}={100 * x[k]:.2f} +/- {100 * x[k + '_std']:.2f}" for k in ['acc', 'nmi', 'ari', 'f1'])), flush=True)

def main():
    global DATA_ROOT, CACHE_DIR
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--dataset', choices=['all'] + list(CONFIGS), default='all')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, help='可复用的精确结构缓存目录，默认 OUTPUT/cache')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('复现实验需要 CUDA 版 PyTorch 和 NVIDIA GPU')
    if len(set(args.seeds)) != len(args.seeds) or any((s < 0 or s > 4 for s in args.seeds)):
        parser.error('seeds 应为 0..4 中不重复的编号')
    DATA_ROOT = args.data_root.resolve()
    output = args.output.resolve()
    CACHE_DIR = args.cache_dir.resolve() if args.cache_dir else output / 'cache'
    if output.exists():
        parser.error('请指定尚不存在的输出目录，避免覆盖结果')
    output.mkdir(parents=True)
    names = list(CONFIGS) if args.dataset == 'all' else [args.dataset]
    write_json(output / 'frozen_config.json', dict(datasets={n: CONFIGS[n] for n in names}, seeds=args.seeds, data_hashes={n: DATA_HASHES[n] for n in names}, code_sha256=sha256(Path(__file__))))
    runs = []
    for name in names:
        for seed in args.seeds:
            p = output / name.replace('/', '__') / f'seed{seed}'
            train_one(name, seed, p)
            runs.append((name, seed, p))
            gc.collect()
            torch.cuda.empty_cache()
    evaluate_runs(runs, output)
if __name__ == '__main__':
    main()
