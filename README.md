# RWOT

Standalone HGNN + RWOT + NCut + InfoNCE + spread implementation with frozen configurations for seven datasets.

## Run

Use Python 3.9 and an NVIDIA GPU with CUDA-enabled PyTorch.

```bash
pip install -r requirements.txt
python train.py --data-root /path/to/data --output runs/five_seed
```

`configs/final.json` contains `seeds: [0, 1, 2, 3, 4]` and the dataset parameters under `datasets`. Training reads these seeds by default and runs all seven datasets sequentially. To run only Cora:

```bash
python train.py --data-root /path/to/data --dataset cocitation/cora --output runs/cora
```

The output directory must not already exist. Training saves checkpoints and predictions, then loads labels after all requested training runs finish. Outputs include per-seed metrics and mean/population-standard-deviation summaries for KMeans(Z) and argmax(S). Parameters and fixed epochs were selected with historical label-assisted tuning; these are not independent test results.

## Data

Provide raw data separately using this directory layout:

```text
data_root/
  cocitation/cora/
  cocitation/citeseer/
  cocitation/pubmed/
  coauthorship/dblp/
  20newsW100/
  Mushroom/
  NTU2012/
```

Each directory contains SciPy sparse `features.npz`, `hypergraph.npz`, and NumPy `labels.npy`; alternatively use `features.pickle`, `hypergraph.pickle`, and `labels.pickle`. Pickle features are dense/sparse matrices; the pickle hypergraph maps hyperedges to lists of zero-based node indices. NPZ incidence matrices are node-by-hyperedge. Labels are class IDs or one-hot rows. Node ordering must match across all inputs. Load only trusted pickle files.

`configs/datasets.json` records class counts and the original input file SHA256 hashes, which are checked on loading. Pubmed is the 3,840-node subset; 20newsW100 is the four-class version. Data is not downloaded automatically.

Initial RWR preprocessing can be slow and memory intensive. Later seeds reuse the cache. Use `--cache-dir /path/to/cache` to reuse a compatible cache across runs.

## Files

- `train.py`: training entry point.
- `rwot/core.py`: complete model, losses, preprocessing, training and evaluation.
- `configs/final.json`: five seeds and frozen dataset parameters.
- `configs/datasets.json`: input versions and class counts.
