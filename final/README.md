# RWOT

Attributed hypergraph clustering with an HGNN encoder, RWR reference, RWOT,
NCut, InfoNCE and spread losses. Training consists of fixed-epoch pretraining,
unlabeled KMeans center initialization (`n_init=100`), and ten joint fine-tuning
steps with learning-rate multiplier 0.1. Prediction is network `argmax(S)`;
inference does not fit KMeans.

## Reproduce

Linux, Python 3.9 with `venv`, Git, and an NVIDIA GPU/driver compatible with
CUDA 12.1 are required. Allow sufficient RAM and disk for dense RWR preprocessing
on the large datasets. The validated environment used an RTX A4000 (16 GB).

```bash
bash reproduce.sh
```

This creates a local virtual environment, installs pinned dependencies, downloads
the pinned CAHC data, verifies input SHA256, and trains all ten datasets with
seeds **101, 102, 103, 104, 105** for every dataset. Internet access is needed for dependencies and datasets.
Model parameters and fixed epochs are in `configs/final.json`; the joint protocol
is in `configs/joint.json`. The configuration contains the fixed selected parameters and epochs.

To use an existing compatible environment and dataset directory:

```bash
RWOT_SKIP_INSTALL=1 PYTHON=/path/to/python bash reproduce.sh --data-root /path/to/data
```

Matching local files are reused. Data layout is
`data/<dataset>/{features.pickle,hypergraph.pickle,labels.pickle}`, where dataset
names are listed in `configs/datasets.json`. The downloader uses
`https://github.com/nilics/CAHC` at commit
`e6975ea33b0734255137e7592f83f7fcd325405d` (`data/dataset/`).
Labels are accessed only during evaluation, after predictions are saved.

For a small full-training check or a no-compute job listing:

```bash
bash reproduce.sh --dataset cocitation/cora --seeds 101
bash reproduce.sh --dry-run
```

Rerun the same command to reuse completed jobs. Incomplete jobs are never silently
overwritten: move the named partial job/evaluation directory aside before retrying.

## Outputs

`runs/reproduction/` contains checkpoints, predictions, logs, fresh per-seed
ACC/NMI/ARI/Macro-F1, five-seed means and population standard deviations, and
reference comparisons. `results/per_seed.csv` is the fixed comparison
fixture, never a substitute for freshly trained results. No pretrained weights
are needed or bundled.

CUDA results may vary across hardware. The runner retains the original acceptance
thresholds: 1 percentage point for means, 1.5 for standard deviations, and 3 for
individual seed metrics. It exits with an error when a threshold is exceeded,
while preserving actual results; one-command execution is not a guarantee of
identical scores. The historical NTU2012 incidence orientation is retained.
The protocol is transductive and uses known cluster counts; frozen hyperparameters
were selected with label-assisted development.

## Individual training and inference

```bash
python scripts/train.py --data-root data --dataset cocitation/cora --seed 101 --output runs/manual
python scripts/predict.py --data-root data --checkpoint runs/manual/cocitation__cora/seed101/final/checkpoint.pt --output runs/prediction --device cuda
python scripts/evaluate.py --data-root data --checkpoint runs/manual/cocitation__cora/seed101/final/checkpoint.pt --output runs/evaluation --device cuda
```

`predict.py` does not read labels. `evaluate.py` reloads the network before scoring.
