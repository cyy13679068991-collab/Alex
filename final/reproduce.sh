#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ " $* " == *" --dry-run "* ]]; then
  exec "${PYTHON:-python3}" scripts/reproduce.py "$@"
fi
if [[ " $* " == *" --no-download "* ]]; then
  echo 'Use without --download to skip downloading datasets.' >&2; exit 2
fi
# Defaults: install exact packages in a local venv, use pinned source data, run ten datasets x five seeds.
if [[ "${RWOT_SKIP_INSTALL:-0}" != "1" ]]; then
  PY="${PYTHON:-python3.9}"
  command -v "$PY" >/dev/null || { echo "Python 3.9 required; set PYTHON=/path/to/python3.9" >&2; exit 2; }
  if [[ ! -e .venv/bin/python ]]; then "$PY" -m venv .venv; fi
  .venv/bin/python -m pip install -r requirements.txt
  exec .venv/bin/python scripts/reproduce.py --download --resume "$@"
else
  exec "${PYTHON:-python3}" scripts/reproduce.py --download --resume "$@"
fi
