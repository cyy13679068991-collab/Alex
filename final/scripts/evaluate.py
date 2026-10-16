from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rwot.api import parser, predict, core
p = parser('Evaluate network argmax(S) on the frozen dataset labels.')
p.add_argument('--checkpoint', type=Path)
p.add_argument('--device', default='cuda')
if __name__ == '__main__':
    a = p.parse_args()
    name, seed, pred = predict(a)
    scores = core.metrics(core.load_labels(name), pred)
    core.write_json(a.output / 'metrics.json', dict(dataset=name, seed=seed, **scores))
    print(scores)
