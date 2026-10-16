from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rwot.api import parser, predict
p = parser('Label-free network clustering; no KMeans at inference.')
p.add_argument('--checkpoint', type=Path)
p.add_argument('--device', default='cuda')
if __name__ == '__main__':
    predict(p.parse_args())
