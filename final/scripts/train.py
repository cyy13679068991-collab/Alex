from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rwot.api import parser, train
p = parser('Full-spread pretraining, saved-center initialization and ten-step joint refinement.')
p.add_argument('--pretrained', type=Path, help='Optional matching fixed-epoch pretraining checkpoint with centers')
if __name__ == '__main__':
    train(p.parse_args())
