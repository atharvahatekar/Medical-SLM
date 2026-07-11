from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

"""Entry point: python scripts/train.py --config configs/small.yaml"""

import argparse
from src.config import Config
from src.trainer import Trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/small.yaml")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="runs/small")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    Trainer(cfg, data_dir=args.data_dir, out_dir=args.out_dir,
            resume=not args.no_resume).fit()


if __name__ == "__main__":
    main()
