"""合成データセットを生成する。例: python generate_dataset.py --n-train 400 --n-val 100"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from scene import load_config  # noqa: E402
from dataset import generate  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = load_config(a.config)
    y = generate(cfg, cfg["paths"]["dataset_dir"], a.n_train, a.n_val, seed=a.seed)
    print("dataset yaml:", y)
