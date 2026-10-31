"""YOLO セグメンテーション（1クラス: chip）を学習する。例: python train_yolo.py --epochs 50"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from scene import load_config  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default="yolo11n-seg.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    from ultralytics import YOLO
    model = YOLO(a.model)
    model.train(data=os.path.join(cfg["paths"]["dataset_dir"], "data.yaml"), epochs=a.epochs, imgsz=a.imgsz,
            batch=a.batch, device=a.device, project=os.path.abspath("runs"), name="chip_seg", exist_ok=True,
                fliplr=0.0, mosaic=0.5, seed=cfg["seed"], plots=True)
    metrics = YOLO("runs/chip_seg/weights/best.pt").val(data=os.path.join(cfg["paths"]["dataset_dir"], "data.yaml"),
                                                        imgsz=a.imgsz, device=a.device)
    print(f"mask mAP50 = {metrics.seg.map50:.3f}, mask mAP50-95 = {metrics.seg.map:.3f}")
