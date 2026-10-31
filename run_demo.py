"""清掃デモを実行して動画を出力する。

例: python run_demo.py                      # 学習済み YOLO で実行（本番）
    python run_demo.py --detector gt        # 真値マスクで実行（YOLO 学習前の動作確認）
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import numpy as np  # noqa: E402
import mujoco  # noqa: E402
from scene import load_config, ChipScene, HOME_Q  # noqa: E402
from ik import BrushIK  # noqa: E402
from planner import StrokePlanner  # noqa: E402
from detector import GTDetector, YoloDetector  # noqa: E402
from controller import CleaningController  # noqa: E402
from video import Recorder  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--detector", choices=["yolo", "gt"], default="yolo")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--min-seconds", type=float, default=75, help="動画がこの長さに達するまで加工サイクルを追加")
    ap.add_argument("--max-seconds", type=float, default=170)
    ap.add_argument("--max-cycles", type=int, default=6)
    ap.add_argument("--scale", type=float, default=1.0, help="動画の解像度倍率（動作確認時は 0.5 など）")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    out_dir = cfg["paths"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out = a.out or os.path.join(out_dir, f"demo_{a.detector}.mp4")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    rng = np.random.default_rng(a.seed)

    sc = ChipScene(cfg)
    sc.set_arm(HOME_Q)
    ik = BrushIK(sc.model)
    ct = cfg["camera_top"]
    if a.detector == "yolo":
        det = YoloDetector(a.weights or cfg["paths"]["weights"], conf=cfg["detector"]["conf"],
                           imgsz=cfg["detector"]["imgsz"])
    else:
        det = GTDetector(sc, mujoco.Renderer(sc.model, ct["height"], ct["width"]))
    rec = Recorder(sc.model, out, cfg, scale=a.scale)
    ctl = CleaningController(sc, ik, StrokePlanner(cfg), det, cfg, recorder=rec)

    results = []
    ctl._set_cmd(sc.arm_q())
    c = 0
    # 1サイクルの目安は約40秒。最低長に届くまでサイクルを追加し、上限を超えそうなら止める
    while c < a.max_cycles and rec.seconds() < a.min_seconds and rec.seconds() + 45 < a.max_seconds:
        c += 1
        rec.status.update(cycle=c, n_cycles="-")
        print(f"===== 加工サイクル {c} =====")
        n = ctl.new_chips(rng)
        r = ctl.run()
        r.update(cycle=c, n_new_chips=n, video_seconds=rec.seconds())
        results.append(r)
    rec.close()
    summary = {"detector": a.detector, "cycles": results, "video_seconds": rec.seconds(),
               "total_strokes": int(sum(r["strokes"] for r in results)),
               "mean_collected_ratio_gt": float(np.mean([r["collected_ratio_gt"] for r in results]))}
    with open(os.path.splitext(out)[0] + "_log.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"動画: {out}（{rec.seconds():.1f} 秒, 等速）")
    print(json.dumps({k: v for k, v in summary.items() if k != "cycles"}, ensure_ascii=False))
