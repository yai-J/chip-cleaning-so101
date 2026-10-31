"""合成データセット生成：俯瞰カメラ画像と、シミュレーション真値のマスクから YOLO-seg 形式のラベルを作る。

ランダム化する要素：切粉の個数・配置・向き、回収トレー内の切粉、照明の明るさ、
台と切粉の色味、アームの姿勢（映り込み）。
"""
import os
import numpy as np
import cv2
import mujoco

from scene import ChipScene, HOME_Q
from detector import GTDetector
from ik import BrushIK, brush_down_R


def _randomize_appearance(sc, rng, base):
    m = sc.model
    m.light_diffuse[:] = base["light"] * rng.uniform(0.6, 1.3)
    g = rng.uniform(0.10, 0.26)
    m.geom_rgba[m.geom("table").id, :3] = [g, g * rng.uniform(0.95, 1.05), g * rng.uniform(0.95, 1.1)]
    c = rng.uniform(0.6, 0.92)
    m.mat_rgba[m.material("chip_mat").id, :3] = [c, c, c * rng.uniform(0.95, 1.05)]


def _random_arm(sc, ik, rng):
    if rng.random() < 0.5:
        return HOME_Q + rng.normal(0, 0.05, 5)
    p = [rng.uniform(0.13, 0.25), rng.uniform(-0.10, 0.10), rng.uniform(0.012, 0.06)]
    q, err, ok = ik.solve(p, brush_down_R(0.0), np.array([0, -0.5, 0.5, 1.2, 0]), iters=200)
    return q if ok else HOME_Q


def generate(cfg, out_dir, n_train=400, n_val=100, seed=0, root=".", log=print):
    rng = np.random.default_rng(seed)
    sc = ChipScene(cfg, root)
    ik = BrushIK(sc.model)
    ct = cfg["camera_top"]
    W, H = ct["width"], ct["height"]
    r = mujoco.Renderer(sc.model, H, W)
    det = GTDetector(sc, r)
    base = {"light": sc.model.light_diffuse.copy()}
    for split, n in [("train", n_train), ("val", n_val)]:
        os.makedirs(f"{out_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{out_dir}/labels/{split}", exist_ok=True)
        for i in range(n):
            _randomize_appearance(sc, rng, base)
            sc.set_arm(HOME_Q)
            sc.scatter_chips(rng, n=int(rng.integers(5, 45)), clustered=rng.random() < 0.8)
            # 一部の切粉を回収トレー内に置く（回収済みの状態も学習させる）
            for j in np.where(sc.active)[0]:
                if rng.random() < 0.15:
                    a = sc.chip_qadr[j]
                    sc.data.qpos[a:a + 3] = [rng.uniform(0.07, 0.31), rng.uniform(-0.19, -0.115), -0.04]
            sc.settle(0.3)
            q = _random_arm(sc, ik, rng)
            sc.data.qpos[sc.arm_qadr] = q  # 描画用に姿勢だけ変更（物理は進めない）
            mujoco.mj_forward(sc.model, sc.data)
            r.update_scene(sc.data, camera="top")
            img = r.render()
            polys = det.polygons()
            name = f"{split}_{i:05d}"
            cv2.imwrite(f"{out_dir}/images/{split}/{name}.jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(f"{out_dir}/labels/{split}/{name}.txt", "w") as f:
                for p in polys:
                    pn = p / np.array([W, H])
                    f.write("0 " + " ".join(f"{v:.5f}" for v in pn.flatten()) + "\n")
            if (i + 1) % 50 == 0:
                log(f"{split}: {i + 1}/{n}")
    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(out_dir)}\ntrain: images/train\nval: images/val\nnames:\n  0: chip\n")
    return yaml_path
