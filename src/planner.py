"""ストローク計画：検出した切粉を塊に分け、最大の塊を右側の回収トレーへ掃く直線を作る。"""
import numpy as np


def cluster_points(xy, radius):
    """距離 radius 以内で連結する点を同じ塊とみなす（DBSCAN の簡易版、外部ライブラリ不要）。"""
    n = len(xy)
    labels = -np.ones(n, int)
    k = 0
    for i in range(n):
        if labels[i] >= 0:
            continue
        stack = [i]
        labels[i] = k
        while stack:
            j = stack.pop()
            near = np.where((labels < 0) & (np.linalg.norm(xy - xy[j], axis=1) < radius))[0]
            labels[near] = k
            stack.extend(near.tolist())
        k += 1
    return labels


class StrokePlanner:
    def __init__(self, cfg):
        self.cfg = cfg
        p = cfg["planner"]
        self.radius = p["cluster_radius"]
        self.back = p["back_margin"]
        self.end_y = p["end_y"]
        self.edge_y = cfg["scene"]["table_edge_y"]
        self.x_lim = (0.13, 0.24)          # 到達性を確認済みの範囲
        self.y_start_max = 0.12
        self.history = []

    def on_table(self, pts):
        """回収トレーに落ちていない（台の上にある）切粉だけを残す。"""
        if len(pts) == 0:
            return pts
        return pts[pts[:, 1] > self.edge_y + 0.003]

    def plan(self, pts):
        """pts: 台上の切粉の world 座標 (N,3)。戻り値: dict(start, end, n_target) または None。"""
        pts = self.on_table(pts)
        if len(pts) == 0:
            return None
        xy = pts[:, :2]
        labels = cluster_points(xy, self.radius)
        counts = np.bincount(labels)
        best = int(np.argmax(counts))
        c = xy[labels == best]
        x = float(np.clip(np.median(c[:, 0]), *self.x_lim))
        # 同じ場所を空振りし続けないよう、直前と同じなら左右にずらす
        if self.history and abs(self.history[-1] - x) < 0.01 and len(self.history) >= 2 \
                and abs(self.history[-2] - x) < 0.01:
            x = float(np.clip(x + (0.025 if len(self.history) % 2 else -0.025), *self.x_lim))
        self.history.append(x)
        y0 = float(min(c[:, 1].max() + self.back, self.y_start_max))
        return {"start": np.array([x, y0]), "end": np.array([x, self.end_y]), "n_target": int(len(c)),
                "n_on_table": int(len(pts))}
