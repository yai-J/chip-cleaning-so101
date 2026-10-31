"""シーン構築：SO-101 + 作業台 + 回収トレー + ブラシ + 切粉 + カメラ を MjSpec で組み立てる。"""
import os
import numpy as np
import mujoco
import yaml

ROBOT_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
HOME_Q = np.array([0.0, -0.6, -1.2, 0.2, 0.0])  # 待機姿勢：アームを真上に立て、俯瞰カメラの視界を空ける


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).flatten())
    return q


def _quat2mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, dtype=float))
    return R.reshape(3, 3)


def _lookat_quat(pos, target, up=(0, 0, 1)):
    """カメラ（-z方向を向く）が target を見る姿勢のクォータニオン。"""
    pos, target, up = map(np.asarray, (pos, target, up))
    z = pos - target
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return _mat2quat(np.stack([x, y, z], axis=1))


def build_model(cfg, root="."):
    sc, br = cfg["scene"], cfg["brush"]
    spec = mujoco.MjSpec.from_file(os.path.join(root, cfg["paths"]["robot_xml"]))
    spec.option.timestep = cfg["sim"]["timestep"]
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    wb = spec.worldbody

    # 照明
    wb.add_light(name="key", pos=[0.3, 0.1, 1.2], dir=[-0.2, -0.1, -1.0], diffuse=[0.7, 0.7, 0.7],
                 specular=[0.3, 0.3, 0.3], castshadow=True)
    wb.add_light(name="fill", pos=[-0.4, -0.5, 0.8], dir=[0.5, 0.5, -0.8], diffuse=[0.3, 0.3, 0.3],
                 specular=[0, 0, 0], castshadow=False)

    # 床・作業台（工作機械のテーブルに見立てる）
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.05],
                pos=[0, 0, -0.30], rgba=[0.55, 0.57, 0.6, 1])
    edge = sc["table_edge_y"]
    ty0, ty1, tx0, tx1 = edge, 0.25, -0.12, 0.40
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[(tx1 - tx0) / 2, (ty1 - ty0) / 2, 0.02],
                pos=[(tx0 + tx1) / 2, (ty0 + ty1) / 2, -0.02], rgba=[0.16, 0.17, 0.19, 1],
                friction=[0.6, 0.005, 0.0001])
    # 台の脚（見た目用）
    wb.add_geom(name="table_leg", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.2, 0.12, 0.13],
                pos=[0.14, 0.08, -0.17], rgba=[0.3, 0.32, 0.35, 1], contype=0, conaffinity=0)

    # 右側の回収トレー（切粉シュート）
    bx0, bx1 = sc["bin"]["x"]
    by0, by1 = sc["bin"]["y"]
    dep = sc["bin"]["depth"]
    bin_rgba = [0.55, 0.45, 0.2, 1]
    wb.add_geom(name="bin_floor", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[(bx1 - bx0) / 2, (by1 - by0) / 2, 0.005],
                pos=[(bx0 + bx1) / 2, (by0 + by1) / 2, -dep - 0.005], rgba=bin_rgba)
    w = 0.003
    for nm, size, pos in [
        ("bin_wall_r", [(bx1 - bx0) / 2, w, dep / 2], [(bx0 + bx1) / 2, by0 - w, -dep / 2]),
        ("bin_wall_n", [w, (by1 - by0) / 2, dep / 2], [bx0 - w, (by0 + by1) / 2, -dep / 2]),
        ("bin_wall_f", [w, (by1 - by0) / 2, dep / 2], [bx1 + w, (by0 + by1) / 2, -dep / 2]),
    ]:
        wb.add_geom(name=nm, type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos, rgba=bin_rgba)

    # ブラシ：グリッパ先端サイトの延長上に取り付け
    grip = spec.body("gripper")
    gf = spec.site("gripperframe")
    R_gf = _quat2mat(gf.quat)
    # brush_tip サイト：z軸＝工具軸（gripperframe の x 軸）、x軸＝ブラシの幅方向
    M = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    R_bt = R_gf @ M
    tip_pos = np.asarray(gf.pos) + R_gf @ np.array([br["tip_offset"], 0, 0])
    q_bt = _mat2quat(R_bt)
    grip.add_site(name="brush_tip", pos=tip_pos, quat=q_bt, size=[0.003, 0, 0], group=4)
    # 毛（衝突あり）と柄（見た目のみ）。brush_tip から -z 側（工具の根元側）へ伸ばす
    hl, hw, ht = br["half_len"], br["half_width"], br["half_thick"]
    bristle_c = tip_pos + R_bt @ np.array([0, 0, -hl])  # 下面＝tip_pos になるよう配置
    grip.add_geom(name="brush_bristle", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hw, ht, hl],
                  pos=bristle_c, quat=q_bt, rgba=[0.85, 0.7, 0.4, 1], contype=1, conaffinity=1,
                  friction=[0.4, 0.005, 0.0001], mass=0.01, group=0)
    head_c = tip_pos + R_bt @ np.array([0, 0, -2 * hl - 0.006])
    grip.add_geom(name="brush_head", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hw + 0.003, ht + 0.004, 0.006],
                  pos=head_c, quat=q_bt, rgba=[0.35, 0.2, 0.1, 1], contype=0, conaffinity=0,
                  mass=0.01, group=0)

    # 切粉（自由剛体のプール。未使用分は画面外に退避）
    chip_mat = spec.add_material(name="chip_mat", rgba=[0.78, 0.79, 0.82, 1], specular=0.8, shininess=0.9)
    hs = sc["chip"]["half_size"]
    for i in range(sc["max_chips"]):
        b = wb.add_body(name=f"chip{i}", pos=[2.0 + 0.03 * i, 2.0, 0.01])
        b.add_freejoint(name=f"chip{i}_free")
        b.add_geom(name=f"chip{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=hs, mass=sc["chip"]["mass"],
                   material="chip_mat", friction=[sc["chip"]["friction"], 0.005, 0.0001],
                   condim=3, solref=[0.004, 1])

    # カメラ
    ct, cv = cfg["camera_top"], cfg["camera_view"]
    # 俯瞰：画像の右＝ロボットの右（-y）、画像の上＝前方（+x）
    R_top = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)
    wb.add_camera(name="top", pos=ct["pos"], quat=_mat2quat(R_top), fovy=ct["fovy"])
    wb.add_camera(name="view", pos=cv["pos"], quat=_lookat_quat(cv["pos"], cv["lookat"]), fovy=cv["fovy"])

    model = spec.compile()
    return model


class ChipScene:
    """モデル・データと、切粉配置やロボット姿勢の操作をまとめたクラス。"""

    def __init__(self, cfg, root="."):
        self.cfg = cfg
        self.model = build_model(cfg, root)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.n_chips_max = cfg["scene"]["max_chips"]
        self.chip_body = np.array([m.body(f"chip{i}").id for i in range(self.n_chips_max)])
        self.chip_geom = np.array([m.geom(f"chip{i}").id for i in range(self.n_chips_max)])
        self.chip_qadr = np.array([m.jnt_qposadr[m.joint(f"chip{i}_free").id] for i in range(self.n_chips_max)])
        self.chip_vadr = np.array([m.jnt_dofadr[m.joint(f"chip{i}_free").id] for i in range(self.n_chips_max)])
        self.arm_qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in ROBOT_JOINTS])
        self.arm_act = np.array([m.actuator(j).id for j in ROBOT_JOINTS])
        self.grip_act = m.actuator("gripper").id
        self.active = np.zeros(self.n_chips_max, bool)

    # ---- ロボット ----
    def set_arm(self, q):
        """関節角を直接設定（初期化用）し、指令値も同じ値にする。"""
        self.data.qpos[self.arm_qadr] = q
        self.data.ctrl[self.arm_act] = q
        self.data.ctrl[self.grip_act] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def arm_q(self):
        return self.data.qpos[self.arm_qadr].copy()

    # ---- 切粉 ----
    def scatter_chips(self, rng, n=None, region=None, clustered=True):
        """全ての切粉をリセットしてから n 個撒く（データ生成・単発テスト用）。"""
        hz = self.cfg["scene"]["chip"]["half_size"][2]
        self.active[:] = False
        for i in range(self.n_chips_max):
            a, va = self.chip_qadr[i], self.chip_vadr[i]
            self.data.qpos[a:a + 7] = [2.0 + 0.03 * (i % 30), 2.0 + 0.03 * (i // 30), hz, 1, 0, 0, 0]
            self.data.qvel[va:va + 6] = 0
        return self.add_chips(rng, n, region, clustered, drop=(0.0005, 0.006))

    def add_chips(self, rng, n=None, region=None, clustered=True, drop=(0.02, 0.07)):
        """未使用の切粉から n 個を、作業エリアの上方から落とす（加工で切粉が出る様子を模擬）。"""
        sc = self.cfg["scene"]
        if n is None:
            n = rng.integers(sc["chip"]["n_min"], sc["chip"]["n_max"] + 1)
        free = np.where(~self.active)[0][:n]
        n = len(free)
        reg = region or sc["workspace"]
        (x0, x1), (y0, y1) = reg["x"], reg["y"]
        pts = []
        if clustered:
            # 実際の切粉のように、いくつかの塊＋散らばりで配置
            n_clusters = rng.integers(2, 5)
            centers = np.c_[rng.uniform(x0 + 0.02, x1 - 0.02, n_clusters), rng.uniform(y0 + 0.02, y1 - 0.02, n_clusters)]
            for k in range(n):
                if rng.random() < 0.75:
                    p = centers[rng.integers(n_clusters)] + rng.normal(0, 0.015, 2)
                else:
                    p = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
                pts.append(np.clip(p, [x0, y0], [x1, y1]))
        else:
            pts = list(np.c_[rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)])
        hz = sc["chip"]["half_size"][2]
        for k, i in enumerate(free):
            a, va = self.chip_qadr[i], self.chip_vadr[i]
            yaw = rng.uniform(0, np.pi)
            z = hz + rng.uniform(*drop)
            self.data.qpos[a:a + 7] = [pts[k][0], pts[k][1], z, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            self.data.qvel[va:va + 6] = 0
            self.active[i] = True
        mujoco.mj_forward(self.model, self.data)
        return n

    def settle(self, seconds=0.5):
        for _ in range(int(seconds / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)

    def chip_positions(self):
        return self.data.xpos[self.chip_body[self.active]].copy()

    def collected_ratio(self):
        """真値：回収トレーに落ちた切粉の割合（評価用）。"""
        p = self.chip_positions()
        if len(p) == 0:
            return 1.0
        return float(np.mean(p[:, 2] < -0.01))
