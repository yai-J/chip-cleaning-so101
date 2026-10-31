"""逆運動学：MuJoCo のヤコビアンを用いた減衰最小二乗法（DLS）。

brush_tip サイトの位置（3）と姿勢（ブラシ軸を鉛直下向き＋幅方向の向き）を目標にする。
外部IKライブラリを使わないので、Colab / Windows / Linux のどこでも同じ結果になる。
"""
import numpy as np
import mujoco

from scene import ROBOT_JOINTS


def brush_down_R(yaw=0.0):
    """ブラシ軸を鉛直下向き、幅方向を world の x 軸から yaw だけ回した姿勢。"""
    x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


class BrushIK:
    def __init__(self, model, site="brush_tip"):
        self.m = model
        self.d = mujoco.MjData(model)
        self.sid = model.site(site).id
        jids = [model.joint(j).id for j in ROBOT_JOINTS]
        self.qadr = np.array([model.jnt_qposadr[j] for j in jids])
        self.dadr = np.array([model.jnt_dofadr[j] for j in jids])
        self.lo = model.jnt_range[jids, 0]
        self.hi = model.jnt_range[jids, 1]
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))

    def fk(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.sid].copy(), self.d.site_xmat[self.sid].reshape(3, 3).copy()

    def solve(self, target_pos, target_R, q0, iters=150, damping=1e-4, rot_w=0.3, tol=5e-4):
        """戻り値: (q, 位置誤差[m], 成功フラグ)"""
        q = np.array(q0, dtype=float)
        target_pos = np.asarray(target_pos, float)
        for _ in range(iters):
            p, R = self.fk(q)
            e_p = target_pos - p
            e_r = 0.5 * sum(np.cross(R[:, i], target_R[:, i]) for i in range(3))
            if np.linalg.norm(e_p) < tol and np.linalg.norm(e_r) < 1e-2:
                break
            mujoco.mj_comPos(self.m, self.d)
            mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.sid)
            J = np.vstack([self.jacp[:, self.dadr], rot_w * self.jacr[:, self.dadr]])
            e = np.concatenate([e_p, rot_w * e_r])
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), e)
            n = np.linalg.norm(dq)
            if n > 0.2:  # 1回の更新量を制限して安定化
                dq *= 0.2 / n
            q = np.clip(q + dq, self.lo, self.hi)
        p, R = self.fk(q)
        err = float(np.linalg.norm(target_pos - p))
        ang = float(np.linalg.norm(0.5 * sum(np.cross(R[:, i], target_R[:, i]) for i in range(3))))
        return q, err, (err < 3e-3 and ang < 0.05)
