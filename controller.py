"""清掃ループ：検出 → ストローク計画 → 掃く → 再検出 を自律的に繰り返す。"""
import time
import numpy as np
import mujoco

from scene import HOME_Q
from ik import brush_down_R
from detector import detections_to_world


class CleaningController:
    CTRL_HZ = 100

    def __init__(self, scene, ik, planner, detector, cfg, recorder=None, log=print):
        self.sc, self.ik, self.planner, self.det = scene, ik, planner, detector
        self.cfg, self.rec, self.log = cfg, recorder, log
        self.m, self.d = scene.model, scene.data
        mo = cfg["motion"]
        self.z_sweep = cfg["brush"]["sweep_height"]
        self.z_up = mo["approach_height"]
        self.v_sweep = mo["sweep_speed"]
        self.t_move = mo["move_time"]
        self.t_settle = mo["settle_time"]
        self.fps = mo["fps"]
        self.R = brush_down_R(0.0)  # ブラシの幅方向を x 軸（掃く方向 -y に直交）に合わせる
        ct = cfg["camera_top"]
        self.cw, self.ch = ct["width"], ct["height"]
        self.r_det = mujoco.Renderer(self.m, self.ch, self.cw)
        self.chip_top_z = 2 * cfg["scene"]["chip"]["half_size"][2]
        self.q_cmd = scene.arm_q()
        self._sim_t_frame = 0.0
        self.records = []

    # ---------- 低レベル ----------
    def _advance(self, seconds):
        """シミュレーションを進め、fps ごとに動画フレームを記録する。"""
        dt = self.m.opt.timestep
        for _ in range(max(1, int(round(seconds / dt)))):
            mujoco.mj_step(self.m, self.d)
            self._sim_t_frame += dt
            if self._sim_t_frame >= 1.0 / self.fps - 1e-9:
                self._sim_t_frame -= 1.0 / self.fps
                if self.rec is not None:
                    self.rec.status["collected"] = 100 * self.sc.collected_ratio()
                    self.rec.frame(self.d)

    def _set_cmd(self, q):
        self.q_cmd = np.asarray(q, float)
        self.d.ctrl[self.sc.arm_act] = self.q_cmd
        self.d.ctrl[self.sc.grip_act] = 0.0

    def move_joint(self, q_goal, T):
        """関節空間でなめらか（余弦補間）に移動する。"""
        q0 = self.q_cmd.copy()
        n = max(1, int(T * self.CTRL_HZ))
        for k in range(1, n + 1):
            s = 0.5 - 0.5 * np.cos(np.pi * k / n)
            self._set_cmd(q0 + s * (q_goal - q0))
            self._advance(1.0 / self.CTRL_HZ)

    def move_line(self, p0, p1, T):
        """手先（ブラシ下面）を直線に沿って動かす。各制御周期で IK を解く。"""
        n = max(1, int(T * self.CTRL_HZ))
        q = self.q_cmd.copy()
        worst = 0.0
        for k in range(1, n + 1):
            s = 0.5 - 0.5 * np.cos(np.pi * k / n)
            p = p0 + s * (p1 - p0)
            q, err, _ = self.ik.solve(p, self.R, q, iters=30)
            worst = max(worst, err)
            self._set_cmd(q)
            self._advance(1.0 / self.CTRL_HZ)
        return worst

    def ik_point(self, p, q_seed=None):
        q, err, ok = self.ik.solve(p, self.R, self.q_cmd if q_seed is None else q_seed, iters=300)
        return q, ok

    # ---------- 検出 ----------
    def detect(self):
        self.r_det.update_scene(self.d, camera="top")
        rgb = self.r_det.render()
        dets = self.det.detect(rgb)
        pts = detections_to_world(dets, self.m, self.d, "top", self.chip_top_z, self.cw, self.ch)
        return dets, pts

    # ---------- 清掃ループ ----------
    def go_home(self):
        self.move_joint(HOME_Q, self.t_move * 1.3)

    def new_chips(self, rng, n=None, label=""):
        """加工完了を模擬：作業エリアに新しい切粉を落とす。"""
        if self.rec:
            self.rec.show_dets = False
            self.rec.status.update(phase=f"MACHINING DONE -> chips {label}", stroke=0, n_table="-")
        n = self.sc.add_chips(rng, n)
        self._advance(1.5)
        return n

    def run_cycles(self, n_cycles, rng, max_strokes=None):
        """「加工→清掃」を n_cycles 回繰り返す（動画を1〜3分にする用途も兼ねる）。"""
        self._set_cmd(self.sc.arm_q())
        results = []
        for c in range(1, n_cycles + 1):
            if self.rec:
                self.rec.status.update(cycle=c, n_cycles=n_cycles)
            self.log(f"===== 加工サイクル {c}/{n_cycles} =====")
            n = self.new_chips(rng)
            res = self.run(max_strokes=max_strokes)
            res.update(cycle=c, n_new_chips=n)
            results.append(res)
        return results

    def run(self, max_strokes=None, hold_detect=1.2):
        max_strokes = max_strokes or self.cfg["planner"]["max_strokes"]
        rec = self.rec
        self.records = []
        self.planner.history = []
        if rec:
            rec.status.update(max_strokes=max_strokes, stroke=0, phase="start")
        self._set_cmd(self.sc.arm_q())
        self._advance(0.3)
        t_wall = time.time()
        for k in range(1, max_strokes + 1):
            # 1) 検出
            dets, pts = self.detect()
            on_table = self.planner.on_table(pts)
            plan = self.planner.plan(pts)
            info = {"stroke": k, "n_det": len(dets), "n_on_table": len(on_table),
                    "collected_gt": self.sc.collected_ratio()}
            if rec:
                rec.dets, rec.show_dets = dets, True
                rec.status.update(phase="DETECT (YOLO-seg)", n_table=len(on_table), stroke=k - 1)
                rec.frame(self.d, repeat=int(hold_detect * self.fps))
            if plan is None:
                self.log(f"[stroke {k}] 台上に切粉なし → 清掃完了")
                info["done"] = True
                self.records.append(info)
                break
            self.log(f"[stroke {k}] 検出 {len(dets)} 個（台上 {len(on_table)}）, 対象の塊 {plan['n_target']} 個, "
                     f"x={plan['start'][0]:.3f} y:{plan['start'][1]:.3f}→{plan['end'][1]:.3f}")
            if rec:
                rec.show_dets = False
                rec.status.update(phase="SWEEP", stroke=k)
            # 2) 実行：接近 → 降下 → 掃引 → 上昇 → 待機姿勢
            s, e = plan["start"], plan["end"]
            p_above = np.array([s[0], s[1], self.z_up])
            p_down = np.array([s[0], s[1], self.z_sweep])
            p_end = np.array([e[0], e[1], self.z_sweep])
            p_lift = np.array([e[0], e[1], self.z_up])
            q_low, ok1 = self.ik_point(p_down, q_seed=np.array([0, -0.5, 0.5, 1.2, 0]))
            q_above, ok2 = self.ik_point(p_above, q_seed=q_low)
            if not (ok1 and ok2):
                self.log("  到達不能のためスキップ")
                info["skipped"] = True
                self.records.append(info)
                continue
            self.move_joint(q_above, self.t_move)
            self.move_line(p_above, p_down, 0.6)
            dist = np.linalg.norm(e - s)
            err = self.move_line(p_down, p_end, max(1.0, dist / self.v_sweep))
            self.move_line(p_end, p_lift, 0.6)
            self.go_home()
            self._advance(self.t_settle)
            info.update(ik_err_max=err, plan=plan)
            self.records.append(info)
        # 最終確認
        dets, pts = self.detect()
        on_table = self.planner.on_table(pts)
        result = {"strokes": len([r for r in self.records if "plan" in r]),
                  "remaining_on_table_detected": int(len(on_table)),
                  "collected_ratio_gt": self.sc.collected_ratio(),
                  "wall_time_s": time.time() - t_wall}
        if rec:
            rec.dets, rec.show_dets = dets, True
            rec.status.update(phase="CLEANING FINISHED", n_table=len(on_table))
            rec.frame(self.d, repeat=int(2.0 * self.fps))
        self.log(f"結果: ストローク {result['strokes']} 回, 台上の残り（検出）{result['remaining_on_table_detected']} 個, "
                 f"回収率（真値）{100 * result['collected_ratio_gt']:.0f}%")
        return result
