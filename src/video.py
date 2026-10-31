"""動画記録：斜め視点（左）と、検出マスクを重ねた俯瞰視点（右）を並べて等速で書き出す。"""
import numpy as np
import cv2
import imageio.v2 as imageio
import mujoco

from detector import draw_detections


class Recorder:
    def __init__(self, model, path, cfg, scale=1.0):
        self.model = model
        cv, ct = cfg["camera_view"], cfg["camera_top"]
        self.vw, self.vh = int(cv["width"] * scale), int(cv["height"] * scale)
        self.tw, self.th = int(ct["width"] * scale), int(ct["height"] * scale)
        self.r_view = mujoco.Renderer(model, self.vh, self.vw)
        self.r_top = mujoco.Renderer(model, self.th, self.tw)
        self.fps = cfg["motion"]["fps"]
        self.scale = scale
        self.writer = imageio.get_writer(path, fps=self.fps, codec="libx264", quality=8,
                                         macro_block_size=1)
        self.n_frames = 0
        self.dets = []
        self.show_dets = False
        self.status = {}

    def _bar(self, img, lines, h=None):
        h = h or int(26 * self.scale) * len(lines) + 6
        cv2.rectangle(img, (0, 0), (img.shape[1], h), (0, 0, 0), -1)
        for i, t in enumerate(lines):
            cv2.putText(img, t, (8, int((22 + 26 * i) * self.scale)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6 * self.scale, (255, 255, 255), max(1, int(round(1.5 * self.scale))), cv2.LINE_AA)
        return img

    def frame(self, data, repeat=1):
        self.r_view.update_scene(data, camera="view")
        left = self.r_view.render().copy()
        self.r_top.update_scene(data, camera="top")
        right = self.r_top.render().copy()
        if self.show_dets and self.dets:
            dets = self.dets
            if self.scale != 1.0:
                dets = [{**d, "poly": d["poly"] * self.scale} for d in self.dets]
            right = draw_detections(right, dets)
        s = self.status
        self._bar(left, [f"SO-101 chip cleaning  |  {s.get('phase', '')}",
                         f"Cycle {s.get('cycle', 1)}   Stroke {s.get('stroke', 0)} (max {s.get('max_strokes', 0)})"])
        self._bar(right, [f"Top camera + YOLO-seg  |  detected on table: {s.get('n_table', '-')}",
                          f"Collected to bin (sim GT): {s.get('collected', 0):.0f}%"])
        img = np.concatenate([left, right], axis=1)
        for _ in range(repeat):
            self.writer.append_data(img)
            self.n_frames += 1
        return img

    def seconds(self):
        return self.n_frames / self.fps

    def close(self):
        self.writer.close()
