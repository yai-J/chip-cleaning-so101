"""切粉の検出：YOLOセグメンテーション推論と、画素座標→台上座標の変換。

GTDetector はシミュレーションの真値マスクを返す検出器で、学習データ作成と
YOLO 学習前の動作確認に使う（清掃デモ本番では YoloDetector を使う）。
"""
import numpy as np
import cv2
import mujoco


def pixel_to_world(model, data, cam, uv, plane_z=0.0, width=640, height=480):
    """画素 (u, v) を通る視線と、高さ plane_z の水平面との交点を返す。"""
    cid = model.camera(cam).id
    fovy = np.deg2rad(model.cam_fovy[cid])
    f = 0.5 * height / np.tan(fovy / 2)
    uv = np.atleast_2d(np.asarray(uv, float))
    d_cam = np.stack([(uv[:, 0] - width / 2) / f, -(uv[:, 1] - height / 2) / f, -np.ones(len(uv))], axis=1)
    R = data.cam_xmat[cid].reshape(3, 3)
    p0 = data.cam_xpos[cid]
    d_w = d_cam @ R.T
    t = (plane_z - p0[2]) / d_w[:, 2]
    return p0 + t[:, None] * d_w


def _poly_center(poly):
    poly = np.asarray(poly, np.float32)
    M = cv2.moments(poly)
    if abs(M["m00"]) < 1e-6:
        return poly.mean(axis=0)
    return np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])


class GTDetector:
    """シミュレーションのセグメンテーション描画から切粉のインスタンスマスクを得る。"""

    def __init__(self, scene, renderer, cam="top", min_area=6):
        self.scene, self.r, self.cam, self.min_area = scene, renderer, cam, min_area

    def polygons(self):
        sc = self.scene
        self.r.enable_segmentation_rendering()
        self.r.update_scene(sc.data, camera=self.cam)
        seg = self.r.render()
        self.r.disable_segmentation_rendering()
        ids, types = seg[..., 0], seg[..., 1]
        out = []
        for gid in sc.chip_geom[sc.active]:
            mask = ((ids == gid) & (types == int(mujoco.mjtObj.mjOBJ_GEOM))).astype(np.uint8)
            if mask.sum() < self.min_area:
                continue
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = max(cnts, key=cv2.contourArea)
            if len(c) < 3:
                continue
            out.append(c.reshape(-1, 2).astype(np.float32))
        return out

    def detect(self, rgb=None):
        return [{"poly": p, "conf": 1.0, "center": _poly_center(p)} for p in self.polygons()]


class YoloDetector:
    """学習済み YOLO セグメンテーションモデルによる切粉検出。"""

    def __init__(self, weights, conf=0.4, imgsz=640, device=None):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf, self.imgsz, self.device = conf, imgsz, device

    def detect(self, rgb):
        res = self.model.predict(np.ascontiguousarray(rgb[..., ::-1]), conf=self.conf, imgsz=self.imgsz, verbose=False,
                                 device=self.device, retina_masks=True)[0]
        out = []
        if res.masks is None:
            return out
        confs = res.boxes.conf.cpu().numpy()
        for poly, c in zip(res.masks.xy, confs):
            if len(poly) < 3:
                continue
            out.append({"poly": poly.astype(np.float32), "conf": float(c), "center": _poly_center(poly)})
        return out


def detections_to_world(dets, model, data, cam, chip_z, width, height):
    if not dets:
        return np.zeros((0, 3))
    uv = np.array([d["center"] for d in dets])
    return pixel_to_world(model, data, cam, uv, plane_z=chip_z, width=width, height=height)


def draw_detections(rgb, dets, color=(255, 60, 60)):
    """マスクを半透明で塗り、輪郭を描いた画像を返す（RGB）。"""
    img = rgb.copy()
    overlay = img.copy()
    for d in dets:
        cv2.fillPoly(overlay, [d["poly"].astype(np.int32)], color)
    img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
    for d in dets:
        cv2.polylines(img, [d["poly"].astype(np.int32)], True, color, 1, cv2.LINE_AA)
    return img
