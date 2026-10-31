"""SO-101 の MuJoCo モデル（TheRobotStudio/SO-ARM100, Apache-2.0）を取得する。"""
import os
import re
import urllib.request

BASE = "https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/main/Simulation/SO101"


def download(dst="assets/so101"):
    os.makedirs(f"{dst}/assets", exist_ok=True)
    xml_path = f"{dst}/so101_new_calib.xml"
    if not os.path.exists(xml_path):
        urllib.request.urlretrieve(f"{BASE}/so101_new_calib.xml", xml_path)
    meshes = sorted(set(re.findall(r'file="([^"]+\.stl)"', open(xml_path, encoding="utf-8").read())))
    for m in meshes:
        p = f"{dst}/assets/{m}"
        if not os.path.exists(p):
            urllib.request.urlretrieve(f"{BASE}/assets/{m}", p)
    print(f"SO-101 model ready: {xml_path} ({len(meshes)} meshes)")
    return xml_path


if __name__ == "__main__":
    download()
