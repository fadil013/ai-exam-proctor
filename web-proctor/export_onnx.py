"""
Run once: python export_onnx.py
Exports rtdetr-x.pt → public/rtdetr-x.onnx for the browser.
"""
from ultralytics import RTDETR
import shutil, pathlib

model = RTDETR("../rtdetr-x.pt")
model.export(format="onnx", imgsz=640, simplify=True, opset=17)

src = pathlib.Path("../rtdetr-x.onnx")
dst = pathlib.Path("public/rtdetr-x.onnx")
dst.parent.mkdir(exist_ok=True)
shutil.move(str(src), str(dst))
print(f"Done → {dst.resolve()}")
