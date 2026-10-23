#!/usr/bin/env python3
"""YOLO 推理

示例:
  # 摄像头（默认 /dev/video0）
  python -m ctts.detect

  # 指定摄像头
  python -m ctts.detect --cam 2

  # 视频文件
  python -m ctts.detect --video test.mp4

  # 指定权重 + 保存输出
  python -m ctts.detect --video test.mp4 -w weights/best.pt --save output.mp4

  # 调整阈值
  python -m ctts.detect --cam 0 --conf 0.3 --iou 0.6
"""

import argparse
import sys
import time

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QKeyEvent
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

from ultralytics import YOLO


# ---------------------------------------------------------------------------
# 默认参数
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = "weights/yolo26n.pt"
DEFAULT_IMGSZ = 640
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.7
DEFAULT_CAM = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO 推理——视频 / 摄像头 (PySide6)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = parser.add_mutually_exclusive_group()
    src.add_argument("--video", "-v", help="视频文件路径")
    src.add_argument("--cam", type=int, default=None, help="摄像头设备 ID（默认: 0）")

    parser.add_argument("-w", "--weights", default=DEFAULT_WEIGHTS, help=f"权重路径（默认: {DEFAULT_WEIGHTS}）")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help=f"推理分辨率（默认: {DEFAULT_IMGSZ}）")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help=f"置信度阈值（默认: {DEFAULT_CONF}）")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help=f"IOU 阈值（默认: {DEFAULT_IOU}）")
    parser.add_argument("--device", default="0", help="推理设备（默认: 0）")

    parser.add_argument("--save", "-s", help="保存输出视频到指定路径")
    parser.add_argument("--hide-labels", action="store_true", help="隐藏类别标签")
    parser.add_argument("--hide-conf", action="store_true", help="隐藏置信度，仅显示类别名")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# 绘制（纯 numpy/opencv，不涉及 GUI）
# ---------------------------------------------------------------------------

def draw_boxes(
    frame: np.ndarray,
    boxes: np.ndarray,
    class_names: dict[int, str],
    hide_labels: bool = False,
    hide_conf: bool = False,
) -> np.ndarray:
    """在帧上绘制检测框。boxes 格式: [x1, y1, x2, y2, conf, cls]"""
    colors = {}

    for box in boxes:
        x1, y1, x2, y2, conf, cls_id = box
        cls_id = int(cls_id)
        name = class_names.get(cls_id, f"cls_{cls_id}")

        if cls_id not in colors:
            rng = np.random.RandomState(cls_id)
            colors[cls_id] = tuple(int(c) for c in rng.randint(80, 255, 3))
        color = colors[cls_id]

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        if not hide_labels:
            label = name if hide_conf else f"{name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (int(x1), int(y1) - th - 4), (int(x1) + tw + 4, int(y1)), color, -1)
            cv2.putText(frame, label, (int(x1) + 2, int(y1) - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


# ---------------------------------------------------------------------------
# BGR ndarray → QPixmap
# ---------------------------------------------------------------------------

def frame_to_pixmap(frame: np.ndarray) -> QPixmap:
    """将 OpenCV BGR 图像转为 QPixmap。"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ---------------------------------------------------------------------------
# PySide6 主窗口
# ---------------------------------------------------------------------------

class DetectorWindow(QMainWindow):
    def __init__(self, args, model, cap, writer, source_name, fps_src):
        super().__init__()
        self.args = args
        self.model = model
        self.cap = cap
        self.writer = writer
        self.source_name = source_name
        self.fps_src = fps_src
        self.class_names = model.names

        # 统计
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_t0 = self.start_time
        self.fps_display = 0.0

        # 窗口
        self.setWindowTitle(f"YOLO Detection — {source_name}")
        self.setMinimumSize(320, 240)

        # 居中显示
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel("加载中...")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("background: #1a1a1a; color: #aaa;")
        layout.addWidget(self.label)

        # 定时器驱动帧循环
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(1)  # 尽可能快

        # 首帧加载后自适应窗口大小
        self._first_frame = True

    def tick(self):
        ret, frame = self.cap.read()
        if not ret:
            print("输入结束")
            self.close()
            return

        self.frame_count += 1

        # ---- 推理 ----
        results = self.model(frame, imgsz=self.args.imgsz, conf=self.args.conf,
                             iou=self.args.iou, device=self.args.device, verbose=False)

        if results[0].boxes is not None:
            boxes = results[0].boxes.data.cpu().numpy()
            frame = draw_boxes(frame, boxes, self.class_names,
                               self.args.hide_labels, self.args.hide_conf)

        # ---- FPS ----
        if self.frame_count % 10 == 0:
            t_now = time.time()
            elapsed = t_now - self.fps_t0
            if elapsed > 0:
                self.fps_display = 10 / elapsed
            self.fps_t0 = t_now

        # ---- 叠加信息 ----
        h, w = frame.shape[:2]
        cv2.putText(frame, f"FPS: {self.fps_display:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Q / ESC — 退出", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

        # ---- 保存 ----
        if self.writer:
            self.writer.write(frame)

        # ---- 显示 ----
        pixmap = frame_to_pixmap(frame)
        self.label.setPixmap(pixmap)

        if self._first_frame:
            self._first_frame = False
            self.resize(w, h)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key.Key_Q, Qt.Key.Key_Escape):
            self.close()

    def closeEvent(self, event):
        self.timer.stop()
        self.cap.release()
        if self.writer:
            self.writer.release()

        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        print(f"\n完成: {self.frame_count} 帧  |  耗时: {elapsed:.1f}s  |  平均 FPS: {avg_fps:.1f}")
        event.accept()


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- 加载模型 ----
    print(f"加载模型: {args.weights}")
    model = YOLO(args.weights)

    # ---- 打开输入源 ----
    if args.video:
        cap = cv2.VideoCapture(args.video)
        source_name = f"视频: {args.video}"
    elif args.cam is not None:
        cap = cv2.VideoCapture(args.cam)
        source_name = f"摄像头: /dev/video{args.cam}"
    else:
        cap = cv2.VideoCapture(DEFAULT_CAM)
        source_name = f"摄像头: /dev/video{DEFAULT_CAM}"

    if not cap.isOpened():
        raise RuntimeError(f"无法打开输入源: {source_name}")

    # 验证首帧可读
    ok, _ = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"输入源打开成功但无法读取帧: {source_name}")

    fps_src = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{source_name}  |  分辨率: {width}x{height}  |  FPS: {fps_src:.1f}")

    # ---- 输出视频写入器 ----
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps_src if fps_src > 0 else 30, (width, height))
        print(f"输出视频: {args.save}")

    # ---- PySide6 应用 ----
    app = QApplication(sys.argv)
    win = DetectorWindow(args, model, cap, writer, source_name, fps_src)
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
