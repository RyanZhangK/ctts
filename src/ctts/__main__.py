#!/usr/bin/env python3
"""CTTS — CNN Train Tools Set 统一入口。

整合视觉 CNN 训练全流程工具：数据准备 → 标注管理 → 训练 → 推理。

CLI 示例:
  python ctts.py data build video.mp4 output/ -i 0.5
  python ctts.py data filter images/ --move
  python ctts.py data split dataset/ -v 0.2
  python ctts.py label project list
  python ctts.py label data stats -p 10
  # 远程 Label Studio（参数跟在子模块后）
  python ctts.py label project -- --base-url http://192.168.1.10:8080 --token <TOKEN> list
  python ctts.py train
  python ctts.py prelabel assess -p 10
  python ctts.py prelabel run -p 10
  python ctts.py detect --cam 0

GUI 模式:
  python ctts.py --gui
"""

import argparse
import json
import os
import sys
import threading
import webbrowser
from pathlib import Path

# 仓根：src/ctts/__main__.py → 上三级
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_REVIEW_MODULE = None


def _load_review_module():
    """在 GUI 主线程懒加载 review，避免在 QThread 内首次导入 SDK。"""
    global _REVIEW_MODULE
    if _REVIEW_MODULE is None:
        from . import review as module
        _REVIEW_MODULE = module
    return _REVIEW_MODULE


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cli_main():
    parser = argparse.ArgumentParser(
        description="CTTS — CNN Train Tools Set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gui", action="store_true", help="启动图形界面")
    sub = parser.add_subparsers(dest="module", help="功能模块")

    # ---- data ----
    p_data = sub.add_parser("data", help="数据准备（视频抽帧 / 数据集划分）")
    p_data_sub = p_data.add_subparsers(dest="action", help="子命令")

    p_data_sub.add_parser("build", help="视频抽帧 + 质量过滤").add_argument(
        "args", nargs=argparse.REMAINDER, help="dataset_build 参数")
    p_data_sub.add_parser("split", help="数据集划分").add_argument(
        "args", nargs=argparse.REMAINDER, help="split_dataset 参数")

    # ---- label ----
    p_label = sub.add_parser("label", help="标注管理（Label Studio）")
    p_label_sub = p_label.add_subparsers(dest="action", help="子命令")

    p_label_sub.add_parser("project", help="项目管理").add_argument(
        "ls_args", nargs=argparse.REMAINDER, help="传递给 label_studio_project 的参数")
    p_label_sub.add_parser("data", help="数据管理").add_argument(
        "ls_args", nargs=argparse.REMAINDER, help="传递给 label_studio_data 的参数")
    p_label_sub.add_parser("auth", help="认证测试").add_argument(
        "ls_args", nargs=argparse.REMAINDER)

    # ---- train ----
    p_train = sub.add_parser("train", help="YOLO 训练（可透传 yolo_train.py 的高级参数）")
    p_train.add_argument("args", nargs=argparse.REMAINDER,
                         help="传递给 yolo_train 的参数，如 --set lr0=0.001")

    # ---- detect ----
    p_detect = sub.add_parser("detect", help="YOLO 推理")
    p_detect.add_argument("args", nargs=argparse.REMAINDER, help="传递给 detect 的参数")

    # ---- prelabel ----
    p_prelabel = sub.add_parser("prelabel", help="小模型预标注流水线（评估→建集→训练→预标注）")
    p_prelabel.add_argument("args", nargs=argparse.REMAINDER,
                            help="传递给 prelabel 的参数")

    args, _unknown = parser.parse_known_args()
    raw = sys.argv[1:]

    if args.gui:
        gui_main()
        return

    if not raw:
        parser.print_help()
        return

    # 手动分发：原样保留顺序转发给子模块，避免 argparse 打乱
    # `--set lr0=0.001` 这类透传参数。
    module, rest = raw[0], raw[1:]
    if rest and rest[0] == "--":
        rest = rest[1:]

    if module == "data":
        action = rest[0] if rest else ""
        _forward_to_module({
            "build": "dataset_build",
            "split": "split_dataset",
        }, action, parser, extra_args=rest[1:])

    elif module == "label":
        action = rest[0] if rest else ""
        _forward_to_module({
            "project": "label_studio_project",
            "data": "label_studio_data",
            "auth": "label_studio_auth",
        }, action, parser, extra_args=rest[1:])

    elif module == "train":
        if rest:
            _forward_module("yolo_train", rest)
        else:
            print("提示: 未传参数时使用 yolo_train.py 的默认配置；")
            print("      自定义参数示例: python ctts.py train --set lr0=0.001")
            _forward_yolo_train()

    elif module == "detect":
        _forward_detect(rest)

    elif module == "prelabel":
        _forward_module("prelabel", rest)

    else:
        parser.print_help()


def _forward_to_module(mapping: dict, action: str, parser, extra_args: list | None = None):
    """将 CLI 请求转发到对应的工具模块。"""
    if action not in mapping:
        parser.print_help()
        return
    _forward_module(mapping[action], extra_args)


def _forward_module(mod_name: str, extra_args: list | None = None):
    """重建 sys.argv 并调用工具模块的 main()。"""
    # 重建 sys.argv: ctts.py data build ... → dataset_build.py ...
    sys.argv = [mod_name]
    if extra_args:
        # 去掉 REMAINDER 捕获的 "--" 分隔符
        sys.argv += [a for a in extra_args if a != "--"]

    import importlib
    mod = importlib.import_module(f".{mod_name}", package=__package__)
    mod.main()


def _forward_yolo_train():
    """启动 YOLO 训练。"""
    try:
        from .yolo_train import main as m
        m()
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("提示: 训练需要 ultralytics 库，请确认在正确的 conda 环境中运行")
        sys.exit(1)


def _forward_detect(extra_args: list):
    """启动 YOLO 推理（detect.py 自带 PySide6 GUI）。"""
    sys.argv = ["-m", "ctts.detect"] + extra_args
    from .detect import main as m
    m()


# ---------------------------------------------------------------------------
# PySide6 GUI
# ---------------------------------------------------------------------------

def gui_main():
    """启动 CTTS 图形界面。"""

    try:
        from PySide6.QtCore import Qt, QThread, QTimer, Signal, QRectF, QPointF
        from PySide6.QtGui import QFont, QPixmap, QPainter, QColor, QPen
        from PySide6.QtWidgets import (
            QApplication, QWidget, QVBoxLayout, QHBoxLayout,
            QTabWidget, QPushButton, QLabel, QLineEdit,
            QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
            QGroupBox, QFormLayout, QSpinBox, QProgressBar,
            QMainWindow, QFileDialog, QTextEdit, QComboBox,
            QCheckBox, QDoubleSpinBox, QSlider, QScrollArea,
            QPlainTextEdit, QDialog, QDialogButtonBox, QSplitter,
        )
    except ModuleNotFoundError:
        venv_dir = os.path.expanduser(os.environ.get("CTTS_VENV", "~/.cache/ctts-venv"))
        if os.name == "nt":
            venv_py = os.path.join(venv_dir, "Scripts", "python.exe")
            activate = os.path.join(venv_dir, "Scripts", "Activate.ps1")
        else:
            venv_py = os.path.join(venv_dir, "bin", "python")
            activate = os.path.join(venv_dir, "bin", "activate")
        setup_py = os.path.join(_PROJECT_ROOT, "scripts", "setup_env.py")
        print("❌ 当前解释器没有 PySide6。")
        print(f"   当前: {sys.executable}")
        print("首次使用请先初始化环境：")
        print(f'   "{sys.executable}" "{setup_py}"')
        print("初始化完成后启动 GUI：")
        print(f'   "{venv_py}" -m ctts --gui')
        if os.path.isfile(venv_py):
            print("或先激活：")
            print(f'   {activate}')
            print("   python -m ctts --gui")
        sys.exit(1)

    try:
        from .label_studio_auth import (
            DEFAULT_BASE_URL,
            get_client,
            reset_clients,
            resolve_base_url,
        )
        from .label_studio_project import list_projects, create_project, delete_project
        from .label_studio_data import list_tasks, delete_tasks_batch, task_stats, import_tasks_from_images
        _ls_import_error = None
    except Exception as e:
        _ls_import_error = e
        DEFAULT_BASE_URL = "http://127.0.0.1:8080"

        def resolve_base_url(base_url=None):
            return base_url or DEFAULT_BASE_URL

        def get_client(*args, **kwargs):
            raise RuntimeError(f"Label Studio 模块不可用: {_ls_import_error}")

        def reset_clients():
            pass

        list_projects = create_project = delete_project = None
        list_tasks = delete_tasks_batch = task_stats = import_tasks_from_images = None
        print(f"⚠ Label Studio 模块加载失败（标注页将不可用）: {e}")

    PRIMARY_BTN_QSS = "font-size: 14px; padding: 8px;"
    MUTED_QSS = "color: gray;"
    STATUS_COLORS = {"ok": "green", "warn": "orange", "err": "red", "idle": "gray"}

    def _set_status(label: QLabel, text: str, level: str = "idle") -> None:
        label.setText(text)
        label.setStyleSheet(f"color: {STATUS_COLORS.get(level, 'gray')};")

    # ---- PrelabelWorker ----
    class PrelabelWorker(QThread):
        """预标注流水线线程：日志 + 阶段事件 + 结果。"""

        log_line = Signal(str)
        event = Signal(object)
        finished = Signal(object)
        error = Signal(str)

        def __init__(self, fn):
            super().__init__()
            self.fn = fn

        def run(self):
            try:
                self.finished.emit(self.fn(log=self._log, on_event=self._event))
            except Exception as e:
                self.error.emit(str(e))

        def _log(self, msg: str):
            self.log_line.emit(msg)

        def _event(self, event):
            self.event.emit(event)

    # ---- StreamWorker ----
    class StreamWorker(QThread):
        log_line = Signal(str)
        progress = Signal(int, int)
        finished = Signal(object)
        error = Signal(str)

        def __init__(self, fn):
            super().__init__()
            self.fn = fn

        def run(self):
            try:
                result = self.fn(log=self._log)
                self.finished.emit(result)
            except Exception as e:
                self.error.emit(str(e))

        def _log(self, msg: str):
            self.log_line.emit(msg)



    # ---- ReviewCanvas / ReviewImageWorker（⑦ 复核页图片预览） ----
    class ReviewCanvas(QWidget):
        """绘制任务图片和预测框；点击框可切换是否认可。"""

        boxClicked = Signal(int)

        def __init__(self):
            super().__init__()
            self.setMinimumSize(420, 320)
            self._pix = None
            self._boxes = []
            self._selected: set[int] = set()
            self._msg = "选择任务查看预测框预览"
            self._task_id = None

        def set_message(self, msg: str):
            self._pix = None
            self._boxes = []
            self._selected = set()
            self._msg = msg
            self.update()

        def set_frame(self, pixmap, boxes, selected, task_id):
            self._pix = pixmap
            self._boxes = list(boxes or [])
            self._selected = set(selected)
            self._task_id = task_id
            self.update()

        def set_selected(self, selected):
            self._selected = set(selected)
            self.update()

        def _layout(self):
            pw = self._pix.width() if self._pix is not None else 1
            ph = self._pix.height() if self._pix is not None else 1
            vw, vh = self.width(), self.height()
            scale = min(vw / max(pw, 1), vh / max(ph, 1))
            ow, oh = int(pw * scale), int(ph * scale)
            return (vw - ow) // 2, (vh - oh) // 2, ow, oh, scale

        def paintEvent(self, _event):
            p = QPainter(self)
            p.fillRect(self.rect(), QColor("#f5f5f5"))
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            if self._pix is None or self._pix.isNull():
                p.setPen(QColor("#888"))
                p.drawText(self.rect(), Qt.AlignCenter, self._msg)
                return
            ox, oy, ow, oh, _ = self._layout()
            target = QRectF(ox, oy, ow, oh)
            p.drawPixmap(target, self._pix, QRectF(self._pix.rect()))
            p.setFont(QFont("Microsoft YaHei UI", 8))
            for i, res in enumerate(self._boxes):
                val = res.get("value", {}) if isinstance(res, dict) else {}
                try:
                    x = float(val.get("x", 0) or 0)
                    y = float(val.get("y", 0) or 0)
                    bw = float(val.get("width", 0) or 0)
                    bh = float(val.get("height", 0) or 0)
                except (TypeError, ValueError):
                    continue
                rect = QRectF(ox + x / 100 * ow, oy + y / 100 * oh,
                              bw / 100 * ow, bh / 100 * oh)
                sel = i in self._selected
                pen = QPen(QColor("#16a34a") if sel else QColor("#ef4444"),
                           2.2 if sel else 1.2)
                if not sel:
                    pen.setStyle(Qt.DashLine)
                p.setPen(pen)
                p.drawRect(rect)
                labels = val.get("rectanglelabels") or []
                text = f"{i + 1}. {'/'.join(str(v) for v in labels) or '?'}"
                score = res.get("score") if isinstance(res, dict) else None
                if score is not None:
                    try:
                        text += f" {float(score):.2f}"
                    except (TypeError, ValueError):
                        pass
                fm = p.fontMetrics()
                tw = fm.horizontalAdvance(text) + 6
                th = fm.height() + 2
                ty = max(oy, rect.top() - th)
                p.fillRect(QRectF(rect.left(), ty, tw, th),
                           QColor(22, 163, 74, 200) if sel else QColor(120, 120, 120, 170))
                p.setPen(QColor("white"))
                p.drawText(QRectF(rect.left() + 3, ty, tw, th),
                           Qt.AlignVCenter | Qt.AlignLeft, text)

        def mousePressEvent(self, event):
            if self._pix is None or self._pix.isNull():
                return
            ox, oy, ow, oh, _ = self._layout()
            pos = event.position()
            for i in range(len(self._boxes) - 1, -1, -1):
                res = self._boxes[i]
                val = res.get("value", {}) if isinstance(res, dict) else {}
                try:
                    x = float(val.get("x", 0) or 0)
                    y = float(val.get("y", 0) or 0)
                    bw = float(val.get("width", 0) or 0)
                    bh = float(val.get("height", 0) or 0)
                except (TypeError, ValueError):
                    continue
                rect = QRectF(ox + x / 100 * ow, oy + y / 100 * oh,
                              bw / 100 * ow, bh / 100 * oh)
                if rect.contains(pos):
                    self.boxClicked.emit(i)
                    return

    class ReviewImageWorker(QThread):
        loaded = Signal(int, bytes)
        failed = Signal(int, str)

        def __init__(self, task_id, client, review_mod):
            super().__init__()
            self.task_id = task_id
            self.client = client
            self.review_mod = review_mod

        def run(self):
            try:
                url = self.review_mod.task_image_url(self.client, self.task_id)
                data = self.review_mod.fetch_image_bytes(self.client, url)
                self.loaded.emit(self.task_id, data)
            except Exception as e:
                self.failed.emit(self.task_id, str(e))

    # ---- 主窗口 ----
    class CTTSWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("CTTS — CNN Train Tools Set")
            self.resize(960, 700)
            self.client = None
            self._workers = []
            self._build_ui()

        def _build_ui(self):
            central = QWidget()
            self.setCentralWidget(central)
            layout = QVBoxLayout(central)

            # 共享进度条（必须在 Tab 构建之前创建，因为 _build_label_tab
            # 会立即触发 _run_task → _show_progress，需要 self._progress 已存在）
            self._progress = QProgressBar()
            self._progress.setRange(0, 0)  # indeterminate
            self._progress.setVisible(False)
            self._progress.setMaximumHeight(18)
            layout.addWidget(self._progress)

            tabs = QTabWidget()
            self._tabs = tabs
            tabs.addTab(self._build_data_tab(), "① 数据准备")
            tabs.addTab(self._build_label_tab(), "② 标注管理")
            tabs.addTab(self._build_prelabel_tab(), "③ 预标注")
            tabs.addTab(self._build_split_tab(), "④ 数据集划分")
            tabs.addTab(self._build_train_tab(), "⑤ 训练")
            tabs.addTab(self._build_detect_tab(), "⑥ 推理")
            tabs.addTab(self._build_review_tab(), "⑦ 复核")
            layout.addWidget(tabs)

            # 共享日志输出
            self._log_output = QTextEdit()
            self._log_output.setReadOnly(True)
            self._log_output.setMaximumHeight(150)
            self._log_output.setPlaceholderText("操作日志将显示在这里...")
            layout.addWidget(self._log_output)

            # 项目 ID 与 ② 标注管理联动；初始化连接状态
            self._ls_task_project.valueChanged.connect(self._pre_project.setValue)
            self._pre_project.valueChanged.connect(self._ls_task_project.setValue)
            self._pre_project.valueChanged.connect(self._pre_on_project_changed)
            self._pre_refresh_models()
            self._pre_update_connection()
            self._tabs.currentChanged.connect(self._on_tab_changed)

        def _on_tab_changed(self, index: int):
            if index == 2:
                self._pre_update_connection()
            elif index == 6:
                self._rv_auto_load()

        # ============================================================
        # ① 数据准备 Tab
        # ============================================================
        def _build_data_tab(self) -> QWidget:
            w = QWidget()
            layout = QVBoxLayout(w)

            grp_build = QGroupBox("视频抽帧 + 质量过滤")
            fl = QFormLayout(grp_build)
            self._build_video = QLineEdit()
            self._build_video.setPlaceholderText("选择视频文件（可多选）...")
            row = QHBoxLayout()
            row.addWidget(self._build_video)
            btn = QPushButton("浏览")
            btn.clicked.connect(lambda: self._browse_files(self._build_video, "视频 (*.mp4 *.avi *.mov *.mkv)"))
            row.addWidget(btn)
            fl.addRow("视频:", row)

            self._build_output = QLineEdit("output_frames")
            fl.addRow("输出目录:", self._build_output)

            self._build_interval = QDoubleSpinBox()
            self._build_interval.setValue(1.0)
            self._build_interval.setSuffix(" 秒")
            fl.addRow("间隔:", self._build_interval)

            # 质量过滤参数
            self._build_lap = QDoubleSpinBox()
            self._build_lap.setRange(0, 1000)
            self._build_lap.setValue(50)
            fl.addRow("清晰度下限:", self._build_lap)

            self._build_entropy = QDoubleSpinBox()
            self._build_entropy.setRange(0, 10)
            self._build_entropy.setValue(3.0)
            self._build_entropy.setSingleStep(0.5)
            fl.addRow("对比度下限:", self._build_entropy)

            self._build_min_brt = QDoubleSpinBox()
            self._build_min_brt.setRange(0, 255)
            self._build_min_brt.setValue(10)
            fl.addRow("亮度下限:", self._build_min_brt)

            self._build_max_brt = QDoubleSpinBox()
            self._build_max_brt.setRange(0, 255)
            self._build_max_brt.setValue(245)
            fl.addRow("亮度上限:", self._build_max_brt)

            btn_build = QPushButton("开始抽帧")
            btn_build.setStyleSheet("font-size: 14px; padding: 8px;")
            btn_build.clicked.connect(self._run_build)
            fl.addRow(btn_build)
            layout.addWidget(grp_build)
            layout.addStretch()
            return w

        def _run_build(self):
            video_text = self._build_video.text().strip()
            output = self._build_output.text().strip()
            interval = self._build_interval.value()
            lap = self._build_lap.value()
            ent = self._build_entropy.value()
            min_b = self._build_min_brt.value()
            max_b = self._build_max_brt.value()
            if not video_text:
                QMessageBox.warning(self, "提示", "请选择视频文件")
                return
            videos = [v.strip() for v in video_text.split(";") if v.strip()]
            self._start_task(
                lambda log: self._do_build(log, videos, output, interval, lap, ent, min_b, max_b),
                "抽帧中",
            )

        def _do_build(self, log, videos, output, interval, lap, ent, min_b, max_b):
            from .dataset_build import collect_videos, extract_frames
            all_videos = collect_videos(videos)
            log(f"找到 {len(all_videos)} 个视频\n")
            for i, v in enumerate(all_videos):
                log(f"[{i+1}/{len(all_videos)}] {v}")
                # 重定向 stdout 以实现流式输出
                import io
                buf = io.StringIO()
                old = sys.stdout
                sys.stdout = buf
                try:
                    extract_frames(v, output, interval,
                        min_laplacian=lap, min_entropy=ent,
                        min_brightness=min_b, max_brightness=max_b)
                finally:
                    sys.stdout = old
                log(buf.getvalue())
            log("\n✅ 抽帧完成")

        # ============================================================
        # ③ 数据集划分 Tab
        # ============================================================
        def _build_split_tab(self) -> QWidget:
            w = QWidget()
            layout = QVBoxLayout(w)

            grp_split = QGroupBox("训练/验证集划分")
            fl = QFormLayout(grp_split)
            self._split_dir = QLineEdit("dataset")
            fl.addRow("数据集目录:", self._split_dir)

            self._split_ratio = QDoubleSpinBox()
            self._split_ratio.setRange(0.01, 0.99)
            self._split_ratio.setValue(0.2)
            self._split_ratio.setSingleStep(0.05)
            fl.addRow("验证集比例:", self._split_ratio)

            btn_split = QPushButton("执行划分")
            btn_split.setStyleSheet("font-size: 14px; padding: 8px;")
            btn_split.clicked.connect(self._run_split)
            fl.addRow(btn_split)

            btn_dry = QPushButton("预览划分 (Dry Run)")
            btn_dry.clicked.connect(self._run_split_dry)
            fl.addRow(btn_dry)
            layout.addWidget(grp_split)
            layout.addStretch()
            return w

        def _run_split(self):
            path = self._split_dir.text().strip()
            ratio = self._split_ratio.value()
            self._start_task(
                lambda log: self._do_split(log, path, ratio),
                "数据集划分",
            )

        def _do_split(self, log, path, ratio):
            import io
            from .split_dataset import split_dataset
            from pathlib import Path
            buf = io.StringIO()
            old = sys.stdout
            sys.stdout = buf
            try:
                split_dataset(Path(path), val_ratio=ratio, seed=42)
            finally:
                sys.stdout = old
            log(buf.getvalue() or "✅ 划分完成")

        def _run_split_dry(self):
            """预览数据集划分结果（不实际复制文件）。"""
            path = self._split_dir.text().strip()
            ratio = self._split_ratio.value()
            self._start_task(
                lambda log: self._do_split_dry(log, path, ratio),
                "划分预览",
            )

        def _do_split_dry(self, log, path, ratio):
            import io
            from .split_dataset import split_dataset
            from pathlib import Path
            buf = io.StringIO()
            old = sys.stdout
            sys.stdout = buf
            try:
                split_dataset(Path(path), val_ratio=ratio, seed=42, dry_run=True)
            finally:
                sys.stdout = old
            log(buf.getvalue() or "✅ 预览完成")

        # ============================================================
        # 标注管理 Tab
        # ============================================================
        def _build_label_tab(self) -> QWidget:
            w = QWidget()
            layout = QVBoxLayout(w)

            # 连接配置（本机 / 远程）
            grp_conn = QGroupBox("Label Studio 连接")
            cl = QFormLayout(grp_conn)
            self._ls_url = QLineEdit(resolve_base_url())
            self._ls_url.setPlaceholderText(DEFAULT_BASE_URL)
            self._ls_url.textEdited.connect(self._ls_invalidate_connection)
            cl.addRow("地址:", self._ls_url)
            self._ls_token = QLineEdit()
            _echo = getattr(QLineEdit, "Password", None)
            if _echo is None and hasattr(QLineEdit, "EchoMode"):
                _echo = QLineEdit.EchoMode.Password
            if _echo is not None:
                self._ls_token.setEchoMode(_echo)
            self._ls_token.setPlaceholderText("远程必填；本机可留空（读本地 DB / 环境变量）")
            env_tok = os.environ.get("LABEL_STUDIO_TOKEN", "")
            if env_tok:
                self._ls_token.setText(env_tok)
            self._ls_token.textEdited.connect(self._ls_invalidate_connection)
            cl.addRow("Token:", self._ls_token)
            conn_btn_row = QHBoxLayout()
            btn_connect = QPushButton("连接 / 重连")
            btn_connect.clicked.connect(self._ls_reconnect)
            conn_btn_row.addWidget(btn_connect)
            self._ls_conn_status = QLabel()
            _set_status(
                self._ls_conn_status,
                "未连接（填写地址/Token 后点「连接 / 重连」）", "warn")
            conn_btn_row.addWidget(self._ls_conn_status)
            conn_btn_row.addStretch()
            cl.addRow(conn_btn_row)
            layout.addWidget(grp_conn)

            # 项目表格
            grp_proj = QGroupBox("项目列表")
            pl = QVBoxLayout(grp_proj)
            btn_row = QHBoxLayout()
            self._ls_refresh_btn = QPushButton("刷新")
            self._ls_refresh_btn.clicked.connect(self._ls_refresh_projects)
            btn_row.addWidget(self._ls_refresh_btn)
            btn_create = QPushButton("新建项目")
            btn_create.clicked.connect(self._ls_create_project)
            btn_row.addWidget(btn_create)
            btn_del = QPushButton("删除项目")
            btn_del.setStyleSheet("color: red;")
            btn_del.clicked.connect(self._ls_delete_project)
            btn_row.addWidget(btn_del)
            btn_row.addStretch()
            self._ls_auto_cb = QCheckBox("自动刷新")
            self._ls_auto_cb.setChecked(True)
            self._ls_auto_cb.setToolTip("停留在本页时每 15 秒静默刷新项目状态")
            self._ls_auto_cb.toggled.connect(self._ls_toggle_auto)
            btn_row.addWidget(self._ls_auto_cb)
            self._ls_last_refresh = QLabel("尚未刷新")
            _set_status(self._ls_last_refresh, "尚未刷新", "idle")
            btn_row.addWidget(self._ls_last_refresh)
            pl.addLayout(btn_row)

            self._ls_project_table = QTableWidget()
            self._ls_project_table.setColumnCount(5)
            self._ls_project_table.setHorizontalHeaderLabels(
                ["ID", "标题", "任务数", "已标注", "进度"])
            self._ls_project_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
            for col in (0, 2, 3, 4):
                self._ls_project_table.horizontalHeader().setSectionResizeMode(
                    col, QHeaderView.ResizeToContents)
            self._ls_project_table.setSelectionBehavior(QTableWidget.SelectRows)
            self._ls_project_table.setEditTriggers(QTableWidget.NoEditTriggers)
            self._ls_project_table.itemSelectionChanged.connect(self._ls_on_project_selected)
            pl.addWidget(self._ls_project_table)
            layout.addWidget(grp_proj)

            # 任务表格
            grp_task = QGroupBox("任务管理")
            tl = QVBoxLayout(grp_task)
            ctrl_row = QHBoxLayout()
            ctrl_row.addWidget(QLabel("项目 ID:"))
            self._ls_task_project = QSpinBox()
            self._ls_task_project.setRange(1, 9999)
            self._ls_task_project.setValue(10)
            ctrl_row.addWidget(self._ls_task_project)
            btn_load = QPushButton("加载任务")
            btn_load.clicked.connect(self._ls_load_tasks)
            ctrl_row.addWidget(btn_load)
            btn_stats = QPushButton("统计")
            btn_stats.clicked.connect(self._ls_show_stats)
            ctrl_row.addWidget(btn_stats)
            btn_del_t = QPushButton("删除选中")
            btn_del_t.setStyleSheet("color: red;")
            btn_del_t.clicked.connect(self._ls_delete_tasks)
            ctrl_row.addWidget(btn_del_t)
            ctrl_row.addStretch()
            tl.addLayout(ctrl_row)

            self._ls_task_table = QTableWidget()
            self._ls_task_table.setColumnCount(4)
            self._ls_task_table.setHorizontalHeaderLabels(["ID", "已标注", "标注数", "数据"])
            self._ls_task_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
            self._ls_task_table.setSelectionBehavior(QTableWidget.SelectRows)
            self._ls_task_table.setEditTriggers(QTableWidget.NoEditTriggers)
            tl.addWidget(self._ls_task_table)
            layout.addWidget(grp_task)

            # 批量上传图片
            grp_upload = QGroupBox("批量上传图片")
            ul = QVBoxLayout(grp_upload)
            up_row = QHBoxLayout()
            up_row.addWidget(QLabel("目标项目 ID:"))
            self._ls_upload_project = QSpinBox()
            self._ls_upload_project.setRange(1, 9999)
            self._ls_upload_project.setValue(10)
            up_row.addWidget(self._ls_upload_project)
            btn_select = QPushButton("选择图片...")
            btn_select.clicked.connect(self._ls_select_images)
            up_row.addWidget(btn_select)
            btn_folder = QPushButton("选择文件夹...")
            btn_folder.clicked.connect(self._ls_select_folder)
            up_row.addWidget(btn_folder)
            self._ls_upload_btn = QPushButton("上传")
            self._ls_upload_btn.setStyleSheet(PRIMARY_BTN_QSS)
            self._ls_upload_btn.clicked.connect(self._ls_upload_images)
            up_row.addWidget(self._ls_upload_btn)
            up_row.addStretch()
            ul.addLayout(up_row)
            self._ls_upload_info = QLabel("未选择文件")
            _set_status(self._ls_upload_info, "未选择文件", "idle")
            ul.addWidget(self._ls_upload_info)
            layout.addWidget(grp_upload)

            # 自动刷新：连接成功且停留在 ② 页签时，每 15s 静默刷新项目状态
            self._ls_connected_ok = False
            self._ls_auto_busy = False
            self._ls_timer = QTimer(self)
            self._ls_timer.setInterval(15000)
            self._ls_timer.timeout.connect(self._ls_auto_tick)

            return w

        def _ls_invalidate_connection(self, *_args):
            """地址 / Token 被手动修改后，旧连接失效，③ 同步回到未连接。"""
            if self.client is None:
                return
            self.client = None
            self._ls_connected_ok = False
            self._ls_timer.stop()
            _set_status(self._ls_conn_status,
                        "连接配置已修改，请点「连接 / 重连」", "warn")
            self._pre_update_connection()
            self._rv_invalidate()

        def _ls_reconnect(self):
            """按界面上的地址 / Token 连接本机或远程 Label Studio。"""
            url = self._ls_url.text().strip() or DEFAULT_BASE_URL
            token = self._ls_token.text().strip() or None
            try:
                reset_clients()
                self.client = get_client(base_url=url, token=token, force_new=True)
                # 轻量探测（SDK）
                self.client.whoami()
                _set_status(self._ls_conn_status,
                            f"已连接: {self.client.base_url}", "ok")
                self._ls_connected_ok = True
                self._ls_timer.start()
                self._ls_refresh_projects()
                # 连接状态同步到 ③ 预标注（否则仍显示“请先连接”）
                self._pre_refresh_models()
                self._pre_update_connection()
                self._rv_invalidate()
            except Exception as e:
                self.client = None
                self._ls_connected_ok = False
                self._ls_timer.stop()
                _set_status(self._ls_conn_status, f"连接失败: {e}", "err")
                self._pre_update_connection()
                QMessageBox.warning(self, "连接失败", str(e))

        def _ls_require_client(self) -> bool:
            if self.client is None:
                QMessageBox.warning(self, "未连接", "请先填写地址/Token 并点击「连接 / 重连」。")
                return False
            return True

        def _ls_refresh_projects(self, quiet: bool = False):
            if not self._ls_require_client():
                return
            if self._ls_auto_busy:
                return
            self._ls_auto_busy = True
            self._ls_refresh_btn.setEnabled(False)
            if not quiet:
                _set_status(self._ls_last_refresh, "刷新中…", "warn")
            selected = self._ls_selected_project_id()

            def settle():
                self._ls_auto_busy = False
                self._ls_refresh_btn.setEnabled(True)

            self._run_task(
                lambda: list_projects(self.client),
                lambda ps: self._ls_populate_projects(ps, selected),
                "加载项目",
                show_progress=not quiet,
                quiet_errors=quiet,
                on_settle=settle,
                on_error=lambda e: _set_status(
                    self._ls_last_refresh, "刷新失败", "err"),
            )

        def _ls_selected_project_id(self):
            rows = self._ls_project_table.selectionModel().selectedRows()
            if not rows:
                return None
            item = self._ls_project_table.item(rows[0].row(), 0)
            return int(item.text()) if item else None

        def _ls_populate_projects(self, projects, select_id=None):
            self._ls_connected_ok = True
            table = self._ls_project_table
            table.blockSignals(True)
            table.setRowCount(len(projects))
            for i, p in enumerate(projects):
                pid = p["id"]
                total = p.get("task_number") or 0
                labeled = p.get("num_tasks_with_annotations") or 0
                pct = f"{labeled / total * 100:.0f}%" if total else "—"
                table.setItem(i, 0, QTableWidgetItem(str(pid)))
                table.setItem(i, 1, QTableWidgetItem(p.get("title", "")))
                table.setItem(i, 2, QTableWidgetItem(str(total)))
                table.setItem(i, 3, QTableWidgetItem(str(labeled)))
                item = QTableWidgetItem(pct)
                item.setToolTip(f"{labeled} / {total}")
                table.setItem(i, 4, item)
            table.blockSignals(False)
            table.clearSelection()
            ids = [int(p["id"]) for p in projects]
            target = select_id
            if target is None and ids and self._pre_project.value() not in ids:
                # 远程项目列表里没有 ③ 当前项目 ID：同步到第一个项目，
                # 避免 ③ 还停留在启动时的本地默认值 10。
                target = ids[0]
            if target is not None:
                # 静默恢复选中行：不触发 selectionChanged，避免自动刷新覆盖
                # 用户手动输入的任务/上传项目 ID。
                table.blockSignals(True)
                for row in range(table.rowCount()):
                    if table.item(row, 0).text() == str(target):
                        table.selectRow(row)
                        break
                table.blockSignals(False)
                if select_id is None and target != self._pre_project.value():
                    self._ls_task_project.setValue(target)
            if self._ls_auto_cb.isChecked() and not self._ls_timer.isActive():
                self._ls_timer.start()
            from datetime import datetime as _dt
            _set_status(self._ls_last_refresh,
                        f"最后刷新 {_dt.now():%H:%M:%S}", "ok")

        def _ls_toggle_auto(self, enabled: bool):
            if enabled and self._ls_connected_ok:
                self._ls_timer.start()
                self._ls_auto_tick()
            elif not enabled:
                self._ls_timer.stop()

        def _ls_auto_tick(self):
            if not self._ls_auto_cb.isChecked():
                return
            if not self._ls_connected_ok or self.client is None:
                return
            if self._ls_auto_busy or self._workers:
                return
            if getattr(self, "_tabs", None) is not None and self._tabs.currentIndex() != 1:
                return
            self._ls_refresh_projects(quiet=True)

        def _ls_on_project_selected(self):
            """项目列表选中行时，自动同步项目 ID 到任务管理和上传区域。"""
            rows = self._ls_project_table.selectionModel().selectedRows()
            if not rows:
                return
            pid = int(self._ls_project_table.item(rows[0].row(), 0).text())
            self._ls_task_project.setValue(pid)
            self._ls_upload_project.setValue(pid)

        def _ls_create_project(self):
            if not self._ls_require_client():
                return
            from PySide6.QtWidgets import QInputDialog
            title, ok = QInputDialog.getText(self, "新建项目", "项目名称:")
            if ok and title:
                self._run_task(
                    lambda: create_project(title, template="detection", client=self.client),
                    lambda p: (QMessageBox.information(self, "完成", f"项目已创建 ID={p['id']}"),
                               self._ls_refresh_projects()),
                    "创建项目",
                )

        def _ls_delete_project(self):
            if not self._ls_require_client():
                return
            rows = self._ls_project_table.selectionModel().selectedRows()
            if not rows:
                return
            pid = int(self._ls_project_table.item(rows[0].row(), 0).text())
            ans = QMessageBox.question(self, "确认", f"删除项目 {pid}？", QMessageBox.Yes | QMessageBox.No)
            if ans == QMessageBox.Yes:
                self._run_task(
                    lambda: delete_project(pid, self.client),
                    lambda _: (QMessageBox.information(self, "完成", f"已删除"), self._ls_refresh_projects()),
                    "删除项目",
                )

        def _ls_load_tasks(self):
            if not self._ls_require_client():
                return
            pid = self._ls_task_project.value()
            self._run_task(
                lambda: list_tasks(pid, page=1, page_size=100, client=self.client),
                self._ls_populate_tasks,
                "加载任务",
            )

        def _ls_populate_tasks(self, tasks):
            self._ls_task_table.setRowCount(len(tasks))
            for i, t in enumerate(tasks):
                self._ls_task_table.setItem(i, 0, QTableWidgetItem(str(t["id"])))
                self._ls_task_table.setItem(i, 1, QTableWidgetItem("✅" if t.get("is_labeled") else "⬜"))
                self._ls_task_table.setItem(i, 2, QTableWidgetItem(str(t.get("total_annotations", 0))))
                ds = str(t.get("data", {}))
                self._ls_task_table.setItem(i, 3, QTableWidgetItem(ds[:80] + "..." if len(ds) > 80 else ds))

        def _ls_show_stats(self):
            if not self._ls_require_client():
                return
            pid = self._ls_task_project.value()
            self._run_task(
                lambda: task_stats(pid, self.client),
                lambda s: QMessageBox.information(self, f"项目 {pid} 统计",
                    f"总任务: {s['total']}\n已标注: {s['labeled']} ({s['labeled']/max(s['total'],1)*100:.1f}%)\n未标注: {s['unlabeled']}"),
                "统计",
            )

        def _ls_delete_tasks(self):
            if not self._ls_require_client():
                return
            rows = self._ls_task_table.selectionModel().selectedRows()
            if not rows:
                return
            ids = [int(self._ls_task_table.item(r.row(), 0).text()) for r in rows]
            ans = QMessageBox.question(self, "确认", f"删除 {len(ids)} 个任务？", QMessageBox.Yes | QMessageBox.No)
            if ans == QMessageBox.Yes:
                pid = self._ls_task_project.value()
                self._run_task(
                    lambda: delete_tasks_batch(pid, ids, self.client),
                    lambda _: (QMessageBox.information(self, "完成", f"已删除 {len(ids)} 个"), self._ls_load_tasks()),
                    "删除任务",
                )

        _IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff", ".tif"}

        @classmethod
        def _scan_images(cls, root: str) -> list[str]:
            """递归扫描目录下所有图片文件。"""
            result = []
            try:
                for entry in os.scandir(root):
                    if entry.is_file() and os.path.splitext(entry.name)[1].lower() in cls._IMG_EXTS:
                        result.append(entry.path)
                    elif entry.is_dir():
                        result.extend(cls._scan_images(entry.path))
            except PermissionError:
                pass
            return sorted(result)

        def _ls_select_images(self):
            """选择要上传的图片文件。"""
            paths, _ = QFileDialog.getOpenFileNames(
                self, "选择图片", "",
                "图片文件 (*.png *.jpg *.jpeg *.bmp *.gif *.webp);;所有文件 (*)",
            )
            if paths:
                self._ls_image_paths = paths
                _set_status(self._ls_upload_info,
                            f"已选择 {len(paths)} 张图片", "ok")

        def _ls_select_folder(self):
            """选择文件夹，递归扫描其中所有图片。"""
            folder = QFileDialog.getExistingDirectory(self, "选择图片所在文件夹")
            if folder:
                paths = self._scan_images(folder)
                self._ls_image_paths = paths
                if paths:
                    _set_status(self._ls_upload_info,
                                f"已选择 {len(paths)} 张图片（来自 {folder}）", "ok")
                else:
                    _set_status(self._ls_upload_info,
                                f"文件夹内无图片: {folder}", "warn")

        def _ls_upload_images(self):
            """批量上传图片到 Label Studio 项目。"""
            if not self._ls_require_client():
                return
            if not getattr(self, "_ls_image_paths", None):
                QMessageBox.warning(self, "提示", "请先选择要上传的图片文件。")
                return

            pid = self._ls_upload_project.value()
            paths = self._ls_image_paths

            ans = QMessageBox.question(
                self, "确认上传",
                f"将 {len(paths)} 张图片上传到项目 {pid}？\n\n这会为每张图片创建一个标注任务。",
                QMessageBox.Yes | QMessageBox.No,
            )
            if ans != QMessageBox.Yes:
                return

            total = len(paths)
            holder = {}

            def do_upload(log):
                return import_tasks_from_images(
                    pid, paths, client=self.client,
                    progress=lambda done, n: holder["w"].progress.emit(done, n),
                    log=log,
                )

            def on_done(result):
                n = result.get("task_count", 0)
                errs = result.get("errors", [])
                if errs:
                    detail = "\n".join(errs[:10])
                    more = f"\n... 还有 {len(errs) - 10} 个错误" if len(errs) > 10 else ""
                    _set_status(self._ls_upload_info,
                                f"上传完成 {n}/{total}，失败 {len(errs)}", "err")
                    QMessageBox.warning(self, "上传完成（有错误）",
                        f"成功 {n} 个，失败 {len(errs)} 个\n\n失败详情:\n{detail}{more}")
                else:
                    _set_status(self._ls_upload_info,
                                f"上传完成 {n}/{total}（项目 {pid}）", "ok")
                    QMessageBox.information(self, "上传完成",
                        f"已导入 {n} 个任务到项目 {pid}")
                self._ls_image_paths = []
                self._ls_upload_btn.setEnabled(True)
                # 上传后自动刷新：项目计数 + 当前任务列表，不再依赖手动刷新
                self._ls_refresh_projects(quiet=True)
                if self._ls_task_project.value() == pid:
                    self._ls_load_tasks()

            def on_fail(message):
                self._ls_upload_btn.setEnabled(True)
                _set_status(self._ls_upload_info, f"上传失败: {message}", "err")

            self._show_progress(True)
            self._progress.setRange(0, max(total, 1))
            self._progress.setValue(0)
            self._ls_upload_btn.setEnabled(False)
            worker = StreamWorker(do_upload)
            holder["w"] = worker
            worker.progress.connect(
                lambda done, n: self._ls_upload_progress(done, n, pid))
            worker.finished.connect(on_done)
            worker.finished.connect(lambda _: self._show_progress(False))
            worker.finished.connect(lambda _: (worker.deleteLater(),
                self._workers.remove(worker) if worker in self._workers else None))
            worker.error.connect(on_fail)
            worker.error.connect(lambda e: (
                QMessageBox.critical(self, "上传失败", str(e)),
                self._show_progress(False)))
            worker.error.connect(lambda _: (worker.deleteLater(),
                self._workers.remove(worker) if worker in self._workers else None))
            self._workers.append(worker)
            worker.start()

        def _ls_upload_progress(self, done: int, total: int, pid: int):
            self._set_progress(done, total)
            _set_status(self._ls_upload_info,
                        f"上传中 {done}/{total} → 项目 {pid}", "warn")

        # ============================================================
        # ③ 预标注 Tab
        # ============================================================
        _PRE_STAGES = ["评估", "建集", "训练", "导出", "预标注"]
        _PRE_STAGE_KEYS = ["assess", "dataset", "train", "export", "prelabel"]

        def _build_prelabel_tab(self) -> QWidget:
            self._pre_running = False
            self._pre_worker = None
            self._pre_label_cancel = threading.Event()
            self._pre_current_stage = None
            self._pre_pending_pipeline = False
            self._pre_last_run_dir = None

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            w = QWidget()
            scroll.setWidget(w)
            layout = QVBoxLayout(w)

            self._pre_conn_hint = QLabel("请先在 ② 标注管理连接 Label Studio")
            _set_status(self._pre_conn_hint, self._pre_conn_hint.text(), "warn")
            layout.addWidget(self._pre_conn_hint)
            self._pre_conn_info = QLabel("连接: 未连接")
            _set_status(self._pre_conn_info, "连接: 未连接", "idle")
            layout.addWidget(self._pre_conn_info)

            # ---- ① 数据就绪 ----
            grp = QGroupBox("① 数据就绪")
            fl = QFormLayout(grp)
            row = QHBoxLayout()
            row.addWidget(QLabel("项目 ID:"))
            self._pre_project = QSpinBox()
            self._pre_project.setRange(1, 999999)
            self._pre_project.setValue(10)
            row.addWidget(self._pre_project)
            self._pre_check_btn = QPushButton("检查数据")
            self._pre_check_btn.clicked.connect(self._pre_assess_only)
            row.addWidget(self._pre_check_btn)
            row.addStretch()
            self._pre_verdict = QLabel("未评估")
            _set_status(self._pre_verdict, "未评估", "idle")
            row.addWidget(self._pre_verdict)
            fl.addRow(row)

            self._pre_summary = QLabel("—")
            _set_status(self._pre_summary, "—", "idle")
            fl.addRow("摘要:", self._pre_summary)

            self._pre_detail_btn = QPushButton("查看明细 ▸")
            self._pre_detail_btn.setCheckable(True)
            self._pre_detail_btn.toggled.connect(self._pre_toggle_detail)
            fl.addRow(self._pre_detail_btn)

            self._pre_detail_table = QTableWidget(0, 4)
            self._pre_detail_table.setHorizontalHeaderLabels(["检查项", "门槛", "实测", "结论"])
            self._pre_detail_table.horizontalHeader().setSectionResizeMode(
                0, QHeaderView.Stretch)
            self._pre_detail_table.setSelectionBehavior(QTableWidget.SelectRows)
            self._pre_detail_table.setEditTriggers(QTableWidget.NoEditTriggers)
            self._pre_detail_table.setMaximumHeight(150)
            self._pre_detail_table.setVisible(False)
            fl.addRow(self._pre_detail_table)
            layout.addWidget(grp)

            # ---- ② 微数据集与模型 ----
            grp2 = QGroupBox("② 微数据集与模型")
            fl2 = QFormLayout(grp2)
            wrow = QHBoxLayout()
            self._pre_weights = QLineEdit("weights/yolo26n.pt")
            wrow.addWidget(self._pre_weights)
            self._pre_weights_btn = QPushButton("浏览")
            self._pre_weights_btn.clicked.connect(
                lambda: self._browse_file(self._pre_weights, "权重 (*.pt)"))
            wrow.addWidget(self._pre_weights_btn)
            fl2.addRow("基座权重:", wrow)

            params = QHBoxLayout()
            params.addWidget(QLabel("尺寸:"))
            self._pre_imgsz = QSpinBox()
            self._pre_imgsz.setRange(320, 1280)
            self._pre_imgsz.setSingleStep(160)
            self._pre_imgsz.setValue(480)
            params.addWidget(self._pre_imgsz)
            params.addWidget(QLabel("轮数:"))
            self._pre_epochs = QSpinBox()
            self._pre_epochs.setRange(10, 1000)
            self._pre_epochs.setValue(150)
            params.addWidget(self._pre_epochs)
            params.addWidget(QLabel("冻结:"))
            self._pre_freeze = QSpinBox()
            self._pre_freeze.setRange(0, 23)
            self._pre_freeze.setValue(8)
            params.addWidget(self._pre_freeze)
            params.addWidget(QLabel("设备:"))
            self._pre_device = QComboBox()
            self._pre_device.addItems(["自动", "CPU", "GPU"])
            params.addWidget(self._pre_device)
            params.addStretch()
            fl2.addRow(params)

            adv, self._pre_adv = self._build_advanced_group(
                "高级设置（覆盖任意训练参数）", self._pre_advanced_defaults)
            fl2.addRow(adv)

            arow = QHBoxLayout()
            self._pre_artifact = QLabel("上次产物: —")
            _set_status(self._pre_artifact, "上次产物: —", "idle")
            arow.addWidget(self._pre_artifact)
            self._pre_open_dir_btn = QPushButton("打开目录")
            self._pre_open_dir_btn.clicked.connect(self._pre_open_dir)
            arow.addWidget(self._pre_open_dir_btn)
            arow.addStretch()
            fl2.addRow(arow)
            layout.addWidget(grp2)

            # ---- ③ 预标注 ----
            grp3 = QGroupBox("③ 预标注")
            fl3 = QFormLayout(grp3)
            mrow = QHBoxLayout()
            self._pre_model = QComboBox()
            self._pre_model.addItem("自动：最新 ONNX", "")
            self._pre_model.currentIndexChanged.connect(self._pre_on_model_changed)
            mrow.addWidget(self._pre_model)
            self._pre_model_btn = QPushButton("选择…")
            self._pre_model_btn.clicked.connect(self._pre_choose_model)
            mrow.addWidget(self._pre_model_btn)
            fl3.addRow("模型:", mrow)

            prow = QHBoxLayout()
            prow.addWidget(QLabel("置信度:"))
            self._pre_conf = QDoubleSpinBox()
            self._pre_conf.setRange(0.01, 0.99)
            self._pre_conf.setSingleStep(0.05)
            self._pre_conf.setDecimals(2)
            self._pre_conf.setValue(0.25)
            prow.addWidget(self._pre_conf)
            prow.addWidget(QLabel("IOU:"))
            self._pre_iou = QDoubleSpinBox()
            self._pre_iou.setRange(0.10, 0.95)
            self._pre_iou.setSingleStep(0.05)
            self._pre_iou.setDecimals(2)
            self._pre_iou.setValue(0.70)
            prow.addWidget(self._pre_iou)
            self._pre_skip = QCheckBox("跳过同版本预测")
            self._pre_skip.setChecked(True)
            prow.addWidget(self._pre_skip)
            self._pre_overwrite = QCheckBox("覆盖同版本")
            prow.addWidget(self._pre_overwrite)
            self._pre_skip.toggled.connect(
                lambda on: self._pre_overwrite.setEnabled(not on))
            self._pre_overwrite.setEnabled(False)
            prow.addStretch()
            fl3.addRow(prow)

            self._pre_tag = QLabel("版本标签: —")
            _set_status(self._pre_tag, "版本标签: —", "idle")
            fl3.addRow(self._pre_tag)
            layout.addWidget(grp3)

            # ---- ④ 复核 ----
            grp4 = QGroupBox("④ 复核")
            rl = QHBoxLayout(grp4)
            self._pre_review = QLabel("待复核 — 张 · 已提交 — 张")
            _set_status(self._pre_review, "待复核 — 张 · 已提交 — 张", "idle")
            rl.addWidget(self._pre_review)
            rl.addStretch()
            btn_ls = QPushButton("打开 Label Studio")
            btn_ls.clicked.connect(self._pre_open_ls)
            rl.addWidget(btn_ls)
            btn_csv = QPushButton("导出抽样 CSV")
            btn_csv.clicked.connect(self._pre_export_sample)
            rl.addWidget(btn_csv)
            layout.addWidget(grp4)

            # ---- 阶段芯片 ----
            chip_row = QHBoxLayout()
            self._pre_chips = []
            for name in self._PRE_STAGES:
                chip = QLabel(f"○ {name}")
                _set_status(chip, f"○ {name}", "idle")
                self._pre_chips.append(chip)
                chip_row.addWidget(chip)
            chip_row.addStretch()
            layout.addLayout(chip_row)

            self._pre_status = QLabel("请先评估数据")
            _set_status(self._pre_status, "请先评估数据", "idle")
            layout.addWidget(self._pre_status)

            # ---- 操作按钮 ----
            brow = QHBoxLayout()
            self._pre_run_btn = QPushButton("评估并预标注")
            self._pre_run_btn.setStyleSheet(PRIMARY_BTN_QSS)
            self._pre_run_btn.clicked.connect(self._pre_run_clicked)
            brow.addWidget(self._pre_run_btn)
            self._pre_assess_btn = QPushButton("仅评估")
            self._pre_assess_btn.clicked.connect(self._pre_assess_only)
            brow.addWidget(self._pre_assess_btn)
            self._pre_predict_btn = QPushButton("仅预标注")
            self._pre_predict_btn.clicked.connect(self._pre_predict_only)
            brow.addWidget(self._pre_predict_btn)
            self._pre_cancel_btn = QPushButton("取消")
            self._pre_cancel_btn.setEnabled(False)
            self._pre_cancel_btn.clicked.connect(self._pre_cancel)
            brow.addWidget(self._pre_cancel_btn)
            brow.addStretch()
            layout.addLayout(brow)
            layout.addStretch()
            return scroll

        # ---- 预标注：状态与联动 ----
        def _pre_controls(self) -> list:
            return [self._pre_project, self._pre_check_btn, self._pre_weights,
                    self._pre_weights_btn, self._pre_imgsz, self._pre_epochs,
                    self._pre_freeze, self._pre_device, self._pre_model,
                    self._pre_model_btn, self._pre_conf, self._pre_iou,
                    self._pre_skip, self._pre_overwrite, self._pre_adv]

        def _pre_update_connection(self):
            ok = self.client is not None
            self._pre_conn_hint.setVisible(not ok)
            if ok:
                _set_status(self._pre_conn_info,
                            f"连接: {self.client.base_url}（来自 ② 标注管理）", "ok")
            else:
                _set_status(self._pre_conn_info, "连接: 未连接", "idle")
            self._pre_apply_enabled()

        def _pre_apply_enabled(self):
            enabled = self.client is not None and not self._pre_running
            for widget in self._pre_controls():
                widget.setEnabled(enabled)
            self._pre_run_btn.setEnabled(enabled)
            self._pre_assess_btn.setEnabled(enabled)
            self._pre_predict_btn.setEnabled(enabled)
            self._pre_cancel_btn.setEnabled(self._pre_running)
            self._pre_run_btn.setText("运行中…" if self._pre_running else "评估并预标注")

        def _pre_set_running(self, running: bool):
            self._pre_running = running
            self._pre_apply_enabled()

        def _pre_on_project_changed(self, _value):
            if self._pre_running:
                return
            _set_status(self._pre_verdict, "未评估", "idle")
            _set_status(self._pre_summary, "—", "idle")
            _set_status(self._pre_tag, "版本标签: —", "idle")
            self._pre_detail_table.setRowCount(0)

        def _pre_toggle_detail(self, checked: bool):
            self._pre_detail_table.setVisible(checked)
            self._pre_detail_btn.setText("收起明细 ▾" if checked else "查看明细 ▸")

        def _pre_chip(self, index: int, state: str):
            mark = {"idle": "○", "run": "◐", "ok": "●", "err": "✕"}[state]
            level = {"idle": "idle", "run": "warn", "ok": "ok", "err": "err"}[state]
            _set_status(self._pre_chips[index],
                        f"{mark} {self._PRE_STAGES[index]}", level)

        def _pre_reset_chips(self):
            for i in range(len(self._pre_chips)):
                self._pre_chip(i, "idle")

        def _pre_show_assessment(self, data: dict):
            verdict = data.get("verdict", "")
            text = {"ready": "● 可训练", "marginal": "◐ 勉强可训练",
                    "insufficient": "✕ 数据不足"}.get(verdict, verdict)
            level = {"ready": "ok", "marginal": "warn"}.get(verdict, "err")
            _set_status(self._pre_verdict, text, level)

            per_class = data.get("per_class") or {}
            covered = sum(1 for v in per_class.values() if v > 0)
            summary = (f"{data.get('labeled_tasks', 0)} 张 · "
                       f"{data.get('instances', 0)} 个框 · "
                       f"{covered}/{len(per_class)} 类 · "
                       f"小目标 {float(data.get('small_ratio', 0)) * 100:.0f}% · "
                       f"无效框 {data.get('invalid_boxes', 0)}")
            _set_status(self._pre_summary, summary, level)

            checks = data.get("checks") or []
            table = self._pre_detail_table
            table.setRowCount(len(checks))
            marks = {"pass": "● 通过", "warn": "◐ 警告", "fail": "✕ 失败"}
            for i, check in enumerate(checks):
                table.setItem(i, 0, QTableWidgetItem(str(check.get("item", ""))))
                table.setItem(i, 1, QTableWidgetItem(str(check.get("require", ""))))
                table.setItem(i, 2, QTableWidgetItem(str(check.get("actual", ""))))
                table.setItem(i, 3, QTableWidgetItem(
                    marks.get(check.get("level"), str(check.get("level", "")))))

            suggestion = data.get("suggestion")
            if suggestion:
                _set_status(self._pre_status, suggestion,
                            "warn" if verdict == "marginal" else "err")

        def _pre_on_model_changed(self, _index=None):
            """选中带 manifest 的模型时，自动采用其导出 imgsz。"""
            model = self._pre_model.currentData()
            if not model:
                return
            try:
                import json
                data = json.loads(
                    (Path(model).parent / "manifest.json").read_text())
                imgsz = int(data.get("imgsz", 0))
            except (OSError, ValueError, TypeError):
                return
            if self._pre_imgsz.minimum() <= imgsz <= self._pre_imgsz.maximum():
                self._pre_imgsz.setValue(imgsz)

        def _pre_refresh_models(self):
            try:
                from .prelabel import RUN_ROOT
            except Exception:
                return
            current = self._pre_model.currentData()
            self._pre_model.blockSignals(True)
            self._pre_model.clear()
            self._pre_model.addItem("自动：最新 ONNX", "")
            for path in sorted(RUN_ROOT.glob("*/model/*.onnx")):
                self._pre_model.addItem(
                    f"{path.parent.parent.name} · {path.name}", str(path))
            if current:
                idx = self._pre_model.findData(current)
                if idx >= 0:
                    self._pre_model.setCurrentIndex(idx)
            self._pre_model.blockSignals(False)

        # ---- 预标注：运行入口 ----
        def _pre_assess_only(self):
            if not self._ls_require_client():
                return
            self._pre_pending_pipeline = False
            self._pre_start_worker(self._pre_make_assess_fn(), "assess")

        def _pre_run_clicked(self):
            if not self._ls_require_client():
                return
            self._pre_pending_pipeline = True
            self._pre_start_worker(self._pre_make_assess_fn(), "assess")

        def _pre_make_assess_fn(self):
            project_id = self._pre_project.value()

            def fn(log, on_event):
                from .prelabel import assess_project
                return assess_project(project_id, client=self.client,
                                      log=log).to_dict()
            return fn

        def _pre_device_value(self) -> str:
            return ["auto", "cpu", "0"][self._pre_device.currentIndex()]

        def _pre_start_pipeline(self):
            train_overrides = self._advanced_overrides(
                self._pre_adv, "预标注训练高级参数")
            if train_overrides is None:
                return
            project_id = self._pre_project.value()
            weights = self._pre_weights.text().strip() or "weights/yolo26n.pt"
            epochs = self._pre_epochs.value()
            imgsz = self._pre_imgsz.value()
            conf = self._pre_conf.value()
            iou = self._pre_iou.value()
            freeze = self._pre_freeze.value()
            device = self._pre_device_value()

            def fn(log, on_event):
                from .prelabel import run_pipeline
                return run_pipeline(
                    project_id, base_weights=weights, epochs=epochs,
                    imgsz=imgsz, conf=conf, iou=iou, freeze=freeze,
                    device=device, train_overrides=train_overrides,
                    client=self.client, on_event=on_event,
                    should_stop=self._pre_label_cancel.is_set, log=log)

            self._pre_start_worker(fn, "pipeline")

        def _pre_predict_only(self):
            if not self._ls_require_client():
                return
            project_id = self._pre_project.value()
            imgsz = self._pre_imgsz.value()
            conf = self._pre_conf.value()
            iou = self._pre_iou.value()
            model = self._pre_model.currentData()
            if not model:
                try:
                    from .prelabel import RUN_ROOT
                    models = sorted(RUN_ROOT.glob("*/model/*.onnx"))
                except Exception:
                    models = []
                if not models:
                    QMessageBox.warning(self, "提示", "没有可用的 ONNX 模型，请先训练。")
                    return
                model = str(models[-1])

            def fn(log, on_event):
                from .prelabel import prelabel_project, tag_from_manifest, make_tag
                tag = tag_from_manifest(model, imgsz) or make_tag("manual", imgsz)
                return prelabel_project(
                    project_id, model=model, tag=tag, conf=conf, iou=iou,
                    imgsz=imgsz,
                    overwrite=self._pre_overwrite.isChecked(),
                    device=self._pre_device_value(), on_event=on_event,
                    should_stop=self._pre_label_cancel.is_set,
                    client=self.client, log=log).to_dict()

            self._pre_start_worker(fn, "predict")

        def _pre_start_worker(self, fn, mode: str):
            self._pre_mode = mode
            self._pre_label_cancel.clear()
            self._pre_reset_chips()
            self._pre_current_stage = None
            self._pre_set_running(True)
            self._show_progress(True)

            worker = PrelabelWorker(fn)
            worker.log_line.connect(self._log_append)
            worker.event.connect(self._pre_on_event)
            worker.finished.connect(self._pre_on_finished)
            worker.error.connect(self._pre_on_error)
            worker.finished.connect(lambda _: self._pre_worker_done(worker))
            worker.error.connect(lambda _: self._pre_worker_done(worker))
            self._pre_worker = worker
            self._workers.append(worker)
            worker.start()

        def _pre_worker_done(self, worker):
            # 只有当前 worker 结束时才复位状态；否则会误伤刚启动的下一阶段。
            if self._pre_worker is worker:
                self._show_progress(False)
                self._pre_set_running(False)
                self._pre_worker = None
            worker.deleteLater()
            if worker in self._workers:
                self._workers.remove(worker)

        def _pre_cancel(self):
            self._pre_label_cancel.set()
            _set_status(self._pre_status, "正在取消…（当前步骤结束后停止）", "warn")
            self._pre_cancel_btn.setEnabled(False)

        # ---- 预标注：事件与结果 ----
        def _pre_on_event(self, event):
            stage = getattr(event, "stage", "")
            status = getattr(event, "status", "")
            message = getattr(event, "message", "")
            fraction = getattr(event, "fraction", None)
            visible = getattr(event, "visible", True)

            if stage in self._PRE_STAGE_KEYS:
                idx = self._PRE_STAGE_KEYS.index(stage)
                if status == "start":
                    self._pre_current_stage = stage
                    self._pre_chip(idx, "run")
                elif status == "done":
                    self._pre_chip(idx, "ok")
                elif status == "error":
                    self._pre_chip(idx, "err")

            if message and visible:
                self._log_append(message)
                _set_status(self._pre_status, message, "warn")

            if fraction is not None:
                self._progress.setVisible(True)
                self._progress.setRange(0, 100)
                self._progress.setValue(max(0, min(100, int(fraction * 100))))
            if status in ("done", "error"):
                self._progress.setVisible(False)

        def _pre_on_finished(self, report):
            if self._pre_mode == "assess":
                data = report
                if data:
                    self._pre_show_assessment(data)
                    _set_status(self._pre_status,
                                f"评估完成：{data.get('verdict', '')}", "ok")
                if self._pre_pending_pipeline:
                    self._pre_pending_pipeline = False
                    verdict = (data or {}).get("verdict")
                    if verdict == "insufficient":
                        QMessageBox.warning(
                            self, "数据不足",
                            (data or {}).get("suggestion") or "请先补充标注。")
                        return
                    if verdict == "marginal":
                        ans = QMessageBox.question(
                            self, "数据偏少",
                            ((data or {}).get("suggestion") or "数据偏少") +
                            "\n\n仍要继续训练并预标注吗？",
                            QMessageBox.Yes | QMessageBox.No)
                        if ans != QMessageBox.Yes:
                            return
                    self._pre_start_pipeline()
                return

            report = report or {}
            if self._pre_mode == "pipeline":
                data = report.get("assessment")
                if data:
                    self._pre_show_assessment(data)
                dataset = report.get("dataset")
                if dataset:
                    _set_status(
                        self._pre_summary,
                        f"{dataset.get('labeled_tasks', 0)} 张 · "
                        f"{dataset.get('instances', 0)} 个框 · "
                        f"train {dataset.get('train', 0)} / "
                        f"val {dataset.get('val', 0)}", "ok")
                train = report.get("train") or {}
                if train:
                    self._pre_last_run_dir = train.get("run_dir")
                    metrics = train.get("metrics") or {}
                    _set_status(
                        self._pre_artifact,
                        f"上次产物: {Path(str(train.get('run_dir', ''))).name} · "
                        f"mAP50 {metrics.get('mAP50', 0):.2f}", "ok")
                manifest = report.get("manifest") or {}
                if manifest.get("tag"):
                    _set_status(self._pre_tag,
                                f"版本标签: {manifest['tag']}", "ok")
                self._pre_show_prelabel_stats(report.get("prelabel"))
                self._pre_refresh_models()
                lines = ["流水线完成。"]
                if train:
                    metrics = train.get("metrics") or {}
                    lines.append(f"mAP50: {metrics.get('mAP50', 0):.2f}")
                stats = report.get("prelabel") or {}
                lines.append(f"预标注: 写入 {stats.get('written', 0)} 张 / "
                             f"失败 {stats.get('failed', 0)} 张 / "
                             f"框 {stats.get('boxes', 0)}")
                QMessageBox.information(self, "完成", "\n".join(lines))

            elif self._pre_mode == "predict":
                self._pre_show_prelabel_stats(report)
                self._pre_refresh_models()

        def _pre_show_prelabel_stats(self, stats):
            if not stats:
                return
            written = stats.get("written", 0)
            failed = stats.get("failed", 0)
            skipped = stats.get("skipped", 0)
            _set_status(
                self._pre_review,
                f"待复核 {written} 张 · 失败 {failed} 张 · 跳过 {skipped} 张",
                "warn" if failed else "ok")
            _set_status(
                self._pre_status,
                f"预标注完成：写入 {written} 张 / 失败 {failed} 张", "ok")

        def _pre_on_error(self, message: str):
            self._pre_pending_pipeline = False
            if "已取消" in str(message):
                _set_status(self._pre_status, "已取消", "idle")
                self._log_append("⏹ 已取消")
                self._pre_reset_chips()
                return
            self._log_append(f"❌ {message}")
            _set_status(self._pre_status, f"失败：{message}", "err")
            if self._pre_current_stage in self._PRE_STAGE_KEYS:
                idx = self._PRE_STAGE_KEYS.index(self._pre_current_stage)
                self._pre_chip(idx, "err")
            QMessageBox.critical(self, "预标注失败", str(message))

        # ---- 预标注：辅助操作 ----
        def _pre_choose_model(self):
            path, _ = QFileDialog.getOpenFileName(
                self, "选择 ONNX 模型", "", "ONNX (*.onnx)")
            if path:
                self._pre_model.addItem(f"手动: {os.path.basename(path)}", path)
                self._pre_model.setCurrentIndex(self._pre_model.count() - 1)

        def _pre_open_ls(self):
            if self.client is not None:
                webbrowser.open(self.client.base_url)

        def _pre_open_dir(self):
            path = self._pre_last_run_dir
            if not path:
                try:
                    from .prelabel import RUN_ROOT
                except Exception:
                    return
                dirs = sorted(RUN_ROOT.glob("*/")) if RUN_ROOT.exists() else []
                if not dirs:
                    return
                path = str(dirs[-1])
            if not os.path.isdir(str(path)):
                return
            import subprocess
            if os.name == "nt":
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])

        def _pre_export_sample(self):
            if not self._ls_require_client():
                return
            path, _ = QFileDialog.getSaveFileName(
                self, "导出复核抽样", "prelabel_sample.csv", "CSV (*.csv)")
            if not path:
                return
            project_id = self._pre_project.value()

            def do_export():
                import csv
                from .label_studio_data import list_tasks
                tasks = list_tasks(project_id, page=1, page_size=1000,
                                   client=self.client)
                by_task: dict[int, list] = {}
                for pred in self.client.predictions.list(project=project_id):
                    version = getattr(pred, "model_version", "") or ""
                    if version and not str(version).startswith("ctts-prelabel:"):
                        continue
                    task_id = getattr(pred, "task", None)
                    if task_id is not None:
                        by_task.setdefault(int(task_id), []).append(pred)

                with open(path, "w", newline="", encoding="utf-8-sig") as fh:
                    writer = csv.writer(fh)
                    writer.writerow(["task_id", "image", "is_labeled",
                                     "pred_boxes", "max_score"])
                    for task in tasks:
                        preds = by_task.get(int(task["id"]), [])
                        boxes = sum(len(getattr(p, "result", []) or [])
                                    for p in preds)
                        scores = [float(getattr(p, "score", 0) or 0)
                                  for p in preds]
                        writer.writerow([
                            task["id"],
                            str((task.get("data") or {}).get("image", ""))[:300],
                            "yes" if task.get("is_labeled") else "no",
                            boxes,
                            f"{max(scores):.3f}" if scores else "",
                        ])
                return len(tasks)

            self._run_task(
                do_export,
                lambda n: QMessageBox.information(
                    self, "导出完成", f"已导出 {n} 条任务到\n{path}"),
                "导出抽样")

        # ============================================================
        # 训练 Tab
        # ============================================================
        def _build_train_tab(self) -> QWidget:
            w = QWidget()
            layout = QVBoxLayout(w)

            grp = QGroupBox("YOLO 训练配置 (yolo_train)")
            fl = QFormLayout(grp)

            self._train_weights = QLineEdit("weights/yolo26n.pt")
            fl.addRow("预训练权重:", self._train_weights)

            self._train_data = QLineEdit("dataset/LabelData/data.yaml")
            fl.addRow("数据集 yaml:", self._train_data)

            self._train_epochs = QSpinBox()
            self._train_epochs.setRange(1, 1000)
            self._train_epochs.setValue(40)
            fl.addRow("Epochs:", self._train_epochs)

            self._train_batch = QSpinBox()
            self._train_batch.setRange(1, 128)
            self._train_batch.setValue(4)
            fl.addRow("Batch size:", self._train_batch)

            self._train_imgsz = QSpinBox()
            self._train_imgsz.setRange(128, 2048)
            self._train_imgsz.setValue(640)
            self._train_imgsz.setSingleStep(32)
            fl.addRow("Image size:", self._train_imgsz)

            self._train_device = QComboBox()
            self._train_device.addItems(["0", "cpu"])
            fl.addRow("设备:", self._train_device)

            adv, self._train_adv = self._build_advanced_group(
                "高级设置（覆盖任意训练参数）", self._train_advanced_defaults)
            fl.addRow(adv)

            btn_train = QPushButton("开始训练")
            btn_train.setStyleSheet("font-size: 14px; padding: 8px;")
            btn_train.clicked.connect(self._run_train)
            fl.addRow(btn_train)

            layout.addWidget(grp)
            return w

        def _run_train(self):
            weights = self._train_weights.text().strip()
            data = self._train_data.text().strip()
            epochs = self._train_epochs.value()
            batch = self._train_batch.value()
            imgsz = self._train_imgsz.value()
            device = self._train_device.currentText()

            if not os.path.exists(weights):
                QMessageBox.warning(self, "提示", f"权重文件不存在: {weights}")
                return

            overrides = self._advanced_overrides(self._train_adv, "训练高级参数")
            if overrides is None:
                return

            self._start_task(
                lambda log: self._do_train(
                    log, weights, data, epochs, batch, imgsz, device, overrides),
                "YOLO 训练",
            )

        def _do_train(self, log, weights, data, epochs, batch, imgsz, device,
                      overrides):
            import io
            from .yolo_train import run_training
            log(f"🚀 启动训练: {weights}\n")
            buf = io.StringIO()
            old = sys.stdout
            sys.stdout = buf
            try:
                run_training(weights, data=data, epochs=epochs, batch=batch,
                             imgsz=imgsz, device=device, overrides=overrides,
                             log=log)
            except Exception as e:
                log(f"训练出错: {e}")
            finally:
                sys.stdout = old
            log(buf.getvalue() or "\n✅ 训练完成")

        # ============================================================
        # 推理 Tab
        # ============================================================
        def _build_detect_tab(self) -> QWidget:
            w = QWidget()
            layout = QVBoxLayout(w)

            grp = QGroupBox("YOLO 实时推理 (detect)")
            fl = QFormLayout(grp)

            self._detect_weights = QLineEdit("weights/yolo26n.pt")
            fl.addRow("权重:", self._detect_weights)

            self._detect_source = QComboBox()
            self._detect_source.addItems(["摄像头 0", "摄像头 1", "视频文件..."])
            fl.addRow("输入源:", self._detect_source)

            self._detect_conf = QDoubleSpinBox()
            self._detect_conf.setRange(0.01, 1.0)
            self._detect_conf.setValue(0.25)
            self._detect_conf.setSingleStep(0.05)
            fl.addRow("置信度:", self._detect_conf)

            btn_detect = QPushButton("启动推理窗口")
            btn_detect.setStyleSheet("font-size: 14px; padding: 8px;")
            btn_detect.clicked.connect(self._run_detect)
            fl.addRow(btn_detect)

            lbl = QLabel("推理使用独立的 PySide6 窗口（detect.py）\n按 Q 或 ESC 退出推理窗口")
            lbl.setStyleSheet("color: gray;")
            fl.addRow(lbl)

            layout.addWidget(grp)
            layout.addStretch()
            return w

        def _run_detect(self):
            """启动 detect.py 的推理窗口。"""
            weights = self._detect_weights.text().strip()
            conf = self._detect_conf.value()
            source = self._detect_source.currentIndex()

            if not os.path.exists(weights):
                QMessageBox.warning(self, "提示", f"权重文件不存在: {weights}")
                return

            # 构建参数
            argv = ["-m", "ctts.detect", "-w", weights, "--conf", str(conf)]
            if source == 0:
                argv += ["--cam", "0"]
            elif source == 1:
                argv += ["--cam", "1"]
            else:
                path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频 (*.mp4 *.avi)")
                if not path:
                    return
                argv += ["--video", path]

            # 在新进程中启动（detect.py 自带 QApplication，不能嵌套）
            import subprocess
            python = sys.executable
            _env = dict(os.environ)
            _env["PYTHONPATH"] = os.path.join(_PROJECT_ROOT, "src") + os.pathsep + _env.get("PYTHONPATH", "")
            proc = subprocess.Popen(
                [python] + argv,
                cwd=_PROJECT_ROOT,
                env=_env,
                stderr=subprocess.PIPE, stdout=subprocess.PIPE,
            )
            # 短暂等待以捕获立即崩溃
            import time
            time.sleep(1.5)
            ret = proc.poll()
            if ret is not None and ret != 0:
                stderr = proc.stderr.read().decode(errors="replace")
                QMessageBox.critical(self, "推理启动失败",
                    f"detect.py 异常退出 (code={ret})\n\n{stderr[-500:]}")

        # ============================================================
        # ⑦ 预标注复核 Tab
        # ============================================================
        def _build_review_tab(self) -> QWidget:
            w = QWidget()
            layout = QVBoxLayout(w)

            grp = QGroupBox("加载预标注结果")
            fl = QFormLayout(grp)
            row = QHBoxLayout()
            self._rv_project = QSpinBox()
            self._rv_project.setRange(0, 99999999)
            self._rv_project.setPrefix("项目 ")
            row.addWidget(self._rv_project)
            self._rv_version = QComboBox()
            self._rv_version.addItem("全部版本", "")
            row.addWidget(QLabel("版本:"))
            row.addWidget(self._rv_version)
            self._rv_only_unlabeled = QCheckBox("仅未标注")
            self._rv_only_unlabeled.setChecked(True)
            row.addWidget(self._rv_only_unlabeled)
            self._rv_batch_size = QSpinBox()
            self._rv_batch_size.setRange(1, 200)
            self._rv_batch_size.setValue(20)
            self._rv_batch_size.setSuffix(" 条/批")
            row.addWidget(QLabel("每批:"))
            row.addWidget(self._rv_batch_size)
            self._rv_load_btn = QPushButton("加载")
            self._rv_load_btn.setStyleSheet(PRIMARY_BTN_QSS)
            self._rv_load_btn.clicked.connect(self._rv_load)
            row.addWidget(self._rv_load_btn)
            row.addStretch()
            fl.addRow(row)
            self._rv_status = QLabel("未加载：先连接 ②，选择项目后点“加载”")
            _set_status(self._rv_status, self._rv_status.text(), "idle")
            fl.addRow(self._rv_status)
            layout.addWidget(grp)

            grp2 = QGroupBox("批次复核")
            bl = QVBoxLayout(grp2)
            bar = QHBoxLayout()
            self._rv_batch_info = QLabel("—")
            bar.addWidget(self._rv_batch_info)
            bar.addStretch()
            self._rv_prev = QPushButton("上一批")
            self._rv_prev.clicked.connect(lambda: self._rv_page(-1))
            self._rv_next = QPushButton("下一批")
            self._rv_next.clicked.connect(lambda: self._rv_page(1))
            bar.addWidget(self._rv_prev)
            bar.addWidget(self._rv_next)
            bl.addLayout(bar)

            self._rv_table = QTableWidget(0, 5)
            self._rv_table.setHorizontalHeaderLabels(["认可", "任务", "预测框", "版本", "状态"])
            self._rv_table.verticalHeader().setVisible(False)
            self._rv_table.setSelectionBehavior(QTableWidget.SelectRows)
            self._rv_table.setEditTriggers(QTableWidget.NoEditTriggers)
            self._rv_table.horizontalHeader().setStretchLastSection(True)
            self._rv_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            self._rv_table.itemChanged.connect(self._rv_item_changed)
            self._rv_table.currentCellChanged.connect(lambda *_: self._rv_on_row_changed())

            # 右侧图片预览：显示任务图片和预测框，点击框切换是否认可。
            self._rv_canvas = ReviewCanvas()
            self._rv_canvas.boxClicked.connect(self._rv_toggle_box)
            self._rv_preview_info = QLabel("选择任务查看预测框预览")
            self._rv_preview_info.setStyleSheet(MUTED_QSS)
            preview = QWidget()
            pl = QVBoxLayout(preview)
            pl.setContentsMargins(0, 0, 0, 0)
            pl.addWidget(self._rv_canvas, 1)
            pl.addWidget(self._rv_preview_info)
            split = QSplitter(Qt.Horizontal)
            split.addWidget(self._rv_table)
            split.addWidget(preview)
            split.setStretchFactor(0, 3)
            split.setStretchFactor(1, 4)
            split.setSizes([460, 520])
            bl.addWidget(split, 1)

            ops = QHBoxLayout()
            for text, slot in (("全选本批", self._rv_select_page),
                               ("全不选", self._rv_select_none),
                               ("调整选中框…", self._rv_adjust_boxes),
                               ("在 LS 打开", self._rv_open_task)):
                btn = QPushButton(text)
                btn.clicked.connect(slot)
                ops.addWidget(btn)
            ops.addStretch()
            self._rv_accept_btn = QPushButton("认可选中")
            self._rv_accept_btn.setStyleSheet(PRIMARY_BTN_QSS)
            self._rv_accept_btn.clicked.connect(lambda: self._rv_accept())
            ops.addWidget(self._rv_accept_btn)
            self._rv_accept_batch_btn = QPushButton("认可本批全部")
            self._rv_accept_batch_btn.setStyleSheet(PRIMARY_BTN_QSS)
            self._rv_accept_batch_btn.clicked.connect(self._rv_accept_batch)
            ops.addWidget(self._rv_accept_batch_btn)
            bl.addLayout(ops)
            hint = QLabel("勾选任务后点“认可选中”；点“认可本批全部”一键认可当前批次；"
                          "默认认可该任务全部预测框，可先用“调整选中框…”只保留部分框。"
                          "已被标注过的任务不会重复认可，认可后直接从待复核列表移出。")
            hint.setStyleSheet(MUTED_QSS)
            bl.addWidget(hint)
            layout.addWidget(grp2)

            # 状态
            self._rv_items = []
            self._rv_checked: set[int] = set()
            self._rv_box_sel: dict[int, list[int]] = {}
            self._rv_pix_cache: dict[int, QPixmap] = {}
            self._rv_preview_pending = None
            self._rv_batch = 0
            self._rv_loaded_project = -1
            self._rv_busy = False

            self._rv_project.valueChanged.connect(self._rv_project_changed)
            self._rv_batch_size.valueChanged.connect(lambda _: self._rv_render())
            self._rv_version.currentIndexChanged.connect(lambda _: self._rv_invalidate())
            self._rv_only_unlabeled.toggled.connect(lambda _: self._rv_invalidate())
            self._ls_task_project.valueChanged.connect(self._rv_project.setValue)
            return w

        def _rv_client(self):
            return self.client if self.client is not None else None

        def _rv_invalidate(self):
            if self._rv_busy:
                return
            self._rv_loaded_project = -1
            self._rv_items = []
            self._rv_checked.clear()
            self._rv_box_sel.clear()
            self._rv_batch = 0
            self._rv_table.setRowCount(0)
            self._rv_batch_info.setText("—")
            self._rv_pix_cache.clear()
            self._rv_preview_pending = None
            self._rv_img_task = None
            self._rv_canvas.set_message("选择任务查看预测框预览")
            self._rv_preview_info.setText("选择任务查看预测框预览")
            self._rv_accept_btn.setEnabled(False)
            self._rv_accept_batch_btn.setEnabled(False)
            _set_status(self._rv_status, "筛选条件已变化，点击“加载”", "warn")

        def _rv_project_changed(self, pid: int):
            if pid != self._rv_loaded_project:
                self._rv_invalidate()

        def _rv_auto_load(self):
            if self._rv_client() is None or self._rv_busy:
                return
            pid = self._ls_task_project.value()
            if pid <= 0:
                return
            if pid != self._rv_project.value():
                self._rv_project.setValue(pid)
            if pid != self._rv_loaded_project:
                self._rv_load()

        def _rv_load(self):
            self._rv_preview_pending = None
            if self._rv_client() is None:
                QMessageBox.warning(self, "提示", "请先在 ② 标注管理连接 Label Studio。")
                return
            if self._rv_busy:
                return
            pid = self._rv_project.value()
            if pid <= 0:
                QMessageBox.warning(self, "提示", "请填写项目 ID。")
                return
            version = self._rv_version.currentData() or ""
            only_unlabeled = self._rv_only_unlabeled.isChecked()
            self._rv_busy = True
            self._rv_load_btn.setEnabled(False)
            _set_status(self._rv_status, "加载中…", "warn")

            def fn():
                from .review import scan_review
                return scan_review(pid, version=version,
                                   only_unlabeled=only_unlabeled,
                                   client=self.client)

            def done(items):
                self._rv_busy = False
                self._rv_load_btn.setEnabled(True)
                self._rv_loaded_project = pid
                self._rv_items = list(items)
                self._rv_checked.clear()
                self._rv_box_sel.clear()
                self._rv_batch = 0
                versions = sorted({i.version for i in items})
                current = self._rv_version.currentData() or ""
                self._rv_version.blockSignals(True)
                self._rv_version.clear()
                self._rv_version.addItem("全部版本", "")
                for v in versions:
                    self._rv_version.addItem(v, v)
                idx = self._rv_version.findData(current)
                self._rv_version.setCurrentIndex(idx if idx >= 0 else 0)
                self._rv_version.blockSignals(False)
                self._rv_render()
                total = len(items)
                if total:
                    _set_status(self._rv_status,
                                f"已加载 {total} 条待复核（项目 {pid}）", "ok")
                else:
                    _set_status(self._rv_status,
                                f"项目 {pid} 没有待复核的预标注结果", "idle")

            def failed(_e):
                self._rv_busy = False
                self._rv_load_btn.setEnabled(True)
                _set_status(self._rv_status, "加载失败", "err")

            self._run_task(fn, done, "加载复核数据", on_error=failed)

        def _rv_pages(self) -> int:
            size = self._rv_batch_size.value()
            return max(1, (len(self._rv_items) + size - 1) // size)

        def _rv_page(self, delta: int):
            target = self._rv_batch + delta
            if 0 <= target < self._rv_pages():
                self._rv_batch = target
                self._rv_render()

        def _rv_page_items(self):
            size = self._rv_batch_size.value()
            start = self._rv_batch * size
            return self._rv_items[start:start + size]

        def _rv_render(self):
            if not hasattr(self, "_rv_table"):
                return
            pag = self._rv_pages()
            self._rv_batch = min(self._rv_batch, pag - 1)
            items = self._rv_page_items()
            self._rv_table.blockSignals(True)
            self._rv_table.setRowCount(len(items))
            for row, item in enumerate(items):
                chk = QTableWidgetItem()
                chk.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                chk.setCheckState(Qt.Checked if item.task_id in self._rv_checked else Qt.Unchecked)
                chk.setData(Qt.UserRole, item.task_id)
                self._rv_table.setItem(row, 0, chk)
                self._rv_table.setItem(row, 1, QTableWidgetItem(str(item.task_id)))
                self._rv_table.setItem(row, 2, QTableWidgetItem(str(item.boxes)))
                self._rv_table.setItem(row, 3, QTableWidgetItem(item.version))
                state = "已标注" if item.is_labeled else "未标注"
                self._rv_table.setItem(row, 4, QTableWidgetItem(state))
            self._rv_table.blockSignals(False)
            if self._rv_items:
                self._rv_batch_info.setText(
                    f"第 {self._rv_batch + 1}/{pag} 批 · 本批 {len(items)} 条 · "
                    f"已勾选 {len(self._rv_checked)} 条 · 共 {len(self._rv_items)} 条")
            else:
                self._rv_batch_info.setText("—")
            if items:
                self._rv_table.blockSignals(True)
                self._rv_table.selectRow(0)
                self._rv_table.blockSignals(False)
                self._rv_on_row_changed()
            else:
                self._rv_canvas.set_message("本批没有任务")
                self._rv_preview_info.setText("—")
            self._rv_sync_action_buttons()

        def _rv_sync_action_buttons(self):
            """按当前勾选/批次/忙碌状态统一刷新两个认可按钮。"""
            self._rv_accept_btn.setEnabled(bool(self._rv_checked) and not self._rv_busy)
            self._rv_accept_batch_btn.setEnabled(
                bool(self._rv_page_items()) and not self._rv_busy)

        def _rv_item_changed(self, item: QTableWidgetItem):
            if item.column() != 0:
                return
            tid = item.data(Qt.UserRole)
            if tid is None:
                return
            if item.checkState() == Qt.Checked:
                self._rv_checked.add(int(tid))
            else:
                self._rv_checked.discard(int(tid))
            self._rv_sync_action_buttons()
            if self._rv_items:
                self._rv_batch_info.setText(
                    f"第 {self._rv_batch + 1}/{self._rv_pages()} 批 · 本批 "
                    f"{len(self._rv_page_items())} 条 · 已勾选 {len(self._rv_checked)} 条 · "
                    f"共 {len(self._rv_items)} 条")

        def _rv_set_page_checked(self, checked: bool) -> set[int]:
            """勾选/取消勾选当前批次，返回本批任务 id。"""
            ids = {item.task_id for item in self._rv_page_items()}
            if checked:
                self._rv_checked |= ids
            else:
                self._rv_checked -= ids
            self._rv_render()
            return ids

        def _rv_select_page(self):
            self._rv_set_page_checked(True)

        def _rv_select_none(self):
            self._rv_set_page_checked(False)

        def _rv_current_item(self):
            row = self._rv_table.currentRow()
            if row < 0:
                return None
            item = self._rv_table.item(row, 1)
            if item is None:
                return None
            tid = int(item.text())
            for it in self._rv_items:
                if it.task_id == tid:
                    return it
            return None

        def _rv_forget_worker(self, w):
            if w in self._workers:
                self._workers.remove(w)

        def _rv_on_row_changed(self):
            item = self._rv_current_item()
            if item is None:
                if hasattr(self, "_rv_canvas"):
                    self._rv_canvas.set_message("选择任务查看预测框预览")
                return
            if getattr(self, "_rv_preview_pending", None) == item.task_id:
                return
            self._rv_img_task = item.task_id
            pix = self._rv_pix_cache.get(item.task_id)
            if pix is not None:
                self._rv_show_canvas(item, pix)
                return
            if self._rv_client() is None:
                self._rv_canvas.set_message("未连接 Label Studio")
                return
            self._rv_canvas.set_message(f"加载任务 {item.task_id} 的图片…")
            self._rv_preview_info.setText(
                f"任务 {item.task_id} · {item.boxes} 个预测框 · 加载图片…")
            try:
                review_mod = _load_review_module()
            except Exception as e:
                self._rv_canvas.set_message(f"预览模块不可用：{e}")
                return
            self._rv_preview_pending = item.task_id
            w = ReviewImageWorker(item.task_id, self.client, review_mod)
            w.loaded.connect(self._rv_preview_loaded)
            w.failed.connect(self._rv_preview_failed)
            w.finished.connect(lambda w=w: self._rv_forget_worker(w))
            self._workers.append(w)
            w.start()

        def _rv_show_canvas(self, item, pix):
            sel = set(self._rv_box_sel.get(item.task_id, range(item.boxes)))
            self._rv_canvas.set_frame(pix, item.result, sel, item.task_id)
            self._rv_preview_info.setText(
                f"任务 {item.task_id} · {item.boxes} 个预测框 · 已选 {len(sel)} 个"
                "（点击框可切换是否认可）")

        def _rv_preview_loaded(self, task_id: int, data: bytes):
            if getattr(self, "_rv_preview_pending", None) == task_id:
                self._rv_preview_pending = None
            pix = QPixmap()
            if not pix.loadFromData(data):
                if task_id == getattr(self, "_rv_img_task", None):
                    self._rv_canvas.set_message("图片下载成功但无法解码")
                return
            self._rv_pix_cache[task_id] = pix
            while len(self._rv_pix_cache) > 24:
                self._rv_pix_cache.pop(next(iter(self._rv_pix_cache)))
            item = self._rv_current_item()
            if (item is not None and item.task_id == task_id
                    and task_id == getattr(self, "_rv_img_task", None)):
                self._rv_show_canvas(item, pix)

        def _rv_preview_failed(self, task_id: int, msg: str):
            if getattr(self, "_rv_preview_pending", None) == task_id:
                self._rv_preview_pending = None
            if task_id != getattr(self, "_rv_img_task", None):
                return
            self._rv_canvas.set_message(f"图片加载失败：{msg}")
            self._rv_preview_info.setText(
                f"任务 {task_id} · 图片加载失败；可点“在 LS 打开”查看")

        def _rv_toggle_box(self, index: int):
            item = self._rv_current_item()
            if item is None or not (0 <= index < item.boxes):
                return
            sel = set(self._rv_box_sel.get(item.task_id, range(item.boxes)))
            if index in sel:
                sel.discard(index)
            else:
                sel.add(index)
            self._rv_box_sel[item.task_id] = sorted(sel)
            pix = self._rv_pix_cache.get(item.task_id)
            if pix is not None:
                self._rv_show_canvas(item, pix)
            else:
                self._rv_canvas.set_selected(sel)
                self._rv_preview_info.setText(
                    f"任务 {item.task_id} · {item.boxes} 个预测框 · 已选 {len(sel)} 个")

        def _rv_select_task(self, task_id: int):
            for row, item in enumerate(self._rv_page_items()):
                if item.task_id == task_id:
                    self._rv_table.selectRow(row)
                    self._rv_on_row_changed()
                    return

        def _rv_adjust_boxes(self):
            item = self._rv_current_item()
            if item is None:
                QMessageBox.information(self, "提示", "先在表格中选中一条任务。")
                return
            default = set(self._rv_box_sel.get(item.task_id, range(item.boxes)))
            dlg = QDialog(self)
            dlg.setWindowTitle(f"调整框 · 任务 {item.task_id}")
            dlg.resize(640, 380)
            lay = QVBoxLayout(dlg)
            info = QLabel(f"预测版本 {item.version} · 共 {item.boxes} 个框；"
                          "只保留勾选的框。")
            lay.addWidget(info)
            tbl = QTableWidget(len(item.result), 7)
            tbl.setHorizontalHeaderLabels(["认可", "#", "类别", "置信度", "x", "y", "宽×高"])
            tbl.verticalHeader().setVisible(False)
            tbl.setEditTriggers(QTableWidget.NoEditTriggers)
            tbl.horizontalHeader().setStretchLastSection(True)
            for i, res in enumerate(item.result):
                chk = QTableWidgetItem()
                chk.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                chk.setCheckState(Qt.Checked if i in default else Qt.Unchecked)
                chk.setData(Qt.UserRole, i)
                tbl.setItem(i, 0, chk)
                tbl.setItem(i, 1, QTableWidgetItem(str(i)))
                val = res.get("value", {}) if isinstance(res, dict) else {}
                labels = val.get("rectanglelabels") or []
                tbl.setItem(i, 2, QTableWidgetItem("/".join(map(str, labels))))
                score = res.get("score") if isinstance(res, dict) else None
                tbl.setItem(i, 3, QTableWidgetItem("" if score is None else f"{float(score):.2f}"))
                tbl.setItem(i, 4, QTableWidgetItem(f"{val.get('x', 0):.2f}"))
                tbl.setItem(i, 5, QTableWidgetItem(f"{val.get('y', 0):.2f}"))
                tbl.setItem(i, 6, QTableWidgetItem(
                    f"{val.get('width', 0):.2f}% × {val.get('height', 0):.2f}%"))
            lay.addWidget(tbl)
            ops = QHBoxLayout()
            all_btn = QPushButton("全选")
            all_btn.clicked.connect(lambda: [tbl.item(i, 0).setCheckState(Qt.Checked)
                                             for i in range(tbl.rowCount())])
            none_btn = QPushButton("全不选")
            none_btn.clicked.connect(lambda: [tbl.item(i, 0).setCheckState(Qt.Unchecked)
                                              for i in range(tbl.rowCount())])
            ops.addWidget(all_btn)
            ops.addWidget(none_btn)
            ops.addStretch()
            lay.addLayout(ops)
            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            buttons.accepted.connect(dlg.accept)
            buttons.rejected.connect(dlg.reject)
            lay.addWidget(buttons)
            if dlg.exec() != QDialog.Accepted:
                return
            chosen = [i for i in range(tbl.rowCount())
                      if tbl.item(i, 0).checkState() == Qt.Checked]
            self._rv_box_sel[item.task_id] = chosen
            if item.task_id not in self._rv_checked:
                self._rv_checked.add(item.task_id)
            self._rv_render()
            self._rv_select_task(item.task_id)
            if not chosen:
                QMessageBox.information(self, "提示",
                                        "该任务未勾选任何框，认可时会跳过。")

        def _rv_open_task(self):
            if self._rv_client() is None:
                return
            item = self._rv_current_item()
            if item is None:
                checked = [i for i in self._rv_items if i.task_id in self._rv_checked]
                item = checked[0] if checked else None
            if item is None:
                QMessageBox.information(self, "提示", "先选中一条任务。")
                return
            base = str(self.client.base_url).rstrip("/")
            pid = self._rv_project.value()
            webbrowser.open(f"{base}/projects/{pid}/data?task={item.task_id}")

        def _rv_accept_batch(self):
            """勾选当前页全部任务并提交（只认可本批）。"""
            if not self._rv_items:
                QMessageBox.information(self, "提示", "请先加载复核数据。")
                return
            page_ids = self._rv_set_page_checked(True)
            if not page_ids:
                QMessageBox.information(self, "提示", "本批没有任务。")
                return
            self._rv_accept(only=page_ids)

        def _rv_accept(self, only: set[int] | None = None):
            if self._rv_client() is None or self._rv_busy:
                return
            chosen = [i for i in self._rv_items
                      if i.task_id in self._rv_checked
                      and (only is None or i.task_id in only)]
            if not chosen:
                QMessageBox.information(self, "提示", "没有勾选任何任务。")
                return
            if QMessageBox.question(
                    self, "确认认可",
                    f"将把 {len(chosen)} 条任务转为人工标注（只保留已选中的框）。\n"
                    "认可后任务会变成“已标注”，不再出现在待复核列表。是否继续？",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No) != QMessageBox.Yes:
                return
            pid = self._rv_project.value()
            selections = {it.task_id: self._rv_box_sel[it.task_id]
                          for it in chosen if it.task_id in self._rv_box_sel}
            self._rv_busy = True
            self._rv_sync_action_buttons()
            self._rv_load_btn.setEnabled(False)
            _set_status(self._rv_status, f"认可 {len(chosen)} 条…", "warn")

            def fn():
                from .review import accept_items
                return accept_items(pid, chosen, selections=selections,
                                    client=self.client)

            def done(stats):
                self._rv_busy = False
                accepted = set(stats.get("accepted_ids") or [])
                # 已被别处标注过的任务也不会再出现在待复核列表，一并移除。
                gone = accepted | set(stats.get("skipped_ids") or [])
                self._rv_items = [i for i in self._rv_items if i.task_id not in gone]
                for tid in gone:
                    self._rv_checked.discard(tid)
                    self._rv_box_sel.pop(tid, None)
                self._rv_batch = max(0, min(self._rv_batch, self._rv_pages() - 1))
                self._rv_render()
                self._rv_load_btn.setEnabled(True)
                dropped = len(gone) - len(accepted)
                status = (f"认可 {stats.get('accepted', 0)} 条，跳过 "
                          f"{stats.get('skipped', 0)}，失败 {stats.get('failed', 0)}")
                if dropped:
                    status += f"（{dropped} 条已标注，移出列表）"
                _set_status(self._rv_status, status,
                            "ok" if not stats.get("failed") else "warn")
                QMessageBox.information(
                    self, "复核完成",
                    f"认可 {stats.get('accepted', 0)} 条（{stats.get('boxes', 0)} 个框）\n"
                    f"跳过 {stats.get('skipped', 0)}，失败 {stats.get('failed', 0)}")
                if stats.get("errors"):
                    self._log_append("❌ 部分任务认可失败：")
                    for err in stats["errors"]:
                        self._log_append(f"  任务 {err['task_id']}: {err['error']}")
                if not self._rv_items:
                    _set_status(self._rv_status,
                                f"{status}；本次已全部处理完，可点“加载”获取新任务",
                                "ok" if not stats.get("failed") else "warn")

            def failed(_e):
                self._rv_busy = False
                self._rv_load_btn.setEnabled(True)
                self._rv_sync_action_buttons()
                _set_status(self._rv_status, "认可失败", "err")

            self._run_task(fn, done, "认可复核结果", on_error=failed)

        # ============================================================
        # 通用方法
        # ============================================================
        def _show_progress(self, show: bool):
            self._progress.setVisible(show)
            if show:
                self._progress.setRange(0, 0)  # indeterminate
            else:
                self._progress.setRange(0, 0)

        def _set_progress(self, done: int, total: int):
            self._progress.setVisible(True)
            self._progress.setRange(0, max(total, 1))
            self._progress.setValue(done)

        def _log_append(self, msg: str):
            self._log_output.append(msg.rstrip())
            # 自动滚动到底部
            sb = self._log_output.verticalScrollBar()
            sb.setValue(sb.maximum())

        def _clear_log(self):
            self._log_output.clear()

        def _start_task(self, fn, desc="处理中"):
            """启动带日志流的后台任务。

            fn 签名为 fn(log: callable) -> Any，log(msg) 会将消息推送到日志面板。
            """
            self._clear_log()
            self._show_progress(True)
            worker = StreamWorker(fn)
            worker.log_line.connect(lambda msg: self._log_append(msg))
            worker.finished.connect(lambda _: (
                self._log_append(f"✅ {desc}完成"), self._show_progress(False)))
            worker.finished.connect(lambda _: (worker.deleteLater(),
                self._workers.remove(worker) if worker in self._workers else None))
            worker.error.connect(lambda e: (
                self._log_append(f"❌ {desc}错误: {e}"), self._show_progress(False),
                QMessageBox.critical(self, "错误", str(e))))
            worker.error.connect(lambda _: (worker.deleteLater(),
                self._workers.remove(worker) if worker in self._workers else None))
            self._workers.append(worker)
            worker.start()

        def _run_task(self, fn, callback, desc="处理中", *,
                      show_progress=True, quiet_errors=False, on_settle=None,
                      on_error=None):
            """启动后台任务（无日志流，仅进度条 + 结果回调）。

            quiet_errors=True 时不弹错误框、不写日志，适合后台静默刷新。
            on_settle 在成功/失败后都会调用，用于复位按钮等状态。
            """
            if show_progress:
                self._show_progress(True)

            # 包装为兼容 StreamWorker 的 fn_with_log 形式
            def wrapper(log):
                return fn()

            def settle():
                if show_progress:
                    self._show_progress(False)
                if on_settle is not None:
                    on_settle()

            def handle_error(e):
                if not quiet_errors:
                    self._log_append(f"❌ {desc}错误: {e}")
                    QMessageBox.critical(self, "错误", str(e))
                if on_error is not None:
                    on_error(e)
                settle()

            worker = StreamWorker(wrapper)
            worker.finished.connect(callback)
            worker.finished.connect(lambda _: settle())
            worker.finished.connect(lambda _: (worker.deleteLater(),
                self._workers.remove(worker) if worker in self._workers else None))
            worker.error.connect(handle_error)
            worker.error.connect(lambda _: (worker.deleteLater(),
                self._workers.remove(worker) if worker in self._workers else None))
            self._workers.append(worker)
            worker.start()

        def _browse_file(self, target: QLineEdit, filter_str: str):
            path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", filter_str)
            if path:
                target.setText(path)

        def _browse_files(self, target: QLineEdit, filter_str: str):
            paths, _ = QFileDialog.getOpenFileNames(self, "选择文件", "", filter_str)
            if paths:
                target.setText("; ".join(paths))

        def _browse_dir(self, target: QLineEdit):
            path = QFileDialog.getExistingDirectory(self, "选择目录")
            if path:
                target.setText(path)

        # ---- 高级训练参数 ----
        def _build_advanced_group(self, title, defaults_loader):
            grp = QGroupBox(title)
            v = QVBoxLayout(grp)
            hint = QLabel(
                "JSON 对象，覆盖同名训练参数；data / project / name / exist_ok "
                "由程序托管。留空表示使用默认值。")
            hint.setWordWrap(True)
            hint.setStyleSheet("color: #6b7280;")
            v.addWidget(hint)

            edit = QPlainTextEdit()
            edit.setPlaceholderText(
                '例如 {"lr0": 0.001, "cos_lr": false, "mosaic": 0.0}')
            edit.setMaximumHeight(120)
            v.addWidget(edit)

            row = QHBoxLayout()
            btn_defaults = QPushButton("载入默认参数")
            btn_defaults.clicked.connect(
                lambda: self._load_advanced_defaults(edit, defaults_loader))
            btn_clear = QPushButton("清空")
            btn_clear.clicked.connect(edit.clear)
            row.addWidget(btn_defaults)
            row.addWidget(btn_clear)
            row.addStretch()
            v.addLayout(row)
            return grp, edit

        def _load_advanced_defaults(self, edit, loader):
            try:
                data = loader()
            except Exception as e:
                QMessageBox.warning(self, "提示", f"无法读取默认参数: {e}")
                return
            edit.setPlainText(json.dumps(data, ensure_ascii=False, indent=2))

        def _advanced_overrides(self, edit, label):
            from .train_args import OverrideError, parse_json_text
            try:
                overrides, ignored = parse_json_text(edit.toPlainText())
            except OverrideError as e:
                QMessageBox.warning(self, "高级参数错误", f"{label}：{e}")
                return None
            if ignored:
                self._log_append(
                    f"⚠ {label}中的托管项已忽略: {', '.join(ignored)}")
            return overrides

        def _train_advanced_defaults(self):
            from .train_args import MANAGED_KEYS
            from .yolo_train import DEFAULT_TRAIN_ARGS
            return {k: v for k, v in DEFAULT_TRAIN_ARGS.items()
                    if k not in MANAGED_KEYS}

        def _pre_advanced_defaults(self):
            from .prelabel import default_train_args
            from .train_args import MANAGED_KEYS
            defaults = default_train_args(self._pre_device_value())
            return {k: v for k, v in defaults.items() if k not in MANAGED_KEYS}

        def closeEvent(self, event):
            for w in self._workers:
                w.quit()
                w.wait()
            event.accept()


    # ---- 启动 ----
    try:
        app = QApplication(sys.argv)
        if sys.platform.startswith("win"):
            gui_font = "Microsoft YaHei UI"
        elif sys.platform == "darwin":
            gui_font = "PingFang SC"
        else:
            gui_font = "Noto Sans CJK SC"
        app.setFont(QFont(gui_font, 10))
        win = CTTSWindow()
        win.show()
        app.exec()
        return win
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    if "--gui" in sys.argv:
        sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:] if a != "--gui"]
        gui_main()
    else:
        cli_main()


if __name__ == "__main__":
    main()
