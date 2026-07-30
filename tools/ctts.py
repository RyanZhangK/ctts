#!/usr/bin/env python3
"""CTTS — CNN Train Tools Set 统一入口。

整合视觉 CNN 训练全流程工具：数据准备 → 标注管理 → 训练 → 推理。

CLI 示例:
  python ctts.py data build video.mp4 output/ -i 0.5
  python ctts.py data filter images/ --move
  python ctts.py data split dataset/ -v 0.2
  python ctts.py label project list
  python ctts.py label data stats -p 10
  python ctts.py train
  python ctts.py detect --cam 0

GUI 模式:
  python ctts.py --gui
  (需要 PySide6，建议: conda activate ML && python ctts.py --gui)
"""

import argparse
import os
import sys


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
    p_train = sub.add_parser("train", help="YOLO 训练（使用 yolo_train.py 硬编码配置）")

    # ---- detect ----
    p_detect = sub.add_parser("detect", help="YOLO 推理")
    p_detect.add_argument("args", nargs=argparse.REMAINDER, help="传递给 detect 的参数")

    args = parser.parse_args()

    if args.gui:
        gui_main()
        return

    if args.module is None:
        parser.print_help()
        return

    # ---- 数据准备模块 ----
    if args.module == "data":
        extra = getattr(args, "args", [])
        _forward_to_module({
            "build": "dataset_build",
            "split": "split_dataset",
        }, args.action, parser, extra_args=extra)

    # ---- Label Studio 模块 ----
    elif args.module == "label":
        _forward_to_module({
            "project": "label_studio_project",
            "data": "label_studio_data",
            "auth": "label_studio_auth",
        }, args.action, parser, extra_args=getattr(args, "ls_args", []))

    # ---- 训练模块 ----
    elif args.module == "train":
        print("提示: CLI 训练使用 yolo_train.py 硬编码配置。")
        print("      如需自定义参数，请使用 GUI 模式: python ctts.py --gui")
        _forward_yolo_train()

    # ---- 推理模块 ----
    elif args.module == "detect":
        _forward_detect(args.args if hasattr(args, "args") else [])


def _forward_to_module(mapping: dict, action: str, parser, extra_args: list | None = None):
    """将 CLI 请求转发到对应的工具模块。"""
    if action not in mapping:
        parser.print_help()
        return

    mod_name = mapping[action]
    # 重建 sys.argv: ctts.py data build ... → dataset_build.py ...
    sys.argv = [mod_name + ".py"]
    if extra_args:
        # 去掉 REMAINDER 捕获的 "--" 分隔符
        filtered = [a for a in extra_args if a != "--"]
        sys.argv += filtered

    # 动态导入并调用 main
    import importlib
    mod = importlib.import_module(mod_name)
    mod.main()


def _forward_yolo_train():
    """启动 YOLO 训练。"""
    try:
        from yolo_train import main as m
        m()
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("提示: 训练需要 ultralytics 库，请确认在正确的 conda 环境中运行")
        sys.exit(1)


def _forward_detect(extra_args: list):
    """启动 YOLO 推理（detect.py 自带 PySide6 GUI）。"""
    sys.argv = ["detect.py"] + extra_args
    from detect import main as m
    m()


# ---------------------------------------------------------------------------
# PySide6 GUI
# ---------------------------------------------------------------------------

def gui_main():
    """启动 CTTS 图形界面。"""
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)

    try:
        from PySide6.QtCore import Qt, QThread, Signal
        from PySide6.QtGui import QFont
        from PySide6.QtWidgets import (
            QApplication, QWidget, QVBoxLayout, QHBoxLayout,
            QTabWidget, QPushButton, QLabel, QLineEdit,
            QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
            QGroupBox, QFormLayout, QSpinBox, QProgressBar,
            QMainWindow, QFileDialog, QTextEdit, QComboBox,
            QCheckBox, QDoubleSpinBox, QSlider,
        )
    except ModuleNotFoundError:
        print("❌ PySide6 未安装。请使用 conda ML 环境：")
        print("   conda activate ML && python ctts.py --gui")
        sys.exit(1)

    from label_studio_auth import get_client
    from label_studio_project import list_projects, get_project, create_project, delete_project
    from label_studio_data import list_tasks, delete_tasks_batch, task_stats, import_tasks_from_images

    # ---- StreamWorker ----
    class StreamWorker(QThread):
        log_line = Signal(str)
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


    # ---- 主窗口 ----
    class CTTSWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("CTTS — CNN Train Tools Set")
            self.resize(960, 700)
            self.client = None
            self._workers = []

            try:
                self.client = get_client()
            except Exception:
                pass

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
            tabs.addTab(self._build_data_tab(), "① 数据准备")
            tabs.addTab(self._build_label_tab(), "② 标注管理")
            tabs.addTab(self._build_split_tab(), "③ 数据集划分")
            tabs.addTab(self._build_train_tab(), "④ 训练")
            tabs.addTab(self._build_detect_tab(), "⑤ 推理")
            layout.addWidget(tabs)

            # 共享日志输出
            self._log_output = QTextEdit()
            self._log_output.setReadOnly(True)
            self._log_output.setMaximumHeight(150)
            self._log_output.setPlaceholderText("操作日志将显示在这里...")
            layout.addWidget(self._log_output)

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
            from dataset_build import collect_videos, extract_frames
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
            from split_dataset import split_dataset
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
            from split_dataset import split_dataset
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

            if self.client is None:
                layout.addWidget(QLabel("⚠️ Label Studio 未连接，仅 CLI 模式可用"))
                return w

            # 项目表格
            grp_proj = QGroupBox("项目列表")
            pl = QVBoxLayout(grp_proj)
            btn_row = QHBoxLayout()
            btn_refresh = QPushButton("刷新")
            btn_refresh.clicked.connect(self._ls_refresh_projects)
            btn_row.addWidget(btn_refresh)
            btn_create = QPushButton("新建项目")
            btn_create.clicked.connect(self._ls_create_project)
            btn_row.addWidget(btn_create)
            btn_del = QPushButton("删除项目")
            btn_del.setStyleSheet("color: red;")
            btn_del.clicked.connect(self._ls_delete_project)
            btn_row.addWidget(btn_del)
            btn_row.addStretch()
            pl.addLayout(btn_row)

            self._ls_project_table = QTableWidget()
            self._ls_project_table.setColumnCount(3)
            self._ls_project_table.setHorizontalHeaderLabels(["ID", "标题", "任务数"])
            self._ls_project_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
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
            btn_upload = QPushButton("上传")
            btn_upload.setStyleSheet("font-weight: bold;")
            btn_upload.clicked.connect(self._ls_upload_images)
            up_row.addWidget(btn_upload)
            up_row.addStretch()
            ul.addLayout(up_row)
            self._ls_upload_info = QLabel("未选择文件")
            self._ls_upload_info.setStyleSheet("color: gray;")
            ul.addWidget(self._ls_upload_info)
            layout.addWidget(grp_upload)

            # 首次加载
            self._ls_refresh_projects()
            return w

        def _ls_refresh_projects(self):
            self._run_task(
                lambda: list_projects(self.client),
                lambda ps: self._ls_populate_projects(ps),
                "加载项目",
            )

        def _ls_populate_projects(self, projects):
            self._ls_project_table.setRowCount(len(projects))
            for i, p in enumerate(projects):
                self._ls_project_table.setItem(i, 0, QTableWidgetItem(str(p["id"])))
                self._ls_project_table.setItem(i, 1, QTableWidgetItem(p.get("title", "")))
                self._ls_project_table.setItem(i, 2, QTableWidgetItem(str(p.get("task_number", 0))))

        def _ls_on_project_selected(self):
            """项目列表选中行时，自动同步项目 ID 到任务管理和上传区域。"""
            rows = self._ls_project_table.selectionModel().selectedRows()
            if not rows:
                return
            pid = int(self._ls_project_table.item(rows[0].row(), 0).text())
            self._ls_task_project.setValue(pid)
            self._ls_upload_project.setValue(pid)

        def _ls_create_project(self):
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
            pid = self._ls_task_project.value()
            self._run_task(
                lambda: task_stats(pid, self.client),
                lambda s: QMessageBox.information(self, f"项目 {pid} 统计",
                    f"总任务: {s['total']}\n已标注: {s['labeled']} ({s['labeled']/max(s['total'],1)*100:.1f}%)\n未标注: {s['unlabeled']}"),
                "统计",
            )

        def _ls_delete_tasks(self):
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
                self._ls_upload_info.setText(f"已选择 {len(paths)} 张图片")
                self._ls_upload_info.setStyleSheet("color: green;")

        def _ls_select_folder(self):
            """选择文件夹，递归扫描其中所有图片。"""
            folder = QFileDialog.getExistingDirectory(self, "选择图片所在文件夹")
            if folder:
                paths = self._scan_images(folder)
                self._ls_image_paths = paths
                if paths:
                    self._ls_upload_info.setText(f"已选择 {len(paths)} 张图片（来自 {folder}）")
                    self._ls_upload_info.setStyleSheet("color: green;")
                else:
                    self._ls_upload_info.setText(f"文件夹内无图片: {folder}")
                    self._ls_upload_info.setStyleSheet("color: orange;")

        def _ls_upload_images(self):
            """批量上传图片到 Label Studio 项目。"""
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

            def do_upload():
                return import_tasks_from_images(pid, paths, client=self.client)

            def on_done(result):
                n = result.get("task_count", 0)
                errs = result.get("errors", [])
                if errs:
                    detail = "\n".join(errs[:10])
                    more = f"\n... 还有 {len(errs) - 10} 个错误" if len(errs) > 10 else ""
                    QMessageBox.warning(self, "上传完成（有错误）",
                        f"成功 {n} 个，失败 {len(errs)} 个\n\n失败详情:\n{detail}{more}")
                else:
                    QMessageBox.information(self, "上传完成",
                        f"已导入 {n} 个任务到项目 {pid}")
                self._ls_image_paths = []
                self._ls_upload_info.setText("未选择文件")
                self._ls_upload_info.setStyleSheet("color: gray;")
                # 刷新任务列表
                if self._ls_task_project.value() == pid:
                    self._ls_load_tasks()

            self._run_task(do_upload, on_done, "上传图片")

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

            self._start_task(
                lambda log: self._do_train(log, weights, data, epochs, batch, imgsz, device),
                "YOLO 训练",
            )

        def _do_train(self, log, weights, data, epochs, batch, imgsz, device):
            import io
            from ultralytics import YOLO
            log(f"🚀 启动训练: {weights}\n")
            model = YOLO(weights)
            buf = io.StringIO()
            old = sys.stdout
            sys.stdout = buf
            try:
                model.train(data=data, epochs=epochs, imgsz=imgsz,
                            batch=batch, device=device, amp=True)
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
            argv = ["detect.py", "-w", weights, "--conf", str(conf)]
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
            proc = subprocess.Popen(
                [python] + argv,
                cwd=os.path.dirname(os.path.abspath(__file__)),
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
        # 通用方法
        # ============================================================
        def _show_progress(self, show: bool):
            self._progress.setVisible(show)
            if show:
                self._progress.setRange(0, 0)  # indeterminate

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

        def _run_task(self, fn, callback, desc="处理中"):
            """启动后台任务（无日志流，仅进度条 + 结果回调）。"""
            self._show_progress(True)
            # 包装为兼容 StreamWorker 的 fn_with_log 形式
            def wrapper(log):
                return fn()
            worker = StreamWorker(wrapper)
            worker.finished.connect(callback)
            worker.finished.connect(lambda _: self._show_progress(False))
            worker.finished.connect(lambda _: (worker.deleteLater(),
                self._workers.remove(worker) if worker in self._workers else None))
            worker.error.connect(lambda e: (
                self._log_append(f"❌ {desc}错误: {e}"), self._show_progress(False),
                QMessageBox.critical(self, "错误", str(e))))
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

        def closeEvent(self, event):
            for w in self._workers:
                w.quit()
                w.wait()
            event.accept()


    # ---- 启动 ----
    app = QApplication(sys.argv)
    app.setFont(QFont("Noto Sans CJK SC", 10))
    win = CTTSWindow()
    win.show()
    app.exec()


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
