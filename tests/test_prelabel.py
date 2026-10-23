#!/usr/bin/env python3
"""prelabel.py 纯逻辑单元测试（无 Label Studio / 无 GPU / 无网络）。

运行:
  PYTHONPATH=src python3 -m pytest tests/test_prelabel.py -v

未安装 httpx / label-studio-sdk 时整文件跳过。
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# 集成层依赖：未安装时整文件跳过（CI 的 [ci] extras 会装上）
pytest.importorskip("httpx")
pytest.importorskip("label_studio_sdk")

from ctts import prelabel as P  # noqa: E402


CFG_XML_STB = (
    '<View>\n'
    '  <Image name="image" value="$image"/>\n'
    '  <RectangleLabels name="label" toName="image">\n'
    '    <Label value="stb" background="#FF0000"/>\n'
    '  </RectangleLabels>\n'
    '</View>'
)

CFG_XML = (
    '<View>\n'
    '  <Image name="image" value="$image"/>\n'
    '  <RectangleLabels name="label" toName="image">\n'
    '    <Label value="stb" background="#FF0000"/>\n'
    '    <Label value="person" background="#00FF00"/>\n'
    '  </RectangleLabels>\n'
    '</View>'
)


def make_annotation(boxes, updated="2026-01-01T00:00:00Z", cancelled=False):
    """boxes: [(x, y, w, h, label)] 百分比坐标。"""
    result = []
    for x, y, w, h, label in boxes:
        result.append({
            "type": "rectanglelabels",
            "from_name": "label",
            "to_name": "image",
            "value": {"x": x, "y": y, "width": w, "height": h,
                      "rotation": 0, "rectanglelabels": [label]},
        })
    return {"id": 1, "result": result, "updated_at": updated,
            "completed_by": 1, "was_cancelled": cancelled}


def make_task(i, boxes, updated="2026-01-01T00:00:00Z"):
    return {
        "id": i,
        "data": {"image": f"/data/upload/{i}.jpg"},
        "is_labeled": True,
        "annotations": [make_annotation(boxes, updated)],
    }


class FakeProjects:
    def __init__(self, project):
        self.project = project

    def get(self, _project_id):
        return self.project


class FakeTasks:
    def __init__(self, tasks):
        self._tasks = tasks

    def list(self, **_kwargs):
        return list(self._tasks)


class FakeClient:
    base_url = "http://127.0.0.1:8080"

    def __init__(self, project, tasks):
        self.projects = FakeProjects(project)
        self.tasks = FakeTasks(tasks)


def make_project(label_config=CFG_XML, total=0):
    return {"id": 1, "label_config": label_config, "task_number": total}


class TestParseLabelConfig(unittest.TestCase):
    def test_labels_and_fields(self):
        cfg = P.parse_label_config(CFG_XML)
        self.assertEqual(cfg.labels, ["stb", "person"])
        self.assertEqual(cfg.from_name, "label")
        self.assertEqual(cfg.to_name, "image")
        self.assertEqual(cfg.image_key, "image")

    def test_missing_rectanglelabels(self):
        with self.assertRaises(ValueError):
            P.parse_label_config('<View><Image name="image" value="$image"/></View>')

    def test_broken_xml(self):
        with self.assertRaises(ValueError):
            P.parse_label_config("<View>")


class TestAnnotations(unittest.TestCase):
    def setUp(self):
        self.cfg = P.LabelConfig(labels=["stb", "person"])

    def test_percent_passthrough(self):
        parsed = P.parse_annotation(
            make_annotation([(10, 20, 30, 40, "stb")]), self.cfg)
        self.assertEqual(parsed.invalid, 0)
        self.assertEqual(parsed.lines, ["0 0.250000 0.400000 0.300000 0.400000"])
        self.assertEqual(parsed.per_class["stb"], 1)

    def test_invalid_and_unknown_boxes(self):
        parsed = P.parse_annotation(make_annotation([
            (10, 10, 0, 10, "stb"),       # 宽度 0
            (120, 10, 10, 10, "stb"),     # 越界
            (10, 10, 10, 10, "cat"),      # 未配置类别
        ]), self.cfg)
        self.assertEqual(parsed.lines, [])
        self.assertEqual(parsed.invalid, 3)
        self.assertEqual(parsed.unknown, ["cat"])

    def test_edge_box_allowed(self):
        parsed = P.parse_annotation(
            make_annotation([(70, 60, 30, 40, "stb")]), self.cfg)
        self.assertEqual(parsed.invalid, 0)
        self.assertEqual(parsed.lines[0], "0 0.850000 0.800000 0.300000 0.400000")

    def test_small_ratio(self):
        parsed = P.parse_annotation(
            make_annotation([(10, 10, 5, 5, "stb")]), self.cfg)
        self.assertEqual(parsed.small, 1)  # 0.25% < 0.5%

    def test_latest_annotation_ignores_cancelled(self):
        task = {
            "annotations": [
                make_annotation([(1, 1, 5, 5, "stb")], "2026-01-01T00:00:00Z"),
                make_annotation([], "2026-02-01T00:00:00Z"),
                make_annotation([(1, 1, 5, 5, "stb")], "2026-03-01T00:00:00Z",
                                cancelled=True),
            ]
        }
        latest = P.latest_annotation(task)
        self.assertEqual(latest["updated_at"], "2026-02-01T00:00:00Z")


class TestAssessment(unittest.TestCase):
    def setUp(self):
        self.cfg = P.LabelConfig(labels=["stb"])

    def _tasks(self, n, per_task=2):
        return [make_task(i, [(10, 10, 20, 20, "stb")] * per_task)
                for i in range(n)]

    def test_ready(self):
        a = P.assess_tasks(self._tasks(50), self.cfg)
        self.assertEqual(a.verdict, "ready")
        self.assertEqual(a.labeled_tasks, 50)
        self.assertEqual(a.instances, 100)

    def test_marginal(self):
        a = P.assess_tasks(self._tasks(40), self.cfg)
        self.assertEqual(a.verdict, "marginal")
        self.assertIn("再标 10 张", a.suggestion)

    def test_insufficient(self):
        a = P.assess_tasks(self._tasks(20), self.cfg)
        self.assertEqual(a.verdict, "insufficient")
        self.assertIn("还差 30 张", a.suggestion)

    def test_missing_class_is_marginal(self):
        cfg = P.LabelConfig(labels=["stb", "person"])
        a = P.assess_tasks(self._tasks(50), cfg)
        self.assertEqual(a.per_class["person"], 0)
        self.assertEqual(a.verdict, "marginal")

    def test_negative_tasks(self):
        tasks = self._tasks(50) + [make_task(999, [])]
        a = P.assess_tasks(tasks, self.cfg)
        self.assertEqual(a.negative_tasks, 1)
        self.assertEqual(a.annotated_tasks, 51)

    def test_assess_project_with_fake_client(self):
        tasks = self._tasks(50)
        client = FakeClient(make_project(label_config=CFG_XML_STB, total=60), tasks)
        a = P.assess_project(1, client=client)
        self.assertEqual(a.project, 1)
        self.assertEqual(a.total_tasks, 60)
        self.assertEqual(a.verdict, "ready")


class TestVersioning(unittest.TestCase):
    def test_next_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for version in ("v1", "v3"):
                d = root / f"run_{version}" / "model"
                d.mkdir(parents=True)
                (d / "manifest.json").write_text(
                    json.dumps({"version": version, "tag": f"x:{version}@480"}))
            self.assertEqual(P.next_version(root), "v4")
            self.assertEqual(P.make_tag("v2", 480), "ctts-prelabel:v2@480")

    def test_next_version_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(P.next_version(tmp), "v1")


class TestClassMapping(unittest.TestCase):
    def test_exact_and_casefold(self):
        mapping = P.resolve_class_mapping({0: "STB", 1: "person"}, ["stb", "person"])
        self.assertEqual(mapping, {0: "stb", 1: "person"})

    def test_override_and_drop(self):
        mapping = P.resolve_class_mapping(
            {0: "car", 1: "cat"}, ["stb"], class_map={"car": "stb"})
        self.assertEqual(mapping, {0: "stb"})


class TestPredictionItems(unittest.TestCase):
    def test_coordinates_and_min_size(self):
        cfg = P.LabelConfig(labels=["stb"])
        boxes = SimpleNamespace(
            xyxy=np.array([[960, 540, 1440, 1080], [10, 10, 11, 11]], dtype=float),
            conf=np.array([0.9, 0.8], dtype=float),
            cls=np.array([0, 0], dtype=int),
        )
        result = SimpleNamespace(orig_shape=(1080, 1920), boxes=boxes)
        items, scores = P._prediction_items(result, cfg, {0: "stb"}, 0.25)
        self.assertEqual(len(items), 1)  # 1px 框被丢弃
        item = items[0]
        self.assertAlmostEqual(item["value"]["x"], 50.0)
        self.assertAlmostEqual(item["value"]["y"], 50.0)
        self.assertAlmostEqual(item["value"]["width"], 25.0)
        self.assertAlmostEqual(item["value"]["height"], 50.0)
        self.assertEqual(item["original_width"], 1920)
        self.assertEqual(item["original_height"], 1080)
        self.assertEqual(item["value"]["rectanglelabels"], ["stb"])
        self.assertAlmostEqual(scores[0], 0.9)

    def test_empty_boxes(self):
        cfg = P.LabelConfig(labels=["stb"])
        result = SimpleNamespace(
            orig_shape=(100, 100),
            boxes=SimpleNamespace(xyxy=np.empty((0, 4)), conf=np.empty(0),
                                  cls=np.empty(0)),
        )
        items, _ = P._prediction_items(result, cfg, {0: "stb"}, 0.25)
        self.assertEqual(items, [])


class TestBuildDataset(unittest.TestCase):
    def test_build_with_local_images(self):
        import cv2

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tasks = []
            for i in range(14):
                img = tmp_path / f"{i}.png"
                cv2.imwrite(str(img), np.zeros((40, 40, 3), dtype=np.uint8))
                tasks.append({
                    "id": i,
                    "is_labeled": True,
                    "data": {"image": str(img)},
                    "annotations": [make_annotation([(10, 10, 20, 20, "stb")])],
                })

            client = FakeClient(make_project(total=14), tasks)
            out = tmp_path / "run" / "dataset"
            info = P.build_micro_dataset(1, out_dir=str(out), client=client)

            self.assertEqual(info.classes, ["stb", "person"])
            self.assertEqual(info.labeled_tasks, 14)
            self.assertGreaterEqual(info.train, 10)
            self.assertGreaterEqual(info.val, 2)
            self.assertTrue((out / "data.yaml").is_file())
            self.assertTrue((out / "classes.txt").is_file())
            self.assertTrue((out / "labels" / "0.txt").is_file())
            content = (out / "labels" / "0.txt").read_text().strip()
            self.assertTrue(content.startswith("0 "))


class PipelineClientTest(unittest.TestCase):
    """③ 预标注必须复用 ② 的连接，而不是回落到本地默认 client。"""

    def test_run_pipeline_forwards_client(self):
        import unittest.mock as mock
        with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
            seen = {}

            def fake_prelabel(project_id, **kwargs):
                seen.update(kwargs)
                return SimpleNamespace(to_dict=lambda: {"ok": True})

            with mock.patch.object(P, "prelabel_project", fake_prelabel):
                P.run_pipeline(1, reuse_model=f.name, skip_train=True, client="REMOTE")
        self.assertEqual(seen["client"], "REMOTE")

    def test_run_pipeline_default_client_is_none(self):
        import unittest.mock as mock
        with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
            seen = {}

            def fake_prelabel(project_id, **kwargs):
                seen.update(kwargs)
                return SimpleNamespace(to_dict=lambda: {"ok": True})

            with mock.patch.object(P, "prelabel_project", fake_prelabel):
                P.run_pipeline(1, reuse_model=f.name, skip_train=True)
        self.assertIsNone(seen.get("client"))


class TrainPathsTest(unittest.TestCase):
    def test_relative_weights_under_run_dir(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d).resolve()
            best = run / "train" / "exp" / "weights" / "best.pt"
            best.parent.mkdir(parents=True)
            best.write_text("x")
            result = P.TrainResult(run_dir=run, best_pt=str(best),
                                   metrics={}, args={})
            self.assertEqual(P._relative_weights(result),
                             "train/exp/weights/best.pt")

    def test_relative_weights_outside_run_dir_falls_back(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d).resolve()
            outside = run.parent / "elsewhere" / "best.pt"
            outside.parent.mkdir(parents=True, exist_ok=True)
            outside.write_text("x")
            result = P.TrainResult(run_dir=run, best_pt=str(outside),
                                   metrics={}, args={})
            self.assertEqual(P._relative_weights(result), str(outside))


class TrainArgsTest(unittest.TestCase):
    """高级训练参数：默认值、合并、托管项与 JSON/--set 校验。"""

    def test_default_train_args_shape(self):
        args = P.default_train_args("cpu")
        self.assertEqual(args["batch"], 8)
        self.assertEqual(args["workers"], 2)
        self.assertFalse(args["amp"])
        self.assertEqual(args["imgsz"], 480)

    def test_build_train_args_merges_and_ignores_managed(self):
        base = {"epochs": 10, "device": "cpu"}
        args, ignored = P.build_train_args(
            base, {"lr0": 0.001, "epochs": 20, "data": "x"}, "cpu")
        self.assertEqual(args["epochs"], 20)
        self.assertEqual(args["lr0"], 0.001)
        self.assertEqual(args["device"], "cpu")
        self.assertEqual(ignored, ["data"])

    def test_parse_json_text_validation(self):
        from ctts.train_args import OverrideError, parse_json_text
        with self.assertRaises(OverrideError):
            parse_json_text("[1, 2]")
        with self.assertRaises(OverrideError):
            parse_json_text('{"augment": {"h": 1}}')

        parsed, ignored = parse_json_text(
            '{"lr0": 0.01, "mosaic": 0.0, "name": "x"}')
        self.assertEqual(parsed, {"lr0": 0.01, "mosaic": 0.0})
        self.assertEqual(ignored, ["name"])

    def test_parse_set_values(self):
        from ctts.train_args import parse_set
        args, ignored = parse_set(
            ["lr0=0.002", "cos_lr=false", "degrees=10", "project=runs/x"])
        self.assertAlmostEqual(args["lr0"], 0.002)
        self.assertIs(args["cos_lr"], False)
        self.assertEqual(args["degrees"], 10)
        self.assertNotIn("project", args)
        self.assertEqual(ignored, ["project"])


if __name__ == "__main__":
    unittest.main()
