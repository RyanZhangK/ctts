"""预标注复核逻辑的单元测试（不依赖真实 Label Studio）。"""
from __future__ import annotations

import unittest

import pytest

# 集成层依赖：未安装时整文件跳过（CI 的 [ci] extras 会装上）
pytest.importorskip("httpx")
pytest.importorskip("label_studio_sdk")

from ctts import review  # noqa: E402


def _box(label="object", score=0.9, x=10.0):
    return {
        "from_name": "label", "to_name": "image", "type": "rectanglelabels",
        "value": {"x": x, "y": 20.0, "width": 10.0, "height": 10.0,
                  "rotation": 0.0, "rectanglelabels": [label]},
        "score": score, "original_width": 640, "original_height": 480,
    }


class _Pred:
    def __init__(self, pid, task, version="ctts-prelabel:v1@320", boxes=1):
        self.id = pid
        self.task = task
        self.model_version = version
        self.result = [_box(x=10.0 * i) for i in range(boxes)]


class _Task:
    def __init__(self, tid, labeled=False, annotations=0):
        self.id = tid
        self.is_labeled = labeled
        self.total_annotations = annotations


class _PredsAPI:
    def __init__(self, preds):
        self.preds = preds

    def list(self, project):
        return list(self.preds)


class _TasksAPI:
    def __init__(self, tasks):
        self.tasks = {t.id: t for t in tasks}

    def list(self, **kwargs):
        return list(self.tasks.values())

    def get(self, tid):
        return self.tasks[tid]


class _AnnotationsAPI:
    def __init__(self, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)

    def create(self, task, **kwargs):
        if task in self.fail_on:
            raise RuntimeError("boom")
        self.calls.append({"task": task, **kwargs})
        return object()


class _Client:
    def __init__(self, preds, tasks, fail_on=()):
        self.predictions = _PredsAPI(preds)
        self.tasks = _TasksAPI(tasks)
        self.annotations = _AnnotationsAPI(fail_on)


class ScanReviewTest(unittest.TestCase):
    def test_filters_and_latest_version(self):
        client = _Client(
            preds=[_Pred(1, 1, boxes=2), _Pred(2, 1, boxes=3),
                   _Pred(3, 2, version="other@1"),
                   _Pred(4, 3, boxes=1)],
            tasks=[_Task(1), _Task(2), _Task(3, labeled=True),
                   _Task(4)],
        )
        items = review.scan_review(1, client=client)
        self.assertEqual([i.task_id for i in items], [1])
        self.assertEqual(items[0].prediction_id, 2)
        self.assertEqual(items[0].boxes, 3)

    def test_limit_and_version_filter(self):
        client = _Client(
            preds=[_Pred(1, 1, version="ctts-prelabel:v1@320"),
                   _Pred(2, 2, version="ctts-prelabel:v2@480")],
            tasks=[_Task(1), _Task(2)],
        )
        items = review.scan_review(1, client=client,
                                   version="ctts-prelabel:v2@480")
        self.assertEqual([i.task_id for i in items], [2])
        limited = review.scan_review(1, client=client, limit=1)
        self.assertEqual(len(limited), 1)
        self.assertEqual(limited[0].task_id, 1)


class AcceptReviewTest(unittest.TestCase):
    def test_accept_partial_boxes_and_strip_score(self):
        preds = [_Pred(1, 1, boxes=3)]
        client = _Client(preds, [_Task(1)])
        items = review.scan_review(1, client=client)
        stats = review.accept_items(1, items, selections={1: [0, 2]},
                                    client=client)
        self.assertEqual(stats["accepted"], 1)
        self.assertEqual(stats["boxes"], 2)
        call = client.annotations.calls[0]
        self.assertEqual(call["task"], 1)
        self.assertEqual(len(call["result"]), 2)
        self.assertNotIn("score", call["result"][0])
        self.assertEqual(call["parent_prediction"], 1)
        self.assertEqual(call["last_action"], "prediction")

    def test_skip_labeled_and_recheck(self):
        client = _Client([_Pred(1, 1, boxes=1), _Pred(2, 2, boxes=1)],
                         [_Task(1, labeled=True), _Task(2)])
        items = review.scan_review(1, client=client, only_unlabeled=False)
        stats = review.accept_items(1, items, client=client)
        self.assertEqual(stats["skipped"], 1)
        self.assertEqual(stats["accepted"], 1)
        self.assertEqual(stats["accepted_ids"], [2])
        # 已标注的任务同时出现在 skipped_ids 里，调用方据此移出待复核列表。
        self.assertEqual(stats["skipped_ids"], [1])

    def test_empty_selection_is_not_skipped_id(self):
        """用户主动不选框 → 跳过但保留在待复核列表，不返回 skipped_ids。"""
        client = _Client([_Pred(1, 1, boxes=2)], [_Task(1)])
        items = review.scan_review(1, client=client)
        stats = review.accept_items(1, items, selections={1: []}, client=client)
        self.assertEqual(stats["skipped"], 1)
        self.assertEqual(stats["accepted"], 0)
        self.assertEqual(stats["skipped_ids"], [])

    def test_error_is_collected(self):
        client = _Client([_Pred(1, 1)], [_Task(1)], fail_on=[1])
        items = review.scan_review(1, client=client)
        stats = review.accept_items(1, items, client=client)
        self.assertEqual(stats["failed"], 1)
        self.assertEqual(stats["accepted"], 0)
        self.assertEqual(stats["errors"][0]["task_id"], 1)


if __name__ == "__main__":
    unittest.main()
