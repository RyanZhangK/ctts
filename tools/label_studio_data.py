#!/usr/bin/env python3
"""Label Studio 任务数据管理 —— 增删查、统计（支持本机 / 远程实例）。

示例:
  # 本机
  python label_studio_data.py list -p 10
  python label_studio_data.py stats -p 10

  # 远程
  python label_studio_data.py --base-url http://192.168.1.10:8080 --token <TOKEN> stats -p 10
"""

import argparse
import json
import os
from typing import Callable
import sys

from label_studio_auth import (
    add_connection_args,
    client_from_args,
    get_client,
    is_local_url,
    to_dict,
)


# ---------------------------------------------------------------------------
# 任务 CRUD
# ---------------------------------------------------------------------------

def list_tasks(
    project_id: int,
    page: int = 1,
    page_size: int = 50,
    client=None,
) -> list[dict]:
    """列出指定项目的任务（分页）。"""
    c = client or get_client()
    tasks = c.tasks.list(
        project=project_id,
        page=page,
        page_size=page_size,
        fields="task_only",
    )
    return [to_dict(t) for t in tasks]


def get_task(task_id: int, client=None) -> dict:
    """获取单个任务详情。"""
    c = client or get_client()
    return to_dict(c.tasks.get(str(task_id)))


def delete_task(task_id: int, client=None) -> dict:
    """删除单个任务。"""
    c = client or get_client()
    c.tasks.delete(str(task_id))
    return {"deleted": 1}


def delete_tasks_batch(project_id: int, task_ids: list[int], client=None) -> dict:
    """批量删除任务。

    SDK 未提供“按 ID 批量删除”的封装（仅提供清空整个项目），
    因此逐条调用 ``tasks.delete``，并汇总失败项。
    """
    c = client or get_client()
    deleted = 0
    errors: list[str] = []
    for task_id in task_ids:
        try:
            c.tasks.delete(str(task_id))
            deleted += 1
        except Exception as e:
            errors.append(f"{task_id}: {e}")

    result = {"deleted": deleted}
    if errors:
        result["errors"] = errors
    return result


# ---------------------------------------------------------------------------
# 批量导入
# ---------------------------------------------------------------------------

def import_tasks_from_images(
    project_id: int,
    image_paths: list[str],
    client=None,
    progress: Callable[[int, int], None] | None = None,
    log: Callable[[str], None] | None = None,
) -> dict:
    """批量导入本地图片作为标注任务。

    通过 SDK 客户端上传每张图片并创建任务。
    图片从调用方机器读取并上传到目标实例（本机或远程均可）。

    progress(done, total) 每处理完一张回调一次，便于 GUI 显示进度；
    log(msg) 用于逐条报告失败项。
    """
    c = client or get_client()

    total = len(image_paths)
    errors = []

    for i, p in enumerate(image_paths, 1):
        filename = os.path.basename(p)
        try:
            c.upload_file(project_id, p)
        except Exception as e:
            message = f"[{i}/{total}] {filename}: {e}"
            errors.append(message)
            if log:
                log(f"上传失败 {message}")
        if progress:
            try:
                progress(i, total)
            except Exception:
                pass

    result = {"task_count": total - len(errors)}
    if errors:
        result["errors"] = errors
    return result


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------

def task_stats(project_id: int, client=None) -> dict:
    """获取项目任务统计。"""
    c = client or get_client()
    project = c.projects.get(project_id)
    total = project.task_number or 0
    annotated = project.num_tasks_with_annotations or 0

    return {
        "total": total,
        "labeled": annotated,
        "unlabeled": total - annotated,
        "total_annotations": project.total_annotations_number or 0,
        "total_predictions": project.total_predictions_number or 0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    conn = argparse.ArgumentParser(add_help=False)
    add_connection_args(conn)

    parser = argparse.ArgumentParser(
        description="Label Studio 任务数据管理（支持本机 / 远程）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[conn],
    )
    sub = parser.add_subparsers(dest="command", help="操作")

    p_list = sub.add_parser("list", help="列出任务")
    p_list.add_argument("-p", "--project", type=int, required=True, help="项目 ID")
    p_list.add_argument("--page", type=int, default=1, help="页码（默认: 1）")
    p_list.add_argument("--page-size", type=int, default=20, help="每页数量（默认: 20）")

    p_get = sub.add_parser("get", help="查看任务详情")
    p_get.add_argument("task_id", type=int, help="任务 ID")

    p_del = sub.add_parser("delete", help="批量删除任务")
    p_del.add_argument("-p", "--project", type=int, required=True, help="项目 ID")
    p_del.add_argument("--ids", required=True, help="任务 ID 列表，逗号分隔（如: 1,2,3）")
    p_del.add_argument("--yes", action="store_true", help="跳过确认")

    p_stats = sub.add_parser("stats", help="任务统计")
    p_stats.add_argument("-p", "--project", type=int, required=True, help="项目 ID")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    try:
        client = client_from_args(args)
        if not is_local_url(client.base_url):
            print(f"→ 远程实例: {client.base_url}")

        if args.command == "list":
            tasks = list_tasks(args.project, args.page, args.page_size, client)
            if not tasks:
                print("(无任务)")
                return
            print(f"{'ID':<8} {'标注':<6} {'数据字段':<50}")
            print("-" * 80)
            for t in tasks:
                labeled = "✅" if t.get("is_labeled") else "⬜"
                data_preview = str(t.get("data", {}))
                if len(data_preview) > 48:
                    data_preview = data_preview[:48] + "..."
                print(f"{t['id']:<8} {labeled:<6} {data_preview:<50}")

        elif args.command == "get":
            t = get_task(args.task_id, client)
            print(f"ID:           {t['id']}")
            print(f"项目:         {t.get('project', '?')}")
            print(f"已标注:       {t.get('is_labeled', False)}")
            print(f"标注数:       {t.get('total_annotations', 0)}")
            print(f"创建时间:     {t.get('created_at', '')}")
            print(f"数据:\n{t.get('data', {})}")
            if t.get("annotations"):
                print(f"标注结果 ({len(t['annotations'])} 条):")
                for a in t["annotations"][:5]:
                    print(f"  - {json.dumps(a, ensure_ascii=False)[:200]}")

        elif args.command == "delete":
            ids = [int(x.strip()) for x in args.ids.split(",") if x.strip()]
            if not ids:
                print("❌ 未指定有效的任务 ID")
                return
            if not args.yes:
                ans = input(f"确认删除 {len(ids)} 个任务? [y/N] ")
                if ans.lower() != "y":
                    print("已取消")
                    return
            result = delete_tasks_batch(args.project, ids, client)
            print(f"✅ 删除结果: {result.get('deleted', 0)}/{len(ids)} 个任务")
            for err in result.get("errors", []):
                print(f"  ❌ {err}")

        elif args.command == "stats":
            stats = task_stats(args.project, client)
            pct = (stats["labeled"] / stats["total"] * 100) if stats["total"] > 0 else 0
            bar_len = 30
            filled = int(bar_len * pct / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"总任务:   {stats['total']}")
            print(f"已标注:   {stats['labeled']}  ({pct:.1f}%)")
            print(f"未标注:   {stats['unlabeled']}")
            print(f"[{bar}]")

    except Exception as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
