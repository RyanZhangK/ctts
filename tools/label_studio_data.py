#!/usr/bin/env python3
"""Label Studio 任务数据管理 —— 增删查、统计。

示例:
  python label_studio_data.py list -p 10
  python label_studio_data.py get 1971
  python label_studio_data.py delete -p 10 --ids 1,2,3
  python label_studio_data.py stats -p 10
"""

import argparse
import json
import os
import sys

from label_studio_auth import get_client


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
    return c.get("/api/tasks/", params={
        "project": project_id,
        "page": page,
        "page_size": page_size,
    })


def get_task(task_id: int, client=None) -> dict:
    """获取单个任务详情。"""
    c = client or get_client()
    return c.get(f"/api/tasks/{task_id}/")


def delete_task(task_id: int, client=None) -> dict:
    """删除单个任务。"""
    c = client or get_client()
    return c.delete(f"/api/tasks/{task_id}/")


def delete_tasks_batch(project_id: int, task_ids: list[int], client=None) -> dict:
    """批量删除任务。"""
    c = client or get_client()
    return c.post("/api/tasks/delete/", data={
        "project": project_id,
        "selected_items": {"all": False, "included": task_ids},
    })


# ---------------------------------------------------------------------------
# 批量导入
# ---------------------------------------------------------------------------

def import_tasks_from_images(
    project_id: int,
    image_paths: list[str],
    client=None,
) -> dict:
    """批量导入本地图片作为标注任务。

    通过 Label Studio 的文件上传 API 将每张图片创建为一个任务。
    每次请求只发送一张图片，避免单次请求过大导致超时或服务端错误。

    Args:
        project_id: 目标项目 ID
        image_paths: 本地图片文件路径列表
        client: Label Studio 客户端（可选）

    Returns:
        API 响应 dict，包含 task_count 和 errors 列表
    """
    import mimetypes

    c = client or get_client()
    url = f"{c.base_url}/api/projects/{project_id}/import"

    # 文件上传需要 multipart/form-data，必须临时移除 session 中硬编码的
    # application/json，否则 Label Studio 会尝试把二进制图片当 JSON 解析
    ct_header = c.session.headers.pop("Content-Type", None)

    total = len(image_paths)
    errors = []

    try:
        for i, p in enumerate(image_paths, 1):
            mime_type = mimetypes.guess_type(p)[0] or "image/png"
            filename = os.path.basename(p)
            try:
                with open(p, "rb") as fh:
                    r = c.session.post(
                        url,
                        files={"file": (filename, fh, mime_type)},
                    )
                if not r.ok:
                    try:
                        detail = r.json()
                    except Exception:
                        detail = r.text[:500]
                    errors.append(f"[{i}/{total}] {filename}: HTTP {r.status_code} — {detail}")
                    continue
            except Exception as e:
                errors.append(f"[{i}/{total}] {filename}: {e}")
                continue
    finally:
        if ct_header:
            c.session.headers["Content-Type"] = ct_header

    result = {"task_count": total - len(errors)}
    if errors:
        result["errors"] = errors
    return result


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------

def task_stats(project_id: int, client=None) -> dict:
    """获取项目任务统计。

    Returns:
        {total, labeled, unlabeled, has_annotations, has_predictions}
    """
    c = client or get_client()
    # 获取项目概览
    project = c.get(f"/api/projects/{project_id}/")
    # 获取已标注任务数
    total = project.get("task_number", 0)
    annotated = project.get("num_tasks_with_annotations", 0)

    return {
        "total": total,
        "labeled": annotated,
        "unlabeled": total - annotated,
        "total_annotations": project.get("total_annotations_number", 0),
        "total_predictions": project.get("total_predictions_number", 0),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Label Studio 任务数据管理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="操作")

    # ---- list ----
    p_list = sub.add_parser("list", help="列出任务")
    p_list.add_argument("-p", "--project", type=int, required=True, help="项目 ID")
    p_list.add_argument("--page", type=int, default=1, help="页码（默认: 1）")
    p_list.add_argument("--page-size", type=int, default=20, help="每页数量（默认: 20）")

    # ---- get ----
    p_get = sub.add_parser("get", help="查看任务详情")
    p_get.add_argument("task_id", type=int, help="任务 ID")

    # ---- delete ----
    p_del = sub.add_parser("delete", help="批量删除任务")
    p_del.add_argument("-p", "--project", type=int, required=True, help="项目 ID")
    p_del.add_argument("--ids", required=True, help="任务 ID 列表，逗号分隔（如: 1,2,3）")
    p_del.add_argument("--yes", action="store_true", help="跳过确认")

    # ---- stats ----
    p_stats = sub.add_parser("stats", help="任务统计")
    p_stats.add_argument("-p", "--project", type=int, required=True, help="项目 ID")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    client = get_client()

    try:
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
            actual = result.get("deleted", result.get("detail", str(result)))
            print(f"✅ 删除结果: {actual}（请求 {len(ids)} 个任务）")

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
