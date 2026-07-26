#!/usr/bin/env python3
"""Label Studio 项目管理 —— 创建、导入、本地存储配置。

示例:
  python label_studio_project.py list
  python label_studio_project.py info 10
  python label_studio_project.py create "新项目" --template detection
  python label_studio_project.py delete 5
  python label_studio_project.py storage-add 10 /path/to/images
  python label_studio_project.py storage-sync 2
"""

import argparse
import sys

from label_studio_auth import get_client


# ---------------------------------------------------------------------------
# 预设 label_config 模板
# ---------------------------------------------------------------------------

PRESET_TEMPLATES = {
    "detection": {
        "description": "目标检测 (RectangleLabels)",
        "config": (
            '<View>\n'
            '  <Image name="image" value="$image"/>\n'
            '  <RectangleLabels name="label" toName="image">\n'
            '    <Label value="object" background="#FF0000"/>\n'
            '  </RectangleLabels>\n'
            '</View>'
        ),
    },
    "classification": {
        "description": "图像分类 (Choices)",
        "config": (
            '<View>\n'
            '  <Image name="image" value="$image"/>\n'
            '  <Choices name="class" toName="image">\n'
            '    <Choice value="A"/>\n'
            '    <Choice value="B"/>\n'
            '  </Choices>\n'
            '</View>'
        ),
    },
}


def list_templates():
    """列出所有预设模板。"""
    for name, info in PRESET_TEMPLATES.items():
        print(f"  {name:20s}  {info['description']}")


# ---------------------------------------------------------------------------
# 项目管理 API
# ---------------------------------------------------------------------------

def list_projects(client=None) -> list[dict]:
    """列出所有项目。"""
    c = client or get_client()
    return c.get("/api/projects/")


def get_project(project_id: int, client=None) -> dict:
    """获取项目详情。"""
    c = client or get_client()
    return c.get(f"/api/projects/{project_id}/")


def create_project(
    title: str,
    label_config: str | None = None,
    template: str | None = None,
    client=None,
) -> dict:
    """创建新项目。

    Args:
        title: 项目名称
        label_config: 自定义 label_config XML 字符串
        template: 预设模板名称 ("detection", "classification")
    """
    c = client or get_client()

    if label_config:
        config = label_config
    elif template and template in PRESET_TEMPLATES:
        config = PRESET_TEMPLATES[template]["config"]
    else:
        config = PRESET_TEMPLATES["detection"]["config"]

    return c.post("/api/projects/", data={
        "title": title,
        "label_config": config,
    })


def delete_project(project_id: int, client=None) -> dict:
    """删除项目。"""
    c = client or get_client()
    return c.delete(f"/api/projects/{project_id}/")


# ---------------------------------------------------------------------------
# 本地文件存储
# ---------------------------------------------------------------------------

def add_local_storage(
    project_id: int,
    path: str,
    regex_filter: str = r".*\.(jpg|jpeg|png|bmp)$",
    title: str = "本地图片",
    client=None,
) -> dict:
    """为项目配置本地文件存储。

    Args:
        project_id: 项目 ID
        path: 图片目录的绝对路径（需在 LOCAL_FILES_DOCUMENT_ROOT 下）
        regex_filter: 文件名过滤正则
        title: 存储标题
    """
    c = client or get_client()
    return c.post("/api/storages/localfiles/", data={
        "project": project_id,
        "path": path,
        "regex_filter": regex_filter,
        "use_blob_urls": True,
        "title": title,
    })


def sync_storage(storage_id: int, client=None) -> dict:
    """触发存储同步。"""
    c = client or get_client()
    return c.post(f"/api/storages/localfiles/{storage_id}/sync")


def list_storages(project_id: int, client=None) -> list[dict]:
    """列出项目关联的本地文件存储。"""
    c = client or get_client()
    return c.get("/api/storages/localfiles/", params={"project": project_id})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Label Studio 项目管理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="模板:\n" + "\n".join(
            f"  {k:20s} {v['description']}" for k, v in PRESET_TEMPLATES.items()
        ),
    )
    sub = parser.add_subparsers(dest="command", help="操作")

    # ---- list ----
    p_list = sub.add_parser("list", help="列出所有项目")
    p_list.add_argument("--base-url", default=None, help="Label Studio 地址")

    # ---- info ----
    p_info = sub.add_parser("info", help="查看项目详情")
    p_info.add_argument("project_id", type=int, help="项目 ID")

    # ---- create ----
    p_create = sub.add_parser("create", help="创建新项目")
    p_create.add_argument("title", help="项目名称")
    p_create.add_argument(
        "-t", "--template", choices=list(PRESET_TEMPLATES.keys()),
        default="detection", help="预设模板（默认: detection）",
    )

    # ---- delete ----
    p_del = sub.add_parser("delete", help="删除项目")
    p_del.add_argument("project_id", type=int, help="项目 ID")
    p_del.add_argument("--yes", action="store_true", help="跳过确认")

    # ---- storage-add ----
    p_sa = sub.add_parser("storage-add", help="添加本地文件存储")
    p_sa.add_argument("project_id", type=int, help="项目 ID")
    p_sa.add_argument("path", help="图片目录绝对路径")
    p_sa.add_argument("--regex", default=r".*\.(jpg|jpeg|png|bmp)$", help="文件名过滤")

    # ---- storage-sync ----
    p_ss = sub.add_parser("storage-sync", help="触发存储同步")
    p_ss.add_argument("storage_id", type=int, help="存储 ID")

    # ---- storage-list ----
    p_sl = sub.add_parser("storage-list", help="列出项目的存储配置")
    p_sl.add_argument("project_id", type=int, help="项目 ID")

    # ---- templates ----
    sub.add_parser("templates", help="列出预设模板")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    base_url = getattr(args, "base_url", None)
    client = get_client(base_url=base_url) if base_url else get_client()

    # ---------------------------------------------------------------
    try:
        if args.command == "list":
            projects = list_projects(client)
            if not projects:
                print("(无项目)")
                return
            print(f"{'ID':<6} {'标题':<30} {'任务数':<8} {'创建时间'}")
            print("-" * 70)
            for p in projects:
                print(f"{p['id']:<6} {p.get('title', '(无标题)')[:28]:<30} "
                      f"{p.get('task_number', 0):<8} "
                      f"{p.get('created_at', '')[:19]}")

        elif args.command == "info":
            p = get_project(args.project_id, client)
            print(f"ID:           {p['id']}")
            print(f"标题:         {p.get('title', '(无)')}")
            print(f"任务数:       {p.get('task_number', 0)}")
            print(f"标注数:       {p.get('total_annotations', 0)}")
            print(f"创建时间:     {p.get('created_at', '')}")
            print(f"label_config:\n{p.get('label_config', '(无)')}")

        elif args.command == "create":
            p = create_project(args.title, template=args.template, client=client)
            print(f"✅ 项目已创建  ID={p['id']}  标题='{p.get('title', '')}'")
            print(f"   模板: {args.template} ({PRESET_TEMPLATES[args.template]['description']})")

        elif args.command == "delete":
            p = get_project(args.project_id, client)
            if not args.yes:
                ans = input(f"确认删除项目 '{p.get('title', '')}' (ID={args.project_id})? [y/N] ")
                if ans.lower() != "y":
                    print("已取消")
                    return
            delete_project(args.project_id, client)
            print(f"✅ 已删除项目 ID={args.project_id}")

        elif args.command == "storage-add":
            result = add_local_storage(args.project_id, args.path, args.regex, client=client)
            print(f"✅ 存储已添加  ID={result['id']}  路径={result['path']}")
            print(f"   同步: python label_studio_project.py storage-sync {result['id']}")

        elif args.command == "storage-sync":
            result = sync_storage(args.storage_id, client)
            status = result.get("status", "?")
            count = result.get("last_sync_count", "?")
            print(f"同步状态: {status}  导入: {count}")

        elif args.command == "storage-list":
            storages = list_storages(args.project_id, client)
            if not storages:
                print("(无存储配置)")
                return
            for s in storages:
                print(f"  ID={s['id']}  path={s['path']}  "
                      f"blob={s.get('use_blob_urls')}  "
                      f"status={s.get('status')}  "
                      f"synced={s.get('last_sync_count', 0)}")

        elif args.command == "templates":
            list_templates()

    except Exception as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
