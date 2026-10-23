#!/usr/bin/env python3
"""Label Studio 项目管理 —— 创建、导入、本地存储配置（支持本机 / 远程实例）。

示例:
  # 本机
  python label_studio_project.py list

  # 远程
  python label_studio_project.py --base-url http://192.168.1.10:8080 --token <TOKEN> list
  python label_studio_project.py --base-url http://host:8080 create "新项目" -t detection

注意: storage-add 的 path 是 Label Studio 服务端本机路径，不是调用方机器路径。
"""

import argparse
import sys
from xml.sax.saxutils import escape

from .label_studio_auth import (
    add_connection_args,
    client_from_args,
    get_client,
    is_local_url,
    resolve_base_url,
    to_dict,
)


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


def detection_label_config(labels: list[str]) -> str:
    """生成目标检测模板，类别顺序即训练数据 classes.txt 顺序。"""
    labels = [str(v).strip() for v in labels if str(v).strip()]
    if not labels:
        raise ValueError("labels 不能为空")
    color = "#FF0000"
    lines = [
        "<View>",
        '  <Image name="image" value="$image"/>',
        '  <RectangleLabels name="label" toName="image">',
    ]
    for label in labels:
        lines.append(f'    <Label value="{escape(label)}" background="{color}"/>')
    lines += [
        "  </RectangleLabels>",
        "</View>",
    ]
    return "\n".join(lines)


def list_templates():
    """列出所有预设模板。"""
    for name, info in PRESET_TEMPLATES.items():
        print(f"  {name:20s}  {info['description']}")


# ---------------------------------------------------------------------------
# 项目管理 API
# ---------------------------------------------------------------------------

def list_projects(client=None) -> list[dict]:
    """列出所有项目（SDK 自动翻页）。"""
    c = client or get_client()
    return [to_dict(p) for p in c.projects.list()]


def get_project(project_id: int, client=None) -> dict:
    """获取项目详情。"""
    c = client or get_client()
    return to_dict(c.projects.get(project_id))


def create_project(
    title: str,
    label_config: str | None = None,
    template: str | None = None,
    labels: list[str] | None = None,
    client=None,
) -> dict:
    """创建新项目。"""
    c = client or get_client()

    if label_config:
        config = label_config
    elif labels:
        config = detection_label_config(labels)
    elif template and template in PRESET_TEMPLATES:
        config = PRESET_TEMPLATES[template]["config"]
    else:
        config = PRESET_TEMPLATES["detection"]["config"]

    return to_dict(c.projects.create(title=title, label_config=config))


def delete_project(project_id: int, client=None) -> dict:
    """删除项目。"""
    c = client or get_client()
    c.projects.delete(project_id)
    return {"deleted": 1}


# ---------------------------------------------------------------------------
# 本地文件存储（路径相对于 Label Studio 服务端）
# ---------------------------------------------------------------------------

def add_local_storage(
    project_id: int,
    path: str,
    regex_filter: str = r".*\.(jpg|jpeg|png|bmp)$",
    title: str = "本地图片",
    recursive_scan: bool = False,
    client=None,
) -> dict:
    """为项目配置本地文件存储。

    path 必须是 Label Studio 进程所在机器上的绝对路径，
    且位于其 LOCAL_FILES_DOCUMENT_ROOT 下。
    """
    c = client or get_client()
    storage = c.import_storage.local.create(
        project=project_id,
        path=path,
        regex_filter=regex_filter,
        use_blob_urls=True,
        title=title,
        recursive_scan=recursive_scan,
    )
    return to_dict(storage)


def sync_storage(storage_id: int, client=None) -> dict:
    """触发存储同步。"""
    c = client or get_client()
    return to_dict(c.import_storage.local.sync(storage_id))


def list_storages(project_id: int, client=None) -> list[dict]:
    """列出项目关联的本地文件存储。"""
    c = client or get_client()
    storages = c.import_storage.local.list(project=project_id)
    return [to_dict(s) for s in storages]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    conn = argparse.ArgumentParser(add_help=False)
    add_connection_args(conn)

    parser = argparse.ArgumentParser(
        description="Label Studio 项目管理（支持本机 / 远程）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[conn],
        epilog="模板:\n" + "\n".join(
            f"  {k:20s} {v['description']}" for k, v in PRESET_TEMPLATES.items()
        ),
    )
    sub = parser.add_subparsers(dest="command", help="操作")

    sub.add_parser("list", help="列出所有项目")

    p_info = sub.add_parser("info", help="查看项目详情")
    p_info.add_argument("project_id", type=int, help="项目 ID")

    p_create = sub.add_parser("create", help="创建新项目")
    p_create.add_argument("title", help="项目名称")
    p_create.add_argument(
        "-t", "--template", choices=list(PRESET_TEMPLATES.keys()),
        default="detection", help="预设模板（默认: detection）",
    )
    p_create.add_argument(
        "--labels", default=None,
        help="检测类别，逗号分隔（如: stb,person）；优先于模板",
    )
    p_create.add_argument(
        "--labels-file", default=None,
        help="从 classes.txt 读取类别（每行一个）",
    )

    p_del = sub.add_parser("delete", help="删除项目")
    p_del.add_argument("project_id", type=int, help="项目 ID")
    p_del.add_argument("--yes", action="store_true", help="跳过确认")

    p_sa = sub.add_parser("storage-add", help="添加本地文件存储（路径在 LS 服务端）")
    p_sa.add_argument("project_id", type=int, help="项目 ID")
    p_sa.add_argument("path", help="LS 服务端上的图片目录绝对路径")
    p_sa.add_argument("--regex", default=r".*\.(jpg|jpeg|png|bmp)$", help="文件名过滤")
    p_sa.add_argument("--recursive", action="store_true", help="递归扫描子目录")

    p_ss = sub.add_parser("storage-sync", help="触发存储同步")
    p_ss.add_argument("storage_id", type=int, help="存储 ID")

    p_sl = sub.add_parser("storage-list", help="列出项目的存储配置")
    p_sl.add_argument("project_id", type=int, help="项目 ID")

    sub.add_parser("templates", help="列出预设模板")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "templates":
        list_templates()
        return

    try:
        client = client_from_args(args)
        url = client.base_url
        if not is_local_url(url):
            print(f"→ 远程实例: {url}")

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
            print(f"标注数:       {p.get('total_annotations_number', 0)}")
            print(f"创建时间:     {p.get('created_at', '')}")
            print(f"label_config:\n{p.get('label_config', '(无)')}")

        elif args.command == "create":
            labels = None
            if args.labels_file:
                labels = [line.strip() for line in
                          open(args.labels_file, encoding="utf-8")
                          if line.strip()]
            elif args.labels:
                labels = [v.strip() for v in args.labels.split(",") if v.strip()]
            p = create_project(args.title, template=args.template,
                               labels=labels, client=client)
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
            if not is_local_url(resolve_base_url(args.base_url)):
                print("⚠ path 必须是远程 Label Studio 服务器上的路径，不是本机路径。")
            result = add_local_storage(
                args.project_id, args.path, args.regex,
                recursive_scan=args.recursive, client=client,
            )
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

    except Exception as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
