#!/usr/bin/env python3
"""Label Studio 标注工具 —— CLI 入口，转发到各子模块（支持本机 / 远程）。

CLI 示例:
  python label_studio_main.py project list
  python label_studio_main.py --base-url http://192.168.1.10:8080 --token <TOKEN> project list
  python label_studio_main.py data stats -p 10

也可以通过 CTTS 统一入口调用:
  python ctts.py label project list
  python ctts.py label -- --base-url http://host:8080 --token <TOKEN> project list

GUI 请使用:
  python ctts.py --gui
"""

import argparse
import sys

from label_studio_auth import DEFAULT_TIMEOUT, add_connection_args


def cli_main():
    """CLI 入口：解析全局连接参数后转发到各模块。"""
    parser = argparse.ArgumentParser(
        description="Label Studio 标注工具集 (CLI，支持本机 / 远程)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_connection_args(parser)
    sub = parser.add_subparsers(dest="module", help="功能模块")

    # ---- project ----
    p_proj = sub.add_parser("project", help="项目管理")
    p_proj_sub = p_proj.add_subparsers(dest="action", help="操作")

    p_proj_sub.add_parser("list", help="列出项目")
    pa = p_proj_sub.add_parser("info", help="项目详情")
    pa.add_argument("project_id", type=int)
    pa = p_proj_sub.add_parser("create", help="创建项目")
    pa.add_argument("title")
    pa.add_argument("-t", "--template", default="detection", choices=["detection", "classification"])
    pa = p_proj_sub.add_parser("delete", help="删除项目")
    pa.add_argument("project_id", type=int)
    pa.add_argument("--yes", action="store_true")
    pa = p_proj_sub.add_parser("storage-add", help="添加本地存储")
    pa.add_argument("project_id", type=int)
    pa.add_argument("path")
    pa.add_argument("--regex", default=r".*\.(jpg|jpeg|png|bmp)$", help="文件名过滤正则")
    pa.add_argument("--recursive", action="store_true", help="递归扫描子目录")
    pa = p_proj_sub.add_parser("storage-sync", help="同步存储")
    pa.add_argument("storage_id", type=int)
    pa = p_proj_sub.add_parser("storage-list", help="列出存储")
    pa.add_argument("project_id", type=int)
    p_proj_sub.add_parser("templates", help="列出模板")

    # ---- data ----
    p_data = sub.add_parser("data", help="数据管理")
    p_data_sub = p_data.add_subparsers(dest="action", help="操作")

    pa = p_data_sub.add_parser("list", help="列出任务")
    pa.add_argument("-p", "--project", type=int, required=True)
    pa.add_argument("--page", type=int, default=1)
    pa.add_argument("--page-size", type=int, default=20)
    pa = p_data_sub.add_parser("get", help="任务详情")
    pa.add_argument("task_id", type=int)
    pa = p_data_sub.add_parser("delete", help="批量删除")
    pa.add_argument("-p", "--project", type=int, required=True)
    pa.add_argument("--ids", required=True)
    pa.add_argument("--yes", action="store_true")
    pa = p_data_sub.add_parser("stats", help="任务统计")
    pa.add_argument("-p", "--project", type=int, required=True)

    # ---- auth ----
    sub.add_parser("auth", help="认证测试")

    args = parser.parse_args()

    if args.module is None:
        parser.print_help()
        return

    # 将全局连接参数注入子模块 argv
    conn_flags: list[str] = []
    if args.base_url:
        conn_flags += ["--base-url", args.base_url]
    if args.token:
        conn_flags += ["--token", args.token]
    if args.db:
        conn_flags += ["--db", args.db]
    if args.timeout is not None and args.timeout != DEFAULT_TIMEOUT:
        conn_flags += ["--timeout", str(args.timeout)]

    try:
        mod_idx = sys.argv.index(args.module)
        rest = [a for a in sys.argv[mod_idx + 1:] if a != "--"]
    except ValueError:
        rest = []

    sys.argv = [sys.argv[0]] + conn_flags + rest

    if args.module == "project":
        from label_studio_project import main as m
    elif args.module == "data":
        from label_studio_data import main as m
    elif args.module == "auth":
        from label_studio_auth import main as m
    else:
        parser.print_help()
        return

    m()


def main():
    cli_main()


if __name__ == "__main__":
    main()
