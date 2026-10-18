#!/usr/bin/env python3
"""CTTS 跨平台环境初始化 —— 创建 venv 并安装依赖。

支持 Linux / macOS / Windows：
  - 优先使用 uv（若已安装），否则回退到 venv + pip
  - venv 默认放在 ~/.cache/ctts-venv（eDisk 不支持 symlink）
  - Windows 下自动把 TMP/TEMP 指向用户目录，避免 eDisk 解压失败

用法:
  python setup_env.py                 # 创建/更新 venv 并安装依赖
  python setup_env.py --check         # 只检查环境，不做修改
  python setup_env.py --run           # 环境就绪后直接启动 GUI
  python setup_env.py --recreate      # 删除并重建 venv
  python setup_env.py --venv PATH     # 自定义 venv 位置
  python setup_env.py --python PATH   # 指定创建 venv 用的解释器

Windows 可直接双击 setup_env.cmd；Linux/macOS 可运行 ./setup_env.sh
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_VENV = Path(
    os.environ.get("CTTS_VENV", Path.home() / ".cache" / "ctts-venv")
).expanduser()

# 安装完成后必须能导入的模块（与 pyproject.toml 依赖对应）
_REQUIRED_MODULES = ("cv2", "numpy", "httpx", "label_studio_sdk", "PySide6", "ultralytics")
_IMPORT_CHECK = "import " + ", ".join(_REQUIRED_MODULES) + "; print('依赖导入正常')"


# ---------------------------------------------------------------------------
# 平台辅助
# ---------------------------------------------------------------------------

def is_windows() -> bool:
    return os.name == "nt"


def venv_python(venv: Path) -> Path:
    """venv 内 Python 的可执行文件路径。"""
    if is_windows():
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def venv_activate_hint(venv: Path) -> str:
    if is_windows():
        return f'& "{venv / "Scripts" / "Activate.ps1"}"'
    return f'source "{venv / "bin" / "activate"}"'


def prepare_platform_env() -> None:
    """Windows 专用：把临时目录与 uv 缓存放到用户目录。"""
    if not is_windows():
        return
    tmp = Path.home() / ".cache" / "ctts-tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = str(tmp)
    os.environ["TEMP"] = str(tmp)
    os.environ.setdefault("UV_CACHE_DIR", str(Path.home() / ".cache" / "uv"))


def _setup_console() -> None:
    """尽量让 Windows 控制台正确输出中文/符号。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


# ---------------------------------------------------------------------------
# 项目信息
# ---------------------------------------------------------------------------

def read_project() -> tuple[list[str], tuple[int, int]]:
    """读取 pyproject.toml 的依赖与最低 Python 版本。"""
    pyproject = SCRIPT_DIR / "pyproject.toml"
    with pyproject.open("rb") as f:
        data = tomllib.load(f)
    project = data.get("project", {})
    deps = list(project.get("dependencies", []))

    min_version = (3, 13)
    requires = project.get("requires-python", "")
    if ">=" in requires:
        try:
            major, minor = requires.split(">=")[1].strip().split(".")[:2]
            min_version = (int(major), int(minor))
        except (IndexError, ValueError):
            pass
    return deps, min_version


def python_ok(executable: str | Path, min_version: tuple[int, int]) -> bool:
    """检查解释器版本是否满足要求。"""
    try:
        out = subprocess.run(
            [str(executable), "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
            capture_output=True, text=True, timeout=20,
        )
        if out.returncode != 0:
            return False
        major, minor = (int(x) for x in out.stdout.strip().split(".")[:2])
        return (major, minor) >= min_version
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def imports_ok(py: Path) -> bool:
    """检查 venv 内依赖是否可导入。"""
    if not py.is_file():
        return False
    try:
        return subprocess.run(
            [str(py), "-c", _IMPORT_CHECK],
            capture_output=True, timeout=180,
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


# ---------------------------------------------------------------------------
# 执行辅助
# ---------------------------------------------------------------------------

def run(cmd: list[str], *, env: dict | None = None, cwd: Path | None = None) -> None:
    """打印并执行命令，失败时抛出 CalledProcessError。"""
    printable = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    print(f"  $ {printable}")
    subprocess.run(cmd, check=True, env=env, cwd=cwd)


def make_venv(venv: Path, base_python: str, min_version: tuple[int, int]) -> tuple[bool, str]:
    """创建 venv，返回 (是否使用 uv, 实际使用的解释器)。"""
    uv = shutil.which("uv")
    if uv:
        use_python = base_python
        if not python_ok(base_python, min_version):
            # 让 uv 自动下载满足 requires-python 的版本
            use_python = f"{min_version[0]}.{min_version[1]}"
            print(f"⚠ 当前解释器版本过低，交给 uv 获取 Python {use_python}")
        print(f"\n[1/3] 使用 uv 创建 venv: {venv}")
        run([uv, "venv", "--python", use_python, str(venv)])
        return True, use_python

    print(f"\n[1/3] 未找到 uv，使用内置 venv 模块创建: {venv}")
    if not python_ok(base_python, min_version):
        raise SystemExit(
            f"❌ 当前 Python {sys.version.split()[0]} 低于项目要求 "
            f"{min_version[0]}.{min_version[1]}，且未安装 uv。\n"
            "   请安装 uv（Arch: paru -S uv）或更高版本的 Python。"
        )
    run([base_python, "-m", "venv", str(venv)])
    return False, base_python


def install_deps(venv: Path, used_uv: bool, deps: list[str]) -> None:
    """安装项目依赖。"""
    py = venv_python(venv)
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(venv)
    env["PATH"] = str(py.parent) + os.pathsep + env.get("PATH", "")

    if used_uv:
        uv = shutil.which("uv")
        print("\n[2/3] 同步依赖 (uv sync)")
        try:
            # eDisk 不支持 symlink，copy 模式更稳妥
            run([uv, "sync", "--active", "--link-mode=copy"], env=env, cwd=SCRIPT_DIR)
            return
        except subprocess.CalledProcessError:
            print("⚠ uv sync 失败，回退到 uv pip install")

        print("\n[2/3] 安装依赖 (uv pip)")
        run([uv, "pip", "install", "--python", str(py), *deps], env=env, cwd=SCRIPT_DIR)
        return

    print("\n[2/3] 安装依赖 (pip)")
    run([str(py), "-m", "pip", "install", "--upgrade", "pip"], env=env)
    run([str(py), "-m", "pip", "install", *deps], env=env)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def ensure_environment(
    venv: Path,
    base_python: str,
    *,
    recreate: bool = False,
    check_only: bool = False,
) -> bool:
    """确保 venv 与依赖就绪，返回是否可以继续运行。"""
    deps, min_version = read_project()
    py = venv_python(venv)

    if recreate and venv.exists():
        if not (venv / "pyvenv.cfg").is_file():
            raise SystemExit(f"❌ 拒绝删除非 venv 目录: {venv}")
        print(f"删除旧 venv: {venv}")
        shutil.rmtree(venv)

    if check_only:
        print(f"Python:      {sys.version.split()[0]}  ({sys.executable})")
        print(f"venv:        {venv}  ({'存在' if py.is_file() else '不存在'})")
        ok = imports_ok(py)
        print(f"依赖:        {'✅ 可导入' if ok else '❌ 缺失'}")
        if not ok:
            print("\n运行 `python setup_env.py` 安装依赖。")
        return ok

    if py.is_file() and imports_ok(py):
        print(f"✅ 环境已就绪: {venv}")
        return True

    if venv_python(venv).is_file():
        # 复用已有 venv（例如上次安装中断），避免 uv venv 报“已存在”
        used_uv = shutil.which("uv") is not None
        print(f"\n[1/3] 复用现有 venv: {venv}")
    else:
        used_uv, _ = make_venv(venv, base_python, min_version)
        if not venv_python(venv).is_file():
            raise SystemExit(f"❌ venv 创建失败: {venv}")

    install_deps(venv, used_uv, deps)

    print("\n[3/3] 验证依赖")
    if not imports_ok(venv_python(venv)):
        raise SystemExit(
            "❌ 依赖安装后仍无法导入。\n"
            f"   可手动运行: \"{venv_python(venv)}\" -c \"{_IMPORT_CHECK}\"\n"
            "   Linux 若提示 libGL 缺失，请安装 mesa（如: paru -S mesa）。"
        )
    print("✅ 依赖导入正常")
    return True


def main() -> int:
    _setup_console()
    prepare_platform_env()

    parser = argparse.ArgumentParser(
        description="CTTS 跨平台环境初始化（Linux / macOS / Windows）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--venv", default=str(DEFAULT_VENV), help=f"venv 路径（默认: {DEFAULT_VENV}）")
    parser.add_argument(
        "--python", default=os.environ.get("CTTS_PYTHON", sys.executable),
        help="用于创建 venv 的解释器（默认: 当前 Python）",
    )
    parser.add_argument("--check", action="store_true", help="只检查环境，不做修改")
    parser.add_argument("--run", action="store_true", help="环境就绪后启动 GUI")
    parser.add_argument("--recreate", action="store_true", help="删除并重建 venv")
    args = parser.parse_args()

    venv = Path(args.venv).expanduser().resolve()
    print(f"CTTS 环境初始化")
    print(f"  平台: {sys.platform}")
    print(f"  venv: {venv}")

    ready = ensure_environment(
        venv, args.python, recreate=args.recreate, check_only=args.check,
    )
    if not ready:
        return 1
    if args.check:
        return 0

    print("\n完成。激活环境:")
    print(f"  {venv_activate_hint(venv)}")
    print("启动 GUI:")
    print(f'  "{venv_python(venv)}" "{SCRIPT_DIR / "ctts.py"}" --gui')

    if args.run:
        print("\n启动 GUI ...")
        os.execv(str(venv_python(venv)), [str(venv_python(venv)), str(SCRIPT_DIR / "ctts.py"), "--gui"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
