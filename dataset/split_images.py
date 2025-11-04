#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将一个文件夹中的图片按顺序切分为“前 80%”与“后 20%”两个文件夹。
—— 所有参数都在本脚本顶部“配置区”内自定义，无需命令行参数。

使用方法：
1. 修改下方“配置区”的路径与参数。
2. 运行：python split_images.py
3. 若 MOVE_OR_COPY = "move"，文件会被移动；若为 "copy"，则会复制。

说明：
- “前 80%/后 20%”基于 ORDER_BY 排序后的文件顺序来划分：
    * name  : 按文件名排序（优先使用 natsort 的自然排序；若未安装则回退为简单自然排序/普通排序）
    * mtime : 按修改时间排序（从旧到新）
    * random: 随机顺序（由 SEED 控制可复现）
- 文件后缀大小写不敏感，受 IMAGE_EXTS 控制。
- 若目标目录中已存在同名文件：
    * OVERWRITE=True  则覆盖（仅 copy 时会覆盖；move 时由系统行为决定，脚本也做了安全处理）。
    * OVERWRITE=False 则自动在文件名后追加 (_1), (_2) 等避免冲突。
- 可先 DRY_RUN=True 进行演练，不会真正写入文件，只打印计划。

作者：你最贴心的小脚本
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

# =========================
#         配置区
# =========================
SOURCE_DIR   = r"/root/llada/dataset/datasets/coco2017/val2017"      # 源图片文件夹
FIRST_DIR    = r"/root/llada/dataset/datasets/coco2017/val2017_split/train" # 前 80% 输出文件夹
SECOND_DIR   = r"/root/llada/dataset/datasets/coco2017/val2017_split/test"  # 后 20% 输出文件夹

RATIO        = 0.80             # 前部分比例（0~1），典型为 0.8
ORDER_BY     = "name"           # 排序方式: "name" | "mtime" | "random"
MOVE_OR_COPY = "copy"           # "move" | "copy"
IMAGE_EXTS   = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"]
SEED         = 42               # 当 ORDER_BY="random" 时用于复现
OVERWRITE    = False            # 是否覆盖同名文件
DRY_RUN      = False            # 设为 True 开启演练模式（不落盘）

# =========================
#        实用函数
# =========================

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in {e.lower() for e in IMAGE_EXTS}

def natural_sort_key_fallback(s: str):
    """
    简单自然排序的回退方案：将连续数字块转为整数比较，其余按小写字符串比较。
    若系统已安装 natsort，则会优先使用 natsort.natsorted。
    """
    _nsre = re.compile(r'(\d+)')
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]

def natsorted_paths(paths: List[Path]) -> List[Path]:
    try:
        from natsort import natsorted  # 优先使用现有库
        return natsorted(paths, key=lambda p: p.name)
    except Exception:
        # 回退到内置的自然排序 key
        return sorted(paths, key=lambda p: natural_sort_key_fallback(p.name))

def ensure_dir(d: Path):
    if not d.exists():
        d.mkdir(parents=True, exist_ok=True)

def unique_destination(dst: Path) -> Path:
    """
    当 OVERWRITE=False 且 dst 已存在时，生成不冲突的新文件名：
    xxx.ext -> xxx_(1).ext, xxx_(2).ext, ...
    """
    if OVERWRITE or not dst.exists():
        return dst
    stem, suffix = dst.stem, dst.suffix
    i = 1
    while True:
        candidate = dst.with_name(f"{stem}_({i}){suffix}")
        if not candidate.exists():
            return candidate
        i += 1

@dataclass
class PlanItem:
    src: Path
    dst: Path

def plan_transfers(files: List[Path], first_dir: Path, second_dir: Path, ratio: float):
    n = len(files)
    cut = int(n * ratio)
    first_part = files[:cut]
    second_part = files[cut:]

    first_plan = [PlanItem(src=f, dst=unique_destination(first_dir / f.name)) for f in first_part]
    second_plan = [PlanItem(src=f, dst=unique_destination(second_dir / f.name)) for f in second_part]
    return first_plan, second_plan

def describe(plan: List[PlanItem], tag: str) -> None:
    print(f"\n=== {tag}（{len(plan)} 个文件）===")
    preview = 10
    for i, item in enumerate(plan[:preview], 1):
        print(f"[{i:>2}] {item.src.name}  ->  {item.dst}")
    if len(plan) > preview:
        print(f"... 其余 {len(plan) - preview} 个文件省略显示")

def transfer(plan: List[PlanItem], mode: str):
    for item in plan:
        if DRY_RUN:
            continue
        # 确保目标目录存在
        ensure_dir(item.dst.parent)
        if mode == "move":
            # move 到同盘等情况可能覆盖或失败，这里我们手动处理：
            # 若需覆盖则先删除已有目标；若不覆盖则生成 unique 名称已在 plan 阶段完成。
            if item.dst.exists():
                if OVERWRITE:
                    if item.dst.is_file():
                        item.dst.unlink()
                    else:
                        shutil.rmtree(item.dst)
                else:
                    item.dst = unique_destination(item.dst)
            shutil.move(str(item.src), str(item.dst))
        elif mode == "copy":
            # copy2 保留元数据
            if item.dst.exists():
                if OVERWRITE:
                    if item.dst.is_file():
                        item.dst.unlink()
                    else:
                        shutil.rmtree(item.dst)
                else:
                    item.dst = unique_destination(item.dst)
            shutil.copy2(str(item.src), str(item.dst))
        else:
            raise ValueError("MOVE_OR_COPY 只能为 'move' 或 'copy'")

def main():
    src_dir = Path(SOURCE_DIR).expanduser().resolve()
    first_dir = Path(FIRST_DIR).expanduser().resolve()
    second_dir = Path(SECOND_DIR).expanduser().resolve()

    if not src_dir.exists() or not src_dir.is_dir():
        print(f"❌ 源目录不存在或不是文件夹：{src_dir}")
        sys.exit(1)

    # 收集图片
    files = [p for p in src_dir.iterdir() if is_image(p)]
    if not files:
        print(f"⚠️ 在 {src_dir} 未找到任何图片（扩展名：{', '.join(IMAGE_EXTS)}）")
        sys.exit(0)

    # 排序
    order = ORDER_BY.lower().strip()
    if order == "name":
        files = natsorted_paths(files)
    elif order == "mtime":
        files = sorted(files, key=lambda p: p.stat().st_mtime)  # 从旧到新
    elif order == "random":
        rnd = random.Random(SEED)
        rnd.shuffle(files)
    else:
        print("⚠️ 未知 ORDER_BY，已回退为 'name' 排序")
        files = natsorted_paths(files)

    # 规划与展示
    ensure_dir(first_dir)
    ensure_dir(second_dir)
    first_plan, second_plan = plan_transfers(files, first_dir, second_dir, RATIO)

    print(f"共发现 {len(files)} 张图片。将按 {ORDER_BY} 排序后切分：前 {int(len(first_plan))} 张 -> {first_dir}；后 {int(len(second_plan))} 张 -> {second_dir}")
    if DRY_RUN:
        print("（DRY_RUN=True：当前为演练模式，不会实际写入文件）")

    describe(first_plan, "前 80%（或 RATIO 指定的前部分）")
    describe(second_plan, "后 20%")

    # 执行
    transfer(first_plan, MOVE_OR_COPY.lower())
    transfer(second_plan, MOVE_OR_COPY.lower())

    action = "移动" if MOVE_OR_COPY.lower() == "move" else "复制"
    if DRY_RUN:
        print("\n✅ 演练完成。未进行任何实际写入。")
    else:
        print(f"\n✅ 完成：已{action} {len(first_plan) + len(second_plan)} 个文件。")
        print(f"   - 前部分输出目录：{first_dir}")
        print(f"   - 后部分输出目录：{second_dir}")

if __name__ == "__main__":
    main()
