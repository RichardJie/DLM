# -*- coding: utf-8 -*-
"""
可视化脚本：左侧显示完整 JSON + 解析出的框坐标；右侧显示图片并绘制所有框。
支持答案中为 bbox_list 或 bbox_groups（或两者并存），会从文本中抓取所有 [x1,y1,x2,y2] 并拉平。
"""

import os, re, json, argparse, textwrap
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # 服务器无显示环境
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm

# ======== 解析相关：匹配任意文本里的 [x1,y1,x2,y2] 四元组 ========
_BBOX_4_RE = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)
# 是否合并解析该样本中所有 gpt 的回复文本（True 更鲁棒）
_PARSE_ALL_GPT = True


def parse_args():
    ap = argparse.ArgumentParser(
        "可视化：左侧完整 JSON + 解析出的 bbox；右侧图片+框"
    )
    # ======= 把“指令/路径”都放在 default，开箱即用 =======
    ap.add_argument(
        "--json",
        default="/hpc2ssd/JH_DATA/spooler/yuxuanzhao/lijungang/wujie/LLaDA-V/dataset/coco2017/llava_multi/coco_val2017_grouped_by_category.json",
        help="数据集 JSON（LLaVA/LLaDA-V 格式），默认使用 COCO val2017 的全实例文件",
    )
    ap.add_argument(
        "--img-root",
        default="/hpc2ssd/JH_DATA/spooler/yuxuanzhao/lijungang/wujie/LLaDA-V/dataset/coco2017",
        help="图片根目录，需与 JSON 中 image 相对路径对齐（默认 COCO 根目录）",
    )
    ap.add_argument(
        "--out",
        default="/hpc2ssd/JH_DATA/spooler/yuxuanzhao/lijungang/wujie/LLaDA-V/dataset/coco2017/vis_val2017",
        help="可视化输出目录（默认写到 coco2017/vis_val2017）",
    )
    ap.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="仅可视化指定样本 id（可多个），默认 None=全量（配合 --limit 使用）",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=50,  # 你要求的默认 50
        help="最多可视化多少条（默认50）",
    )
    ap.add_argument("--dpi", type=int, default=150, help="导出 DPI（默认150）")
    ap.add_argument(
        "--wrap",
        type=int,
        default=100,
        help="左侧每行最大字符数，用于长行折行（默认100）",
    )
    ap.add_argument(
        "--max_height",
        type=float,
        default=22.0,
        help="画布最大高度（英寸），防止爆高（默认22）",
    )
    return ap.parse_args()


def collect_all_gpt_values(conv):
    """收集该样本中所有 gpt 的 value 文本（按出现顺序）"""
    out = []
    if isinstance(conv, list):
        for turn in conv:
            if isinstance(turn, dict) and turn.get("from") == "gpt":
                v = turn.get("value", "")
                if isinstance(v, str) and v.strip():
                    out.append(v)
    return out


def find_last_human_and_gpt(conv):
    """返回最后一对 (human_value, gpt_value)，保留以防需要"""
    if not isinstance(conv, list) or len(conv) < 2:
        return None, None
    last_gpt_idx = None
    for i in range(len(conv) - 1, -1, -1):
        if isinstance(conv[i], dict) and conv[i].get("from") == "gpt":
            last_gpt_idx = i
            break
    if last_gpt_idx is None:
        return None, None
    human_value = None
    for j in range(last_gpt_idx - 1, -1, -1):
        if isinstance(conv[j], dict) and conv[j].get("from") == "human":
            human_value = conv[j].get("value", "")
            break
    gpt_value = conv[last_gpt_idx].get("value", "")
    return human_value, gpt_value


def extract_bbox_list(ans_str):
    """
    从答案文本中抓取所有 [x1,y1,x2,y2]（无论是 bbox_list 还是 bbox_groups），
    返回 flatten 后的一维列表：[[x1,y1,x2,y2], ...]
    """
    if not ans_str or not isinstance(ans_str, str):
        return None
    s = ans_str.strip()
    # 去掉常见前缀（可选，不影响 _BBOX_4_RE 抓取）
    for prefix in ("bbox_list", "bbox_groups", "bbox_group", "bbox"):
        if s.lower().startswith(prefix):
            colon = s.find(":")
            if colon >= 0:
                s = s[colon + 1 :].strip()
                break

    boxes = []
    for m in _BBOX_4_RE.finditer(s):
        try:
            x1, y1, x2, y2 = map(float, m.groups())
            boxes.append([x1, y1, x2, y2])
        except Exception:
            pass
    return boxes if boxes else None


def draw_one(sample, img_root, out_dir, dpi=150, wrap_width=100, max_height=22.0):
    img_rel = sample.get("image")
    if not img_rel:
        return False, "缺少 image 字段"
    img_path = os.path.join(img_root, img_rel)
    if not os.path.exists(img_path):
        return False, f"缺图: {img_path}"

    conv = sample.get("conversations", [])
    if _PARSE_ALL_GPT:
        gpt_texts = collect_all_gpt_values(conv)
        to_parse = "\n".join(gpt_texts) if gpt_texts else ""
    else:
        human_txt, gpt_txt = find_last_human_and_gpt(conv)
        to_parse = gpt_txt or ""

    boxes = extract_bbox_list(to_parse)

    # 完整 JSON（pretty）
    pretty_json = json.dumps(sample, ensure_ascii=False, indent=2)

    # 左侧文本拼装：完整 JSON + 解析到的 bbox（归一化）
    def wrap_block(text):
        out_lines = []
        for line in text.splitlines():
            if wrap_width and wrap_width > 0:
                out_lines.extend(
                    textwrap.wrap(
                        line,
                        width=wrap_width,
                        replace_whitespace=False,
                        drop_whitespace=False,
                    )
                )
            else:
                out_lines.append(line)
        return out_lines

    left_lines = []
    left_lines.append("FULL JSON:")
    left_lines.extend(wrap_block(pretty_json))
    left_lines.append("")
    left_lines.append(
        "---- Parsed boxes (flattened from bbox_list / bbox_groups, normalized 0-1) ----"
    )
    if boxes:
        for i, b in enumerate(boxes, 1):
            left_lines.append(
                f"{i:02d}: [{b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f}, {b[3]:.3f}]"
            )
    else:
        left_lines.append("(no bbox parsed)")

    # 根据文本行数决定画布高度
    n_lines = len(left_lines)
    height = min(max(6.0, n_lines * 0.18), max_height)

    # 创建画布（左文本，右图像）
    fig = plt.figure(figsize=(16, height), dpi=dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 2.0])
    ax_text = fig.add_subplot(gs[0, 0])
    ax_img = fig.add_subplot(gs[0, 1])

    # 左侧文本
    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        "\n".join(left_lines),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
    )

    # 右侧图片+框
    ax_img.axis("off")
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    ax_img.imshow(im)

    colors = [
        "red",
        "lime",
        "cyan",
        "yellow",
        "magenta",
        "orange",
        "deepskyblue",
        "springgreen",
        "violet",
        "gold",
    ]
    if boxes:
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = b
            px1, py1 = x1 * W, y1 * H
            px2, py2 = x2 * W, y2 * H
            w, h = max(1.0, px2 - px1), max(1.0, py2 - py1)
            color = colors[i % len(colors)]
            ax_img.add_patch(
                Rectangle((px1, py1), w, h, fill=False, linewidth=2.0, edgecolor=color)
            )
            ax_img.text(
                px1 + 2,
                py1 + 14,
                f"{i+1}",
                color=color,
                fontsize=10,
                bbox=dict(
                    facecolor="black", alpha=0.3, edgecolor="none", boxstyle="round,pad=0.2"
                ),
            )

    # 保存
    os.makedirs(out_dir, exist_ok=True)
    base = str(sample.get("id", "sample")).replace("/", "_")
    out_path = os.path.join(out_dir, f"{base}.jpg")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True, out_path


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.ids:
        ids = set(args.ids)
        data = [x for x in data if str(x.get("id", "")) in ids]
        if not data:
            print("未匹配到指定 --ids 样本，退出。")
            return

    total = len(data)
    print("总样本：", total)

    done = 0
    for ex in tqdm(data, desc="render"):
        if args.limit is not None and done >= args.limit:
            break
        ok, msg = draw_one(
            ex,
            args.img_root,
            args.out,
            dpi=args.dpi,
            wrap_width=args.wrap,
            max_height=args.max_height,
        )
        if not ok:
            print("跳过：", msg)
            continue
        done += 1
    print(f"完成，可视化输出 {done} 张，保存到：{args.out}")


if __name__ == "__main__":
    main()
