#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build SFT data (English prompt/answer) from COCO2017 *val2017_split* layout,
with strict two-line answers and a global cap: singles ≤ SINGLE_MAX_RATIO * total.

Save JSON to: dataset/datasets/coco2017/val2017_split/llava_multi/
Read images from: dataset/datasets/coco2017/val2017_split/{train|test}/
Read annotations from: dataset/datasets/coco2017/val2017_split/annotations[/annotations]/instances_val2017.json
"""

import os, json, random, hashlib, math
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm

# =========================
# --- 路径配置 / Paths ---
# =========================

# 根目录：按你的环境修改
COCO_ROOT = "/root/llada/dataset/datasets/coco2017/val2017_split"

# 标注固定来自 val2017（你是把 val2017 切成 train/test）
ANN_SPLIT = "val2017"

# 只改这个就能在 train/test 之间切换
SPLIT = "test"   # or "test"

# 输出
SAVE_DIR = os.path.join(COCO_ROOT, "llava_multi")
OUT_JSON = os.path.join(SAVE_DIR, f"coco_{SPLIT}_sft_version2.json")

# =========================
# --- 数据生成配置 / Knobs
# =========================

PREC = 3                         # 坐标小数位
SKIP_CROWD = True                # 过滤 iscrowd=1
NEGATIVE_SAMPLES_PER_IMAGE = 1   # 每图负样本数
SINGLE_MAX_RATIO = 0.30          # **single（正+负）在全集中的最大占比**

PROMPT_DIVERSITY = True
PROMPT_SEED_SALT  = "dlm-grounding-v1"
random.seed(1234)

# =========================
# --- Prompt 模板池 ---
# =========================

MULTI_CAT_PROMPTS = [
    "<image>\nPlease give the coordinates of {cats_list}. For each category, specify its count. Output a final dictionary for all coordinates.",
    "<image>\nDetect the following objects: {cats_list}. For all of them, provide a count for each class and a single `bbox_dict` for all boxes.",
    "<image>\nGenerate a report for the objects: {cats_list}. The report should contain per-category counts and a `bbox_dict` with their locations.",
]

SINGLE_CAT_PROMPTS = [
    "<image>\nPlease give the total number of {cat_key} in this image, then provide the bounding box coordinates for each {cat_key}.\nThe coordinate format is [x1,y1,x2,y2] (0-1 normalization, three decimal places);",
    "<image>\nFind all instances of {cat_key} in the picture. Output the total count and a dictionary with their bounding box coordinates.",
    "<image>\nHow many {cat_key} are there and where are they located? Provide the answer as a count followed by a bbox_dict.",
    "<image>\nI need the locations of every {cat_key}. Please respond with the number of instances and a JSON-like dictionary `bbox_dict`.",
]

NEGATIVE_PROMPTS = [
    "<image>\nAre there any {cat_key} in this image? If so, provide their bounding boxes.",
    "<image>\nFind all {cat_key} and list their coordinates.",
    "<image>\nPlease provide the locations of any {cat_key} present in the photo.",
]

# 负样本答案：为了一致可学性，**总是结构化**（避免自然语言）
NEGATIVE_STRUCT_ANSWER = "count: 0\nbbox_dict: {}"

# =========================
# --- Helpers -------------
# =========================

def _pick(pool, salt: str) -> str:
    if not PROMPT_DIVERSITY:
        return pool[0]
    h = hashlib.md5((PROMPT_SEED_SALT + "|" + salt).encode("utf-8")).hexdigest()
    return pool[int(h, 16) % len(pool)]

def nfmt(x: float, prec=PREC) -> float:
    return float(f"{x:.{prec}f}")

def norm_box_xywh_to_xyxy_norm(x, y, w, h, W, H):
    x1 = max(0.0, min(1.0, x / W))
    y1 = max(0.0, min(1.0, y / H))
    x2 = max(0.0, min(1.0, (x + w) / W))
    y2 = max(0.0, min(1.0, (y + h) / H))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    eps = 10**(-PREC)
    if x2 - x1 < eps: x2 = min(1.0, x1 + eps)
    if y2 - y1 < eps: y2 = min(1.0, y1 + eps)
    return [nfmt(x1), nfmt(y1), nfmt(x2), nfmt(y2)]

def _box_sort_key(b):
    x1,y1,x2,y2 = b
    return (x1, y1, x2 - x1, y2 - y1)

def _sort_and_dedup_boxes(boxes: List[List[float]]) -> List[List[float]]:
    seen = set()
    uniq = []
    for b in sorted(boxes, key=_box_sort_key):
        rb = tuple(b)  # 已经保留 PREC 位
        if rb in seen:
            continue
        seen.add(rb); uniq.append(list(rb))
    return uniq

def _counts_str(cat2boxes: Dict[str, List]) -> str:
    keys = sorted(cat2boxes.keys())
    return "counts: " + ", ".join([f"{k}({len(cat2boxes[k])})" for k in keys])

def _bbox_dict_str_ordered(cat2boxes: Dict[str, List[List[float]]]) -> str:
    keys = sorted(cat2boxes.keys())
    ordered = {k: cat2boxes[k] for k in keys}
    return "bbox_dict: " + json.dumps(ordered, ensure_ascii=False)

def load_coco():
    # 适配两种 annotations 目录层级
    a1 = os.path.join(COCO_ROOT, "annotations", f"instances_{ANN_SPLIT}.json")
    a2 = os.path.join(COCO_ROOT, "annotations", "annotations", f"instances_{ANN_SPLIT}.json")
    ann_file = a1 if os.path.isfile(a1) else a2
    if not os.path.isfile(ann_file):
        raise FileNotFoundError(f"instances_{ANN_SPLIT}.json not found:\n  {a1}\n  {a2}")
    with open(ann_file, "r", encoding="utf-8") as f:
        coco = json.load(f)
    id2img = {im["id"]: im for im in coco["images"]}
    id2cat = {c["id"]: c for c in coco["categories"]}
    img2anns = defaultdict(list)
    for ann in coco["annotations"]:
        if SKIP_CROWD and ann.get("iscrowd", 0) == 1:
            continue
        img2anns[ann["image_id"]].append(ann)
    return ann_file, id2img, id2cat, img2anns

# 只替换我们允许的占位符，避免 .format 吃掉 JSON 花括号
def fill_tpl(tpl: str, **kwargs) -> str:
    for k, v in kwargs.items():
        tpl = tpl.replace("{" + k + "}", v)
    return tpl

# =========================
# --- 主流程 / Main -------
# =========================

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    _, id2img, id2cat, img2anns = load_coco()

    subset_dir = os.path.join(COCO_ROOT, SPLIT)
    if not os.path.isdir(subset_dir):
        raise FileNotFoundError(f"Subset dir not found: {subset_dir}")
    subset_fns = {
        fn for fn in os.listdir(subset_dir)
        if fn.lower().endswith((".jpg", ".jpeg", ".png"))
    }

    all_cat_names = {c["name"] for c in id2cat.values()}

    # 先分别累积，再统一合并以便做 single 全局截断
    items_multi = []
    items_single_pos = []
    items_single_neg = []
    stats = defaultdict(int)

    for img_id, anns in tqdm(img2anns.items(), ncols=100, desc=f"Build {SPLIT}"):
        im = id2img.get(img_id)
        if not im:
            stats["skip_no_image_meta"] += 1
            continue
        fn = im["file_name"]
        if fn not in subset_fns:
            stats["skip_not_in_subset"] += 1
            continue

        W, H = im["width"], im["height"]
        image_rel = os.path.join(SPLIT, fn)

        # 聚合该图 GT
        cat2boxes = defaultdict(list)  # type: Dict[str, List[List[float]]]
        for a in anns:
            cat = id2cat.get(a["category_id"])
            if not cat:
                continue
            cat2boxes[cat["name"]].append(norm_box_xywh_to_xyxy_norm(*a["bbox"], W, H))

        if not cat2boxes:
            stats["skip_no_boxes"] += 1
            continue

        # 类内排序 + 去重（按 PREC）
        for k in list(cat2boxes.keys()):
            cat2boxes[k] = _sort_and_dedup_boxes(cat2boxes[k])

        present_cats = sorted(cat2boxes.keys())

        # ---- Multi（≥2 类）----
        if len(present_cats) >= 2:
            user_tpl = _pick(MULTI_CAT_PROMPTS, fn)
            user = fill_tpl(user_tpl, cats_list=", ".join(present_cats))
            ans  = _counts_str(cat2boxes) + "\n" + _bbox_dict_str_ordered(cat2boxes)
            items_multi.append({
                "id": f"sft-{SPLIT}-{img_id}-multi",
                "image": image_rel,
                "conversations": [
                    {"from": "human", "value": user},
                    {"from": "gpt",   "value": ans},
                ],
            })
            stats["multi"] += 1

        # ---- Single 正样本（先全量生成，稍后统一截断占比）----
        for c in present_cats:
            user_tpl = _pick(SINGLE_CAT_PROMPTS, f"{fn}|{c}")
            user = fill_tpl(user_tpl, cat=c, cat_key=c)
            ans  = f"count: {len(cat2boxes[c])}\n" + _bbox_dict_str_ordered({c: cat2boxes[c]})
            items_single_pos.append({
                "id": f"sft-{SPLIT}-{img_id}-single-{c.replace(' ','_')}",
                "image": image_rel,
                "conversations": [
                    {"from": "human", "value": user},
                    {"from": "gpt",   "value": ans},
                ],
            })

        # ---- Single 负样本（结构化负例）----
        absent = list(all_cat_names - set(present_cats))
        if NEGATIVE_SAMPLES_PER_IMAGE > 0 and absent:
            k = min(NEGATIVE_SAMPLES_PER_IMAGE, len(absent))
            for neg_cat in random.sample(absent, k=k):
                user_tpl = _pick(NEGATIVE_PROMPTS, f"{fn}|NEG|{neg_cat}")
                user = fill_tpl(user_tpl, cat=neg_cat, cat_key=neg_cat)
                ans  = NEGATIVE_STRUCT_ANSWER
                items_single_neg.append({
                    "id": f"sft-{SPLIT}-{img_id}-neg-{neg_cat.replace(' ','_')}",
                    "image": image_rel,
                    "conversations": [
                        {"from": "human", "value": user},
                        {"from": "gpt",   "value": ans},
                    ],
                })

    # === 统一控制 single 占比（正+负一起截断到 SINGLE_MAX_RATIO）===
    n_multi = len(items_multi)
    n_single_pos = len(items_single_pos)
    n_single_neg = len(items_single_neg)

    # ✅ 这里的 non_single 只能是 n_multi（负样本已归入 singles 池）
    non_single = n_multi

    if SINGLE_MAX_RATIO >= 1.0:
        max_single_allowed = len(items_single_pos) + len(items_single_neg)
    else:
        # single_total ≤ r/(1-r) * non_single
        max_single_allowed = int(math.floor((SINGLE_MAX_RATIO / (1.0 - SINGLE_MAX_RATIO)) * non_single))

    singles_all = items_single_pos + items_single_neg
    if len(singles_all) > max_single_allowed:
        random.seed(1234)
        singles_all = random.sample(singles_all, max_single_allowed)

    # 只包含 multi +（抽样后的）single
    final_items = items_multi + singles_all

    random.seed(1234)
    random.shuffle(final_items)


    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_items, f, ensure_ascii=False, indent=2)

    # === 打印摘要 ===
    kept_single = len(singles_all)
    total = len(final_items)
    kept_pos = sum(1 for it in singles_all if "-neg-" not in it["id"])
    kept_neg = len(singles_all) - kept_pos
    print(f"#single_pos(kept): {kept_pos}")
    print(f"#single_neg(kept): {kept_neg}")
    print("\n=== SFT data built ===")
    print(f"COCO_ROOT      : {COCO_ROOT}")
    print(f"ANN_SPLIT      : {ANN_SPLIT}")
    print(f"SPLIT          : {SPLIT}")
    print(f"Images dir     : {os.path.join(COCO_ROOT, SPLIT)}")
    print(f"Output json    : {OUT_JSON}")
    print(f"#multi         : {n_multi}")
    print(f"#single_pos(all): {n_single_pos}")
    print(f"#single_neg(all): {n_single_neg}")
    print(f"#single_kept    : {kept_single}  (cap={max_single_allowed})")
    print(f"#total          : {total}")
    print(f"single_ratio    : {kept_single/total:.3f}")
    print("Done.")

if __name__ == "__main__":
    main()
