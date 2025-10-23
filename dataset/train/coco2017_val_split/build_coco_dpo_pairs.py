import os
import json
import random
from collections import defaultdict
from tqdm import tqdm

# ==============================================================================
# ====== 1. 配置参数 (应与 SFT 脚本保持一致) ======
# ==============================================================================

# --- 路径配置 ---
# 注意：这里建议直接指向 val2017_split（包含 train/ 与 test/）
COCO_ROOT = "/hpc2ssd/JH_DATA/spooler/yuxuanzhao/lijungang/wujie/LLaDA-V/dataset/coco2017/val2017_split"

# 标注来源 split（保持为 'val2017'，因为 train/test 都是从 val2017 切出来的）
ANN_SPLIT = "val2017"

# “图片子集目录名”：'train' 或 'test'
SPLIT = "train"

SAVE_DIR = os.path.join(COCO_ROOT, "llava_multi")
OUT_DPO = os.path.join(SAVE_DIR, f"coco_{SPLIT}_dpo_final_mixed_v2_en.jsonl")  # 区分 train/test

# --- 数据生成配置 ---
PREC = 3
SMALL_THR = 0.020
FALLBACK_TO_SINGLE_CAT_DPO = True
MAX_TRIES_PER_SAMPLE = 5
SET_IOU_THR_FOR_VALIDITY = 0.95

# （可选）固定随机种子，保证硬负样本与抖动可复现
# random.seed(42)

# ==============================================================================
# ====== 2. 多样化 Prompt 库 (与 SFT 脚本完全一致) ======
# ==============================================================================

# --- 单类别查询 (Positive) ---
SINGLE_CAT_PROMPTS = [
    "<image>\nPlease give the total number of {cat_key} in this image, then provide the bounding box coordinates for each {cat_key}.\nThe coordinate format is [x1,y1,x2,y2] (0-1 normalization, three decimal places);",
    "<image>\nFind all instances of {cat_key} in the picture. Output the total count and a dictionary with their bounding box coordinates.",
    "<image>\nHow many {cat_key} are there and where are they located? Provide the answer as a count followed by a bbox_dict.",
    "<image>\nI need the locations of every {cat_key}. Please respond with the number of instances and a JSON-like dictionary `bbox_dict`.",
]

# --- 多类别查询 (Positive) ---
MULTI_CAT_PROMPTS = [
    "<image>\nPlease give the coordinates of {cats_list}. For each category, specify its count. Output a final dictionary for all coordinates.",
    "<image>\nDetect the following objects: {cats_list}. For all of them, provide a count for each class and a single `bbox_dict` for all boxes.",
    "<image>\nGenerate a report for the objects: {cats_list}. The report should contain per-category counts and a `bbox_dict` with their locations.",
]

# ==============================================================================
# ====== 3. 工具函数与数据加载 ======
# ==============================================================================

def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aw = max(0.0, ax2 - ax1); ah = max(0.0, ay2 - ay1)
    bw = max(0.0, bx2 - bx1); bh = max(0.0, by2 - by1)
    union = aw*ah + bw*bh - inter + 1e-12
    return inter / union

def mean_set_iou(A, B):
    A = A[:]; B = B[:]
    used = [False]*len(B)
    s = 0.0; m = 0
    for a in A:
        best, bestj = -1, -1
        for j, b in enumerate(B):
            if used[j]: continue
            v = iou(a, b)
            if v > best: best, bestj = v, j
        if bestj >= 0:
            used[bestj] = True
            s += best; m += 1
    return s / max(1, m)

def rounded_equal_boxes(A, B, prec=PREC):
    def rbox(b): return [float(f"{x:.{prec}f}") for x in b]
    return sorted([rbox(x) for x in A]) == sorted([rbox(x) for x in B])

def pair_is_valid(chosen_boxes, rejected_boxes, prec=PREC, set_iou_thr=SET_IOU_THR_FOR_VALIDITY):
    if len(chosen_boxes) != len(rejected_boxes): return True
    if not rounded_equal_boxes(chosen_boxes, rejected_boxes, prec=prec): return True
    if mean_set_iou(chosen_boxes, rejected_boxes) < set_iou_thr: return True
    return False

def nfmt(x, prec=PREC):
    return float(f"{x:.{prec}f}")

def norm_box_xywh_to_xyxy_norm(x, y, w, h, W, H):
    x1, y1 = max(0.0, min(1.0, x / W)), max(0.0, min(1.0, y / H))
    x2, y2 = max(0.0, min(1.0, (x + w) / W)), max(0.0, min(1.0, (y + h) / H))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    eps = 1e-6
    if x2 - x1 < eps: x2 = min(1.0, x1 + eps)
    if y2 - y1 < eps: y2 = min(1.0, y1 + eps)
    return [nfmt(x1), nfmt(y1), nfmt(x2), nfmt(y2)]

def area_norm(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

def jitter_box(b, amp):
    x1, y1, x2, y2 = b
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = random.uniform(-amp, amp) * w, random.uniform(-amp, amp) * h
    sw, sh = 1.0 + random.uniform(-amp, amp), 1.0 + random.uniform(-amp, amp)
    nw, nh = max(1e-6, w * sw), max(1e-6, h * sh)
    nx1 = max(0.0, min(1.0, cx - nw / 2 + dx)); ny1 = max(0.0, min(1.0, cy - nh / 2 + dy))
    nx2 = max(0.0, min(1.0, nx1 + nw)); ny2 = max(0.0, min(1.0, ny1 + nh))
    return [nfmt(nx1), nfmt(ny1), nfmt(nx2), nfmt(ny2)]

# --- 路径查找与创建 ---
ANNO_DIR_1 = os.path.join(COCO_ROOT, "annotations")
ANNO_DIR_2 = os.path.join(ANNO_DIR_1, "annotations")
CAND_ANN = [
    os.path.join(d, f"instances_{ANN_SPLIT}.json")
    for d in [ANNO_DIR_1, ANNO_DIR_2]
    if os.path.exists(os.path.join(d, f"instances_{ANN_SPLIT}.json"))
]
assert CAND_ANN, f"找不到 instances_{ANN_SPLIT}.json"
ANN_FILE = CAND_ANN[0]
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 子集目录与文件名集合（关键新增） ---
SUBSET_DIR = os.path.join(COCO_ROOT, SPLIT)
assert os.path.isdir(SUBSET_DIR), f"子集目录不存在: {SUBSET_DIR}"
subset_fns = {
    fn for fn in os.listdir(SUBSET_DIR)
    if fn.lower().endswith((".jpg", ".jpeg", ".png"))
}
print(f"{SPLIT} 子集中发现图像文件数: {len(subset_fns)}")

# --- 加载与预处理 COCO 数据 ---
print("加载标注文件:", ANN_FILE)
with open(ANN_FILE, "r", encoding="utf-8") as f:
    coco = json.load(f)

print("预处理标注数据...")
id2img = {im["id"]: im for im in coco["images"]}
id2cat = {c["id"]: c["name"] for c in coco["categories"]}

img2cat2boxes = defaultdict(lambda: defaultdict(list))
for a in coco["annotations"]:
    if a.get("iscrowd", 0) == 1:
        continue
    img = id2img.get(a["image_id"])
    if not img:
        continue
    # 只保留当前子集中的图片
    if img["file_name"] not in subset_fns:
        continue
    cat_name = id2cat.get(a["category_id"])
    if not cat_name:
        continue
    box = norm_box_xywh_to_xyxy_norm(*a["bbox"], img["width"], img["height"])
    img2cat2boxes[a["image_id"]][cat_name].append(box)

# ==============================================================================
# ====== 4. DPO 核心函数 (为混合模式重构) ======
# ==============================================================================

def ans_from_single_cat(boxes, cat_name):
    count = len(boxes)
    answer_dict = {cat_name: boxes}
    return f"count: {count}\nbbox_dict: {json.dumps(answer_dict, ensure_ascii=False)}"

def ans_from_multi_cat(catname2boxes):
    sorted_items = sorted(catname2boxes.items())
    count_breakdown = ", ".join([f"{cat}({len(boxes)})" for cat, boxes in sorted_items])
    sorted_dict = {k: v for k, v in sorted_items}
    return f"counts: {count_breakdown}\nbbox_dict: {json.dumps(sorted_dict, ensure_ascii=False)}"

def build_rejected_multi_cat(catname2boxes_orig, target_cat, target_idx, mode):
    catname2boxes_rej = {k: [b[:] for b in v] for k, v in catname2boxes_orig.items()}
    target_boxes = catname2boxes_rej[target_cat]
    if mode == "drop":
        del target_boxes[target_idx]
        if not target_boxes:
            del catname2boxes_rej[target_cat]
        return catname2boxes_rej, {"neg_mode": f"drop_small_in_multi", "target": target_cat}
    if mode == "jitter":
        amp = random.uniform(0.08, 0.20)
        target_boxes[target_idx] = jitter_box(target_boxes[target_idx], amp)
        return catname2boxes_rej, {"neg_mode": f"jitter_small_in_multi", "target": target_cat, "amp": round(amp, 3)}
    return None, None

# ==============================================================================
# ====== 5. DPO 样本生成主逻辑 (混合模式) ======
# ==============================================================================
print("开始构建混合模式DPO样本...")
final_items = []
stats = defaultdict(int)

for img_id, cat2boxes in tqdm(img2cat2boxes.items(), desc="Processing images"):
    im = id2img.get(img_id)
    if not im:
        stats["skipped_images"] += 1
        continue

    # 只处理当前子集的图片
    if im["file_name"] not in subset_fns:
        stats["not_in_subset"] += 1
        continue

    image_rel = os.path.join(SPLIT, im["file_name"])
    if not cat2boxes:
        continue

    dpo_pair_generated_for_image_multi = False

    small_objects_pool = []
    if len(cat2boxes) > 1:
        for cat, boxes in cat2boxes.items():
            for i, box in enumerate(boxes):
                if area_norm(box) < SMALL_THR:
                    small_objects_pool.append({'cat': cat, 'idx': i})

    if small_objects_pool:
        target_small_obj = random.choice(small_objects_pool)
        target_cat, target_idx = target_small_obj['cat'], target_small_obj['idx']
        rejection_mode = "drop" if random.random() < 0.8 else "jitter"
        chosen_dict = cat2boxes
        rejected_dict, neg_meta = build_rejected_multi_cat(cat2boxes, target_cat, target_idx, rejection_mode)
        if rejected_dict:
            cats_list_str = ", ".join(sorted(chosen_dict.keys()))
            prompt_template = random.choice(MULTI_CAT_PROMPTS)
            prompt = prompt_template.format(cats_list=cats_list_str)
            item = {
                "image": image_rel,
                "prompt": prompt,
                "chosen": ans_from_multi_cat(chosen_dict),
                "rejected": ans_from_multi_cat(rejected_dict),
                "meta": {"img_id": img_id, "type": "multi_object_small_focus", **neg_meta}
            }
            final_items.append(item)
            stats['multi_object_dpo_samples'] += 1
            dpo_pair_generated_for_image_multi = True

    if FALLBACK_TO_SINGLE_CAT_DPO and not dpo_pair_generated_for_image_multi:
        for cat, boxes in cat2boxes.items():
            if not boxes:
                continue
            amp = random.uniform(0.06, 0.15)
            n_jitter = min(len(boxes), random.choice([1, 1, 2]))
            idxs_to_jitter = random.sample(range(len(boxes)), n_jitter)
            rej_boxes = [b[:] for b in boxes]
            for i in idxs_to_jitter:
                rej_boxes[i] = jitter_box(rej_boxes[i], amp)
            # 有效性检查：必须是“不同集合”
            if pair_is_valid(boxes, rej_boxes):
                prompt_template = random.choice(SINGLE_CAT_PROMPTS)
                prompt = prompt_template.format(cat_key=cat)
                item = {
                    "image": image_rel,
                    "prompt": prompt,
                    "chosen": ans_from_single_cat(boxes, cat),
                    "rejected": ans_from_single_cat(rej_boxes, cat),
                    "meta": {"img_id": img_id, "type": "single_object_fallback", "neg_mode": "jitter_any"}
                }
                final_items.append(item)
                stats['single_object_dpo_samples'] += 1

# ==============================================================================
# ====== 6. 结果汇总与保存 ======
# ==============================================================================
print("\n" + "="*50)
print("DPO 数据集生成完毕!")
print("\n生成样本统计:")
for key, value in stats.items():
    print(f"- {key.replace('_', ' ').title()}: {value}")
print(f"\n总计生成 DPO 样本数: {len(final_items)}")
print("="*50)

print(f"\n正在将结果写入: {OUT_DPO}")
with open(OUT_DPO, "w", encoding="utf-8") as f:
    for item in final_items:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("\n脚本执行成功!")
print("下一步: 使用这个混合模式的 DPO 数据集，配合您最终版的 SFT 模型，进行 DPO 训练。")
