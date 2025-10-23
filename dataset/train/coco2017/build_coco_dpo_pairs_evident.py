# -*- coding: utf-8 -*-
"""
构建 DPO 数据集（混合模式，强化小目标）：
1) 多类别：优先对“小目标”执行 drop/jitter 生成 (chosen, rejected)
2) 回退单类别：对该类部分框 jitter 生成 (chosen, rejected)
3) 证据对比：为含小目标的图像，生成 (同一答案, 但 image 与 image_neg 不同) 的证据对比样本

输出：逐行 JSONL，每行包含：
- image: 相对路径（如 "val2017/xxxxx.jpg"）
- prompt: 文本 prompt
- chosen: 文本答案（count + bbox_dict）
- rejected: 文本答案（count + bbox_dict）
- image_neg: 仅证据对比样本有；其余样本无该键
- meta: 记录样本类型/幅度等

注意：
- 与 SFT 的 prompt/answer 模板严格保持一致（坐标精度、键名稳定）
- SMALL_THR 等阈值与训练侧一致
- 证据对比样本需要 PIL；若无 pillow 请先 `pip install pillow`
"""

import os
import json
import random
import math
from collections import defaultdict
from tqdm import tqdm

# ============= 1. 配置参数（按需修改） =============

# 路径
COCO_ROOT = "/root/llada/dataset/coco2017"
SPLIT = "val2017"
SAVE_DIR = os.path.join(COCO_ROOT, "llava_multi")
OUT_DPO = os.path.join(SAVE_DIR, f"coco_{SPLIT}_dpo_evidenet.jsonl")

# 精度与小目标阈值
PREC = 3
SMALL_THR = 0.020            # 归一化面积阈值：< 该值视为小目标（与你先前一致）
SET_IOU_THR_FOR_VALIDITY = 0.95

# 生成策略
FALLBACK_TO_SINGLE_CAT_DPO = True
MAX_TRIES_PER_SAMPLE = 5

# 证据对比图像（退化图）相关
ENABLE_EVIDENCE_CONTRAST = True
EVI_LOCAL_DEGRADE = True     # 对所有小框局部像素化/模糊
EVI_GLOBAL_DEGRADE = True    # 再做一次全局降采样-上采样
PIXELATE_DOWN = 8            # 像素化强度
GAUSSIAN_BLUR_RADIUS = 3     # 模糊半径
GLOBAL_DEGRADE_FACTOR = 2    # 全局缩放因子（2=缩至1/2边长再放大）
EVI_DIR = os.path.join(SAVE_DIR, "evidence_degraded")  # 退化图保存目录（相对 SAVE_DIR）

# ============= 2. Prompt 库（和 SFT 完全一致） =============

SINGLE_CAT_PROMPTS = [
    "<image>\nPlease give the total number of {cat_key} in this image, then provide the bounding box coordinates for each {cat_key}.\nThe coordinate format is [x1,y1,x2,y2] (0-1 normalization, three decimal places);",
    "<image>\nFind all instances of {cat_key} in the picture. Output the total count and a dictionary with their bounding box coordinates.",
    "<image>\nHow many {cat_key} are there and where are they located? Provide the answer as a count followed by a bbox_dict.",
    "<image>\nI need the locations of every {cat_key}. Please respond with the number of instances and a JSON-like dictionary `bbox_dict`.",
]

MULTI_CAT_PROMPTS = [
    "<image>\nPlease give the coordinates of {cats_list}. For each category, specify its count. Output a final dictionary for all coordinates.",
    "<image>\nDetect the following objects: {cats_list}. For all of them, provide a count for each class and a single `bbox_dict` for all boxes.",
    "<image>\nGenerate a report for the objects: {cats_list}. The report should contain per-category counts and a `bbox_dict` with their locations.",
]

# ============= 3. 基础工具函数 =============

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

def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1e-12, ua + ub - inter)

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

# ============= 4. 证据对比：退化图工具（PIL） =============

try:
    from PIL import Image, ImageFilter
    PIL_OK = True
except Exception:
    PIL_OK = False
    print("[Warn] pillow 未安装，证据对比图像将被跳过（ENABLE_EVIDENCE_CONTRAST=False）")
    ENABLE_EVIDENCE_CONTRAST = False

def load_img(path): 
    return Image.open(path).convert("RGB")

def save_img(img, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path, quality=95)

def gaussian_blur_region(img, box, radius=GAUSSIAN_BLUR_RADIUS):
    W, H = img.size
    x1, y1 = int(box[0]*W), int(box[1]*H)
    x2, y2 = int(box[2]*W), int(box[3]*H)
    crop = img.crop((x1, y1, x2, y2)).filter(ImageFilter.GaussianBlur(radius=radius))
    img_p = img.copy()
    img_p.paste(crop, (x1, y1, x2, y2))
    return img_p

def pixelate_region(img, box, down=PIXELATE_DOWN):
    W, H = img.size
    x1, y1 = int(box[0]*W), int(box[1]*H)
    x2, y2 = int(box[2]*W), int(box[3]*H)
    crop = img.crop((x1, y1, x2, y2))
    w, h = max(1, x2-x1), max(1, y2-y1)
    crop_small = crop.resize((max(1,w//down), max(1,h//down)), Image.NEAREST)
    crop_pix = crop_small.resize((w, h), Image.NEAREST)
    img_p = img.copy()
    img_paste_box = (x1, y1, x1 + w, y1 + h)
    img_p.paste(crop_pix, img_paste_box)
    return img_p

def global_lowres_degrade(img, factor=GLOBAL_DEGRADE_FACTOR):
    W, H = img.size
    img_small = img.resize((max(1, W//factor), max(1, H//factor)), Image.BICUBIC)
    return img_small.resize((W, H), Image.BICUBIC)

# ============= 5. COCO 标注加载与预处理 =============

ANNO_DIR_1 = os.path.join(COCO_ROOT, "annotations")
ANNO_DIR_2 = os.path.join(ANNO_DIR_1, "annotations")
CAND_ANN = [os.path.join(d, f"instances_{SPLIT}.json") for d in [ANNO_DIR_1, ANNO_DIR_2] if os.path.exists(os.path.join(d, f"instances_{SPLIT}.json"))]
assert CAND_ANN, f"找不到 instances_{SPLIT}.json"
ANN_FILE = CAND_ANN[0]
os.makedirs(SAVE_DIR, exist_ok=True)
if ENABLE_EVIDENCE_CONTRAST:
    os.makedirs(EVI_DIR, exist_ok=True)

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
    cat_name = id2cat.get(a["category_id"])
    if not cat_name: 
        continue
    box = norm_box_xywh_to_xyxy_norm(*a["bbox"], img["width"], img["height"])
    img2cat2boxes[a["image_id"]][cat_name].append(box)

# ============= 6. DPO 文本构造 =============

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
        return catname2boxes_rej, {"neg_mode": "drop_small_in_multi", "target": target_cat}
    if mode == "jitter":
        amp = random.uniform(0.08, 0.20)
        target_boxes[target_idx] = jitter_box(target_boxes[target_idx], amp)
        return catname2boxes_rej, {"neg_mode": "jitter_small_in_multi", "target": target_cat, "amp": round(amp, 3)}
    return None, None

# ============= 7. 主循环：混合模式 + 证据对比 =============

print("开始构建混合模式 DPO 样本...")
final_items = []
stats = defaultdict(int)

for img_id, cat2boxes in tqdm(img2cat2boxes.items(), desc="Processing images"):
    im = id2img.get(img_id)
    if not im or not cat2boxes:
        continue

    image_rel = os.path.join(SPLIT, im['file_name'])

    # ----------- A) 多类别优先：针对小目标构造负样本 -----------
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

    # ----------- B) 单类别回退：任意 jitter 若干框 -----------
    if FALLBACK_TO_SINGLE_CAT_DPO and not dpo_pair_generated_for_image_multi:
        for cat, boxes in cat2boxes.items():
            if not boxes: 
                continue
            # 小框加大 jitter 幅度，大框较小幅度
            amp_base = random.uniform(0.06, 0.12)
            amp_small = random.uniform(0.15, 0.30)
            rej_boxes = [b[:] for b in boxes]
            n_jitter = min(len(boxes), random.choice([1, 1, 2]))
            idxs_to_jitter = random.sample(range(len(boxes)), n_jitter)
            for i in idxs_to_jitter:
                amp = amp_small if area_norm(rej_boxes[i]) < SMALL_THR else amp_base
                rej_boxes[i] = jitter_box(rej_boxes[i], amp)

            # 有效性检查（避免 chosen/rejected 太相似）
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

    # ----------- C) 证据对比样本（仅对含小目标图像） -----------
    if ENABLE_EVIDENCE_CONTRAST and PIL_OK:
        small_boxes_all = [b for _, boxes in cat2boxes.items() for b in boxes if area_norm(b) < SMALL_THR]
        if len(small_boxes_all) > 0:
            orig_img_path = os.path.join(COCO_ROOT, SPLIT, im['file_name'])
            try:
                img = load_img(orig_img_path)

                img_deg = img.copy()
                if EVI_LOCAL_DEGRADE:
                    # 依次对小框区域像素化或模糊（默认像素化）
                    for sb in small_boxes_all:
                        img_deg = pixelate_region(img_deg, sb, down=PIXELATE_DOWN)
                        # 或者使用高斯模糊： img_deg = gaussian_blur_region(img_deg, sb, radius=GAUSSIAN_BLUR_RADIUS)
                if EVI_GLOBAL_DEGRADE:
                    img_deg = global_lowres_degrade(img_deg, factor=GLOBAL_DEGRADE_FACTOR)

                deg_rel = os.path.join("evidence_degraded", SPLIT, im['file_name'])
                deg_abs = os.path.join(SAVE_DIR, deg_rel)
                save_img(img_deg, deg_abs)

                cats_list_str = ", ".join(sorted(cat2boxes.keys()))
                prompt_template = random.choice(MULTI_CAT_PROMPTS)
                prompt = prompt_template.format(cats_list=cats_list_str)

                item_evi = {
                    "image": image_rel,                 # 原图
                    "image_neg": deg_rel,               # 退化图（相对 SAVE_DIR）
                    "prompt": prompt,
                    "chosen": ans_from_multi_cat(cat2boxes),  # 同一个答案文本
                    "rejected": ans_from_multi_cat(cat2boxes),
                    "meta": {
                        "img_id": img_id,
                        "type": "evidence_contrast_small",
                        "num_small_boxes": len(small_boxes_all),
                        "small_area_sum": sum(area_norm(b) for b in small_boxes_all)
                    }
                }
                final_items.append(item_evi)
                stats['evidence_contrast_samples'] += 1
            except Exception as e:
                # 出错则跳过证据对比，不影响其它样本
                stats['evidence_contrast_errors'] += 1

# ============= 8. 统计与保存 =============

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
print("下一步：")
print("1) 若已按建议改了 Collator/Trainer，可利用 `image_neg` 做证据对比 DPO；")
print("2) 否则先用现有字段训练（忽略 `image_neg`），等训练链路改好再启用。")
