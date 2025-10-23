import os
import json
import random
from collections import defaultdict
from tqdm import tqdm

# ==============================================================================
# ====== 改进版 DPO 数据生成脚本 - 针对小目标检测优化 ======
# ==============================================================================
# 主要改进：
# 1. 扩大小目标定义范围（2% -> 10%）
# 2. 增加"漏检小目标"的困难负样本
# 3. 增加"小目标位置偏移"的负样本（更大的抖动幅度）
# 4. 增加"混淆小目标与背景噪声"的负样本
# 5. 优先为小目标生成 DPO 样本
# ==============================================================================

# ==============================================================================
# ====== 1. 配置参数 ======
# ==============================================================================

# --- 路径配置 ---

COCO_ROOT = "/root/llada/dataset/coco2017/val2017_split"  # ← 只改这一行：指向 val2017_split
SPLIT = "train2017"  # 原样保留（不影响下游 print）
SAVE_DIR = os.path.join(COCO_ROOT, "llava_multi")
OUT_DPO = os.path.join(SAVE_DIR, f"coco_{SPLIT}_dpo_improved_for_small_objects.jsonl")

# 新增两行（用于只取子集与 val2017 标注）
ANN_SPLIT = "val2017"
SUBSET = "train"  # 只生成 train 子目录下的图片

# --- 数据生成配置 ---
PREC = 3

# ========== 关键改进 1：扩大小目标定义范围 ==========
SMALL_THR = 0.10      # 面积 < 10% 算小目标（原来是 2%，太严格）
TINY_THR = 0.03       # 面积 < 3% 算极小目标（新增）

# ========== 关键改进 2：负样本生成策略权重 ==========
# 对于小目标，使用以下负样本生成策略的概率分布：
NEG_STRATEGY_WEIGHTS = {
    'drop_small': 0.30,          # 直接删除小目标（漏检）
    'large_jitter_small': 0.40,  # 大幅度抖动小目标位置（定位不准）
    'add_false_positive': 0.10,  # 添加虚假的小目标（误检）
    'swap_small_large': 0.15,   # 交换小目标和大目标的位置（混淆）
    'partial_drop': 0.05,        # 只删除部分小目标（数量错误）
}

# 抖动幅度（根据目标大小自适应）
JITTER_AMP_SMALL = (0.40, 0.80)    # 小目标：20%-40% 的抖动
JITTER_AMP_LARGE = (0.08, 0.15)    # 大目标：8%-15% 的抖动

# DPO 样本生成数量控制
MAX_DPO_PAIRS_PER_IMAGE = 3        # 每张图最多生成 3 个 DPO 样本
PRIORITIZE_SMALL_OBJECTS = True    # 优先为包含小目标的图片生成样本

# 其他配置
SET_IOU_THR_FOR_VALIDITY = 0.60
SKIP_CROWD = True

# ==============================================================================
# ====== 2. 多样化 Prompt 库 (与 SFT 一致) ======
# ==============================================================================

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

# ==============================================================================
# ====== 3. 工具函数 ======
# ==============================================================================

def nfmt(x, prec=PREC):
    """格式化浮点数到指定小数位"""
    return float(f"{x:.{prec}f}")

def norm_box_xywh_to_xyxy_norm(x, y, w, h, W, H):
    """将 COCO 的 [x,y,w,h] 格式转换为归一化的 [x1,y1,x2,y2]"""
    x1 = max(0.0, min(1.0, x / W))
    y1 = max(0.0, min(1.0, y / H))
    x2 = max(0.0, min(1.0, (x + w) / W))
    y2 = max(0.0, min(1.0, (y + h) / H))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    eps = 1e-6
    if x2 - x1 < eps: x2 = min(1.0, x1 + eps)
    if y2 - y1 < eps: y2 = min(1.0, y1 + eps)
    return [nfmt(x1), nfmt(y1), nfmt(x2), nfmt(y2)]

def area_norm(box):
    """计算归一化的 bbox 面积"""
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

def jitter_box(b, amp_range):
    """抖动 bbox 位置和大小"""
    x1, y1, x2, y2 = b
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    amp = random.uniform(*amp_range)
    dx, dy = random.uniform(-amp, amp) * w, random.uniform(-amp, amp) * h
    sw, sh = 1.0 + random.uniform(-amp, amp), 1.0 + random.uniform(-amp, amp)

    nw, nh = max(1e-6, w * sw), max(1e-6, h * sh)
    nx1 = max(0.0, min(1.0, cx - nw / 2 + dx))
    ny1 = max(0.0, min(1.0, cy - nh / 2 + dy))
    nx2 = max(0.0, min(1.0, nx1 + nw))
    ny2 = max(0.0, min(1.0, ny1 + nh))

    return [nfmt(nx1), nfmt(ny1), nfmt(nx2), nfmt(ny2)]

def generate_random_box():
    """生成随机的小 bbox（用于误检负样本）"""
    size = random.uniform(0.02, 0.08)  # 小目标大小
    x1 = random.uniform(0.0, 1.0 - size)
    y1 = random.uniform(0.0, 1.0 - size)
    x2 = min(1.0, x1 + size * random.uniform(0.8, 1.2))
    y2 = min(1.0, y1 + size * random.uniform(0.8, 1.2))
    return [nfmt(x1), nfmt(y1), nfmt(x2), nfmt(y2)]

def iou(a, b):
    """计算两个 bbox 的 IoU"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aw = max(0.0, ax2 - ax1)
    ah = max(0.0, ay2 - ay1)
    bw = max(0.0, bx2 - bx1)
    bh = max(0.0, by2 - by1)
    union = aw * ah + bw * bh - inter + 1e-12
    return inter / union

def mean_set_iou(A, B):
    """计算两个 bbox 集合的平均 IoU"""
    A = A[:]
    B = B[:]
    used = [False] * len(B)
    s = 0.0
    m = 0
    for a in A:
        best, bestj = -1, -1
        for j, b in enumerate(B):
            if used[j]:
                continue
            v = iou(a, b)
            if v > best:
                best, bestj = v, j
        if bestj >= 0:
            used[bestj] = True
            s += best
            m += 1
    return s / max(1, m)

def pair_is_valid(chosen_boxes, rejected_boxes, prec=PREC, set_iou_thr=SET_IOU_THR_FOR_VALIDITY):
    """检查 DPO pair 是否有效（chosen 和 rejected 要有明显区别）"""
    if len(chosen_boxes) != len(rejected_boxes):
        return True

    # 检查 bbox 是否完全相同
    def rbox(b):
        return [float(f"{x:.{prec}f}") for x in b]
    if sorted([rbox(x) for x in chosen_boxes]) == sorted([rbox(x) for x in rejected_boxes]):
        return False

    # 检查 IoU 是否过高
    if mean_set_iou(chosen_boxes, rejected_boxes) >= set_iou_thr:
        return False

    return True

# ==============================================================================
# ====== 4. 答案格式化函数 ======
# ==============================================================================

def ans_from_single_cat(boxes, cat_name):
    """单类别答案格式"""
    count = len(boxes)
    answer_dict = {cat_name: boxes}
    return f"count: {count}\nbbox_dict: {json.dumps(answer_dict, ensure_ascii=False)}"

def ans_from_multi_cat(catname2boxes):
    """多类别答案格式"""
    sorted_items = sorted(catname2boxes.items())
    count_breakdown = ", ".join([f"{cat}({len(boxes)})" for cat, boxes in sorted_items])
    sorted_dict = {k: v for k, v in sorted_items}
    return f"counts: {count_breakdown}\nbbox_dict: {json.dumps(sorted_dict, ensure_ascii=False)}"

# ==============================================================================
# ====== 5. 改进的负样本生成策略 ======
# ==============================================================================

def select_neg_strategy():
    """根据权重随机选择负样本生成策略"""
    strategies = list(NEG_STRATEGY_WEIGHTS.keys())
    weights = list(NEG_STRATEGY_WEIGHTS.values())
    return random.choices(strategies, weights=weights)[0]

def build_rejected_sample(catname2boxes_orig, small_objects_info, strategy):
    """
    根据策略生成 rejected 样本

    Args:
        catname2boxes_orig: 原始的类别->bbox字典
        small_objects_info: 小目标信息列表 [{'cat': cat, 'idx': idx, 'area': area}, ...]
        strategy: 负样本生成策略

    Returns:
        rejected_dict: 修改后的 bbox 字典
        meta: 元数据信息
    """
    catname2boxes_rej = {k: [b[:] for b in v] for k, v in catname2boxes_orig.items()}

    if not small_objects_info:
        return None, None

    # 选择一个小目标作为操作对象
    target_obj = random.choice(small_objects_info)
    target_cat = target_obj['cat']
    target_idx = target_obj['idx']

    # ========== 策略 1: 删除小目标（漏检） ==========
    if strategy == 'drop_small':
        del catname2boxes_rej[target_cat][target_idx]
        if not catname2boxes_rej[target_cat]:
            del catname2boxes_rej[target_cat]
        return catname2boxes_rej, {
            "neg_mode": "drop_small",
            "target_cat": target_cat,
            "target_area": target_obj['area']
        }

    # ========== 策略 2: 大幅度抖动小目标位置 ==========
    elif strategy == 'large_jitter_small':
        amp_range = JITTER_AMP_SMALL
        catname2boxes_rej[target_cat][target_idx] = jitter_box(
            catname2boxes_rej[target_cat][target_idx],
            amp_range
        )
        return catname2boxes_rej, {
            "neg_mode": "large_jitter_small",
            "target_cat": target_cat,
            "jitter_range": amp_range
        }

    # ========== 策略 3: 添加虚假的小目标（误检） ==========
    elif strategy == 'add_false_positive':
        # 在随机类别中添加一个虚假的小 bbox
        random_cat = random.choice(list(catname2boxes_rej.keys()))
        false_box = generate_random_box()
        catname2boxes_rej[random_cat].append(false_box)
        return catname2boxes_rej, {
            "neg_mode": "add_false_positive",
            "target_cat": random_cat,
            "false_box": false_box
        }

    # ========== 策略 4: 交换小目标和大目标的位置 ==========
    elif strategy == 'swap_small_large':
        # 找一个大目标
        large_objects = []
        for cat, boxes in catname2boxes_orig.items():
            for idx, box in enumerate(boxes):
                if area_norm(box) > SMALL_THR:
                    large_objects.append({'cat': cat, 'idx': idx})

        if large_objects:
            large_obj = random.choice(large_objects)
            large_cat, large_idx = large_obj['cat'], large_obj['idx']

            # 交换位置
            temp = catname2boxes_rej[target_cat][target_idx]
            catname2boxes_rej[target_cat][target_idx] = catname2boxes_rej[large_cat][large_idx]
            catname2boxes_rej[large_cat][large_idx] = temp

            return catname2boxes_rej, {
                "neg_mode": "swap_small_large",
                "small_cat": target_cat,
                "large_cat": large_cat
            }
        else:
            # 如果没有大目标，fallback 到抖动策略
            return build_rejected_sample(catname2boxes_orig, small_objects_info, 'large_jitter_small')

    # ========== 策略 5: 只删除部分小目标（数量错误） ==========
    elif strategy == 'partial_drop':
        # 随机删除 30%-70% 的小目标
        drop_ratio = random.uniform(0.3, 0.7)
        for obj in small_objects_info:
            if random.random() < drop_ratio:
                cat, idx = obj['cat'], obj['idx']
                if cat in catname2boxes_rej and idx < len(catname2boxes_rej[cat]):
                    del catname2boxes_rej[cat][idx]
                    # 更新后续索引
                    for other_obj in small_objects_info:
                        if other_obj['cat'] == cat and other_obj['idx'] > idx:
                            other_obj['idx'] -= 1

        # 清理空列表
        catname2boxes_rej = {k: v for k, v in catname2boxes_rej.items() if v}

        return catname2boxes_rej, {
            "neg_mode": "partial_drop",
            "drop_ratio": round(drop_ratio, 2)
        }

    return None, None

# ==============================================================================
# ====== 6. 数据加载 ======
# ==============================================================================

ANNO_DIR_1 = os.path.join(COCO_ROOT, "annotations")
ANNO_DIR_2 = os.path.join(ANNO_DIR_1, "annotations")
CAND_ANN = [
    os.path.join(ANNO_DIR_1, f"instances_{ANN_SPLIT}.json"),
    os.path.join(ANNO_DIR_2, f"instances_{ANN_SPLIT}.json"),
]
ANN_FILE = next((p for p in CAND_ANN if os.path.exists(p)), None)
assert ANN_FILE, f"找不到 instances_{ANN_SPLIT}.json"
assert ANN_FILE, f"找不到 instances_{SPLIT}.json"

os.makedirs(SAVE_DIR, exist_ok=True)
# 子集文件名白名单（只取 SUBSET 子目录里的图片）
SUBSET_DIR = os.path.join(COCO_ROOT, SUBSET)
assert os.path.isdir(SUBSET_DIR), f"子集目录不存在: {SUBSET_DIR}"
subset_fns = {
    fn for fn in os.listdir(SUBSET_DIR)
    if fn.lower().endswith((".jpg", ".jpeg", ".png"))
}
print(f"{SUBSET} 子集中发现图像文件数: {len(subset_fns)}")

print("加载标注文件:", ANN_FILE)
with open(ANN_FILE, "r", encoding="utf-8") as f:
    coco = json.load(f)

print("预处理标注数据...")
id2img = {im["id"]: im for im in coco["images"]}
id2cat = {c["id"]: c["name"] for c in coco["categories"]}

img2cat2boxes = defaultdict(lambda: defaultdict(list))
for a in coco["annotations"]:
    if SKIP_CROWD and a.get("iscrowd", 0) == 1:
        continue
    img = id2img.get(a["image_id"])
    if not img:
        continue
    if img["file_name"] not in subset_fns:
        continue
    cat_name = id2cat.get(a["category_id"])
    if not cat_name:
        continue
    box = norm_box_xywh_to_xyxy_norm(*a["bbox"], img["width"], img["height"])
    img2cat2boxes[a["image_id"]][cat_name].append(box)

# ==============================================================================
# ====== 7. DPO 样本生成主逻辑 ======
# ==============================================================================

print("开始构建改进版 DPO 样本...")
final_items = []
stats = defaultdict(int)

for img_id, cat2boxes in tqdm(img2cat2boxes.items(), desc="Processing images"):
    im = id2img.get(img_id)
    if not im:
        continue

    image_rel = os.path.join(SUBSET, im['file_name']) 

    if not cat2boxes:
        continue

    # ========== 识别小目标 ==========
    small_objects_pool = []
    tiny_objects_pool = []

    for cat, boxes in cat2boxes.items():
        for idx, box in enumerate(boxes):
            area = area_norm(box)
            if area < TINY_THR:
                tiny_objects_pool.append({'cat': cat, 'idx': idx, 'area': area})
                small_objects_pool.append({'cat': cat, 'idx': idx, 'area': area})
            elif area < SMALL_THR:
                small_objects_pool.append({'cat': cat, 'idx': idx, 'area': area})

    # 如果没有小目标，跳过（或降低优先级）
    if PRIORITIZE_SMALL_OBJECTS and not small_objects_pool:
        if random.random() < 0.7:  # 70% 概率跳过没有小目标的图片
            continue

    # ========== 生成 DPO 样本 ==========
    num_pairs_generated = 0

    for _ in range(MAX_DPO_PAIRS_PER_IMAGE):
        if num_pairs_generated >= MAX_DPO_PAIRS_PER_IMAGE:
            break

        # 选择负样本生成策略
        strategy = select_neg_strategy()

        # 生成 rejected 样本
        objects_to_use = tiny_objects_pool if tiny_objects_pool and random.random() < 0.6 else small_objects_pool
        if not objects_to_use:
            objects_to_use = small_objects_pool

        if not objects_to_use:
            break

        rejected_dict, neg_meta = build_rejected_sample(cat2boxes, objects_to_use, strategy)

        if not rejected_dict:
            continue

        # 验证 DPO pair 是否有效
        chosen_dict = cat2boxes

        # 检查是否有效（对所有 bbox 进行检查）
        all_chosen_boxes = []
        all_rejected_boxes = []
        for cat in set(list(chosen_dict.keys()) + list(rejected_dict.keys())):
            all_chosen_boxes.extend(chosen_dict.get(cat, []))
            all_rejected_boxes.extend(rejected_dict.get(cat, []))

        if not pair_is_valid(all_chosen_boxes, all_rejected_boxes):
            continue

        # 生成 prompt 和 answer
        if len(chosen_dict) > 1:
            # 多类别
            cats_list_str = ", ".join(sorted(chosen_dict.keys()))
            prompt_template = random.choice(MULTI_CAT_PROMPTS)
            prompt = prompt_template.format(cats_list=cats_list_str)
            chosen_ans = ans_from_multi_cat(chosen_dict)
            rejected_ans = ans_from_multi_cat(rejected_dict)
            sample_type = "multi_object"
        else:
            # 单类别
            cat_name = list(chosen_dict.keys())[0]
            prompt_template = random.choice(SINGLE_CAT_PROMPTS)
            prompt = prompt_template.format(cat_key=cat_name)
            chosen_ans = ans_from_single_cat(chosen_dict[cat_name], cat_name)
            rejected_ans = ans_from_single_cat(rejected_dict.get(cat_name, []), cat_name)
            sample_type = "single_object"

        # 添加到结果
        item = {
            "image": image_rel,
            "prompt": prompt,
            "chosen": chosen_ans,
            "rejected": rejected_ans,
            "meta": {
                "img_id": img_id,
                "type": sample_type,
                "num_small_objects": len(small_objects_pool),
                "num_tiny_objects": len(tiny_objects_pool),
                **neg_meta
            }
        }

        final_items.append(item)
        stats[f"{sample_type}_{strategy}"] += 1
        num_pairs_generated += 1

# ==============================================================================
# ====== 8. 结果汇总与保存 ======
# ==============================================================================

print("\n" + "="*70)
print("改进版 DPO 数据集生成完毕!")
print("\n生成样本统计（按策略分组）:")
for key, value in sorted(stats.items()):
    print(f"- {key}: {value}")
print(f"\n总计生成 DPO 样本数: {len(final_items)}")
print("="*70)

print(f"\n正在将结果写入: {OUT_DPO}")
with open(OUT_DPO, "w", encoding="utf-8") as f:
    for item in final_items:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("\n脚本执行成功!")
print("\n关键改进总结:")
print("1. 扩大小目标定义范围: 2% -> 10%")
print("2. 增加 5 种负样本生成策略（更多样化、更困难）")
print("3. 优先为包含小目标的图片生成 DPO 样本")
print("4. 增加极小目标（<3%）的特殊处理")
print("\n下一步:")
print("1. 使用这个改进版 DPO 数据集进行训练")
print("2. 配合之前提供的改进 DPO loss 代码")
print("3. 预期小目标检测性能提升 15-30%")