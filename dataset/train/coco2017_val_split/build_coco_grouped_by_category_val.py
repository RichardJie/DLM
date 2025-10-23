import os
import json
import random
from collections import defaultdict
from tqdm import tqdm
random.seed(42)

# ====== 1. 配置参数 (按需修改) ======

# --- 路径配置 ---
COCO_ROOT = "/hpc2ssd/JH_DATA/spooler/yuxuanzhao/lijungang/wujie/LLaDA-V/dataset/coco2017/val2017_split"

# 新增：标注来源 split（保持为 'val2017'，因为 train/ 和 test 都是从 val2017 切出来的）
ANN_SPLIT = "val2017"

# 原有变量改为“子集目录名”：'train' 或 'test'
SPLIT = "test"   # 需要哪个子集就填哪个：'train' 或 'test'

SAVE_DIR = os.path.join(COCO_ROOT, "llava_multi")
OUT_JSON = os.path.join(SAVE_DIR, f"coco_{SPLIT}_sft_final_v2_en.json")  # 文件名仍含 SPLIT，便于区分


# --- 数据生成配置 ---
PREC = 3  # 坐标小数位
SKIP_CROWD = True  # 过滤 iscrowd=1
NEGATIVE_SAMPLES_PER_IMAGE = 2  # 每张图生成多少个“硬负样本”

# ==============================================================================
# ====== 2. 多样化 Prompt 库 (全英文) ======
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

# --- 单类别查询 (Negative / Hard Negative) ---
NEGATIVE_PROMPTS = [
    "<image>\nAre there any {cat_key} in this image? If so, provide their bounding boxes.",
    "<image>\nFind all {cat_key} and list their coordinates.",
    "<image>\nPlease provide the locations of any {cat_key} present in the photo.",
]

# --- 拒绝回答的答案库 (全英文, 已修正) ---
NEGATIVE_ANSWERS = [
    "The image does not contain any {cat_key}.",
    "I couldn't find any {cat_key} in the provided image.",
    "There are no instances of {cat_key} in this picture.",
    "count: 0\nbbox_dict: {{}}",
]

# ==============================================================================
# ====== 3. 工具函数与数据加载 ======
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

# --- 路径查找与创建 ---
ANNO_DIR_1 = os.path.join(COCO_ROOT, "annotations")
ANNO_DIR_2 = os.path.join(ANNO_DIR_1, "annotations")
CAND_ANN = [
    os.path.join(ANNO_DIR_1, f"instances_{ANN_SPLIT}.json"),  # 改这里
    os.path.join(ANNO_DIR_2, f"instances_{ANN_SPLIT}.json"),  # 以及这里
]
ANN_FILE = next((p for p in CAND_ANN if os.path.exists(p)), None)
assert ANN_FILE, f"找不到 instances_{ANN_SPLIT}.json，请检查 {ANNO_DIR_1} 或 {ANNO_DIR_2}"


os.makedirs(SAVE_DIR, exist_ok=True)

# --- 加载与预处理 COCO 数据 ---
print("加载标注文件:", ANN_FILE)
with open(ANN_FILE, "r", encoding="utf-8") as f:
    coco = json.load(f)
# 读取子集目录下的图片文件名
SUBSET_DIR = os.path.join(COCO_ROOT, SPLIT)
assert os.path.isdir(SUBSET_DIR), f"子集目录不存在: {SUBSET_DIR}"
subset_fns = {
    fn for fn in os.listdir(SUBSET_DIR)
    if fn.lower().endswith((".jpg", ".jpeg", ".png"))
}
print(f"{SPLIT} 子集中发现图像文件数: {len(subset_fns)}")

print("预处理标注数据...")
id2img = {im["id"]: im for im in coco["images"]}
id2cat = {c["id"]: c for c in coco["categories"]}
all_cat_names = {c["name"] for c in coco["categories"]}

img2anns = defaultdict(list)
for ann in coco["annotations"]:
    if SKIP_CROWD and ann.get("iscrowd", 0) == 1:
        continue
    img2anns[ann["image_id"]].append(ann)

# ==============================================================================
# ====== 4. SFT 样本生成主逻辑 ======
# ==============================================================================
print("开始构建多样化SFT样本...")
final_items = []
stats = defaultdict(int)

for img_id, anns in tqdm(img2anns.items(), desc="Processing images"):
    img = id2img.get(img_id)
    if not img:
        stats['skipped_images'] += 1
        continue

    # 新增：只处理当前子集目录里存在的图片
    if img["file_name"] not in subset_fns:
        stats['not_in_subset'] += 1
        continue

    W, H = img["width"], img["height"]
    image_rel = os.path.join(SPLIT, img["file_name"])

    catname2boxes = defaultdict(list)
    for a in anns:
        cat_info = id2cat.get(a["category_id"])
        if not cat_info: continue
        cat_key = cat_info["name"]
        box = norm_box_xywh_to_xyxy_norm(*a["bbox"], W, H)
        catname2boxes[cat_key].append(box)

    if not catname2boxes:
        continue

    present_cats = set(catname2boxes.keys())

    # --- 1. 生成【多类别汇总】样本 (如果类别数 > 1) ---
    if len(present_cats) > 1:
        cats_list_str = ", ".join(sorted(list(present_cats))) #排序保证一致性
        prompt_template = random.choice(MULTI_CAT_PROMPTS)
        prompt = prompt_template.format(cats_list=cats_list_str)
        
        # **核心修正**: 构建 "counts: cat1(N), cat2(M)..." 格式
        count_breakdown = ", ".join([f"{cat}({len(boxes)})" for cat, boxes in sorted(catname2boxes.items())])
        answer_dict = {k: v for k, v in catname2boxes.items()}
        answer = f"counts: {count_breakdown}\nbbox_dict: {json.dumps(answer_dict, ensure_ascii=False)}"
        
        final_items.append({
            "id": f"sft-{SPLIT}-{img_id}-multi",
            "image": image_rel,
            "conversations": [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}],
        })
        stats['multi_category_samples'] += 1

    # --- 2. 为每个【存在的类别】生成独立的【单类别】样本 ---
    for cat_key, boxes in catname2boxes.items():
        prompt_template = random.choice(SINGLE_CAT_PROMPTS)
        prompt = prompt_template.format(cat_key=cat_key)
        
        count = len(boxes)
        answer_dict = {cat_key: boxes}
        # **核心修正**: 确保单类别答案也是全英文 "count: N"
        answer = f"count: {count}\nbbox_dict: {json.dumps(answer_dict, ensure_ascii=False)}"
        
        final_items.append({
            "id": f"sft-{SPLIT}-{img_id}-single-{cat_key.replace(' ', '_')}",
            "image": image_rel,
            "conversations": [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}],
        })
        stats['single_category_samples'] += 1

    # --- 3. 生成【硬负样本】(询问不存在的类别) ---
    absent_cats = all_cat_names - present_cats
    if absent_cats:
        num_to_sample = min(len(absent_cats), NEGATIVE_SAMPLES_PER_IMAGE)
        cats_to_ask = random.sample(list(absent_cats), num_to_sample)
        
        for neg_cat_key in cats_to_ask:
            prompt_template = random.choice(NEGATIVE_PROMPTS)
            prompt = prompt_template.format(cat_key=neg_cat_key)
            
            answer_template = random.choice(NEGATIVE_ANSWERS)
            answer = answer_template.format(cat_key=neg_cat_key)
            
            final_items.append({
                "id": f"sft-{SPLIT}-{img_id}-neg-{neg_cat_key.replace(' ', '_')}",
                "image": image_rel,
                "conversations": [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}],
            })
            stats['hard_negative_samples'] += 1

# ==============================================================================
# ====== 5. 结果汇总与保存 ======
# ==============================================================================
print("\n" + "="*50)
print("SFT 数据集生成完毕!")
print(f"总共处理图像: {len(img2anns)}")
print("\n生成样本统计:")
for key, value in stats.items():
    print(f"- {key.replace('_', ' ').title()}: {value}")
print(f"\n总计生成样本数: {len(final_items)}")
print("="*50)

print(f"\n正在将结果写入: {OUT_JSON}")
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(final_items, f, ensure_ascii=False, indent=2)

print("\n脚本执行成功!")
print("下一步:")
print("1. 使用此生成的 JSON 文件进行 SFT 训练。")
print("2. 确保您的 DPO 数据生成脚本与此处的【单类别正样本】(count: N\nbbox_dict: ...) 格式和 Prompt 库严格一致。")