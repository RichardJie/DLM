# build_llava_from_hf.py
from datasets import load_dataset
from PIL import Image
import os, json

OUT_ROOT = "/hpc2ssd/JH_DATA/spooler/yuxuanzhao/lijungang/wujie/dataset/textvqa_bbox_ms/llada/grounding_llava"         # 你想输出到哪
IMG_OUT  = os.path.join(OUT_ROOT, "images")
JSON_OUT = os.path.join(OUT_ROOT, "textvqa_bbox_llava.json")
os.makedirs(IMG_OUT, exist_ok=True)

ds = load_dataset("jrzhang/TextVQA_GT_bbox", split="train")  # 4,370 条
items = []

for i, ex in enumerate(ds, 1):
    # 字段兼容：HF 这版里是 answer(list), bbox(list[4]), image(Image)
    q = ex["question"]
    answers = ex.get("answer") or ex.get("answers") or []
    a = answers[0] if isinstance(answers, list) and answers else ""
    bbox = ex["bbox"]  # [x,y,w,h]，像素坐标
    x, y, w, h = bbox

    # 保存图像到本地
    img: Image.Image = ex["image"]  # datasets 会把远端图片拉到缓存
    W, H = img.size
    img_name = f"{i:06d}.jpg"
    img.save(os.path.join(IMG_OUT, img_name))

    # 像素 -> 归一化 [0,1]，三位小数
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    nx1 = round(max(0.0, min(1.0, x1 / W)), 3)
    ny1 = round(max(0.0, min(1.0, y1 / H)), 3)
    nx2 = round(max(0.0, min(1.0, x2 / W)), 3)
    ny2 = round(max(0.0, min(1.0, y2 / H)), 3)

    items.append({
        "id": f"textvqa-{i:06d}",
        "image": f"images/{img_name}",
        "conversations": [
            {"from": "human", "value": "<image>\nPlease locate and output the target coordinates of the problem on the entire graph, in the format [x1,y1,x2,y2] (normalized to 0-1, retaining three decimal places):{}".format(q)},
            {"from": "gpt",   "value": f"bbox: [{nx1:.3f}, {ny1:.3f}, {nx2:.3f}, {ny2:.3f}]"}
        ]
    })

with open(JSON_OUT, "w", encoding="utf-8") as f:
    json.dump(items, f, ensure_ascii=False, indent=2)

print("wrote", JSON_OUT, "with", len(items), "samples")
