import os, json, glob
from PIL import Image
from tqdm import tqdm

DATA_ROOT = "/hpc2ssd/JH_DATA/spooler/yuxuanzhao/lijungang/wujie/LLaDA-V/dataset"
MS_DIR    = os.path.join(DATA_ROOT, "textvqa_bbox_ms")
OUT_DIR   = os.path.join(DATA_ROOT, "textvqa_bbox_coords_384")
OUT_IMG   = os.path.join(OUT_DIR, "images")          # 按你示例使用 images/
OUT_JSON  = os.path.join(OUT_DIR, "textvqa_bbox_coords_llava_384.json")
os.makedirs(OUT_IMG, exist_ok=True)

# 读取记录（优先 parquet；否则常见 json/jsonl）
records = []
parquets = glob.glob(os.path.join(MS_DIR, "**/*.parquet"), recursive=True)

if parquets:
    from datasets import load_dataset
    ds = load_dataset("parquet", data_files={"train": parquets})["train"]
    for ex in tqdm(ds, desc="loading parquet"):
        img = ex["image"]            # PIL.Image 或可转 PIL
        q   = ex.get("question", "")
        bbox= ex.get("bbox", None)   # 期望 [x,y,w,h] 像素
        if bbox and len(bbox)==4:
            records.append({"img_pil": img, "img_name": None, "q": q, "bbox": bbox})
else:
    cand = ["val.json", "data.json", "annotations.json", "dataset.json", "train.json", "train.jsonl"]
    src = next((os.path.join(MS_DIR,c) for c in cand if os.path.exists(os.path.join(MS_DIR,c))), None)
    assert src, f"未在 {MS_DIR} 找到 parquet 或常见 json 标注文件，请检查目录结构"
    if src.endswith(".jsonl"):
        lines = [json.loads(x) for x in open(src,"r",encoding="utf-8") if x.strip()]
    else:
        data = json.load(open(src,"r",encoding="utf-8"))
        lines = data if isinstance(data, list) else data.get("data", [])
    for ex in tqdm(lines, desc="loading json"):
        img_rel = ex.get("image") or ex.get("image_path")
        if not img_rel: continue
        img_path = img_rel if os.path.isabs(img_rel) else os.path.join(MS_DIR, img_rel)
        if not os.path.exists(img_path): continue
        img = Image.open(img_path).convert("RGB")
        q   = ex.get("question", "")
        bbox= ex.get("bbox", None)
        if bbox and len(bbox)==4:
            records.append({"img_pil": img, "img_name": os.path.basename(img_rel), "q": q, "bbox": bbox})

print(f"loaded {len(records)} samples")
assert len(records)>0, "没有加载到样本，请检查下载目录内容"

def clamp(v, lo, hi): 
    return max(lo, min(hi, v))

items=[]
for i, ex in enumerate(tqdm(records, desc="resizing & writing"), 1):
    im = ex["img_pil"].convert("RGB")
    W, H = im.size
    sx, sy = 384.0/W, 384.0/H

    # 1) resize 到 384×384
    im384 = im.resize((384,384), Image.BICUBIC)

    # 2) bbox: [x,y,w,h] (像素, 原图)
    x, y, w, h = ex["bbox"]
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h

    # 3) 映射到 384×384 像素坐标
    nx1 = x1 * sx; ny1 = y1 * sy
    nx2 = x2 * sx; ny2 = y2 * sy
    # 越界裁剪到像素框 [0,384]
    nx1 = clamp(nx1, 0.0, 384.0); ny1 = clamp(ny1, 0.0, 384.0)
    nx2 = clamp(nx2, 0.0, 384.0); ny2 = clamp(ny2, 0.0, 384.0)
    # 保证左上/右下有序且最小尺寸>0
    if nx2 < nx1: nx1, nx2 = nx2, nx1
    if ny2 < ny1: ny1, ny2 = ny2, ny1
    if nx2 - nx1 < 1e-6: nx2 = min(384.0, nx1 + 1.0)
    if ny2 - ny1 < 1e-6: ny2 = min(384.0, ny1 + 1.0)

    # 4) 归一化到 [0,1] 并保留三位小数
    def n3(v): return float(f"{(v/384.0):.3f}")
    bx1, by1, bx2, by2 = n3(nx1), n3(ny1), n3(nx2), n3(ny2)

    # 保存图片
    out_name = ex["img_name"] or f"{i:06d}.jpg"
    if not out_name.endswith(".jpg") and not out_name.endswith(".png"):
        out_name = f"{i:06d}.jpg"
    out_name = f"{i:06d}_{os.path.basename(out_name)}"
    im384.save(os.path.join(OUT_IMG, out_name), quality=95)

    # 生成你要的对话样本：gpt 输出就是 bbox 坐标字符串
    prompt = (
      "<image>\n"
      "Please locate and output the target coordinates of the problem on the entire image, "
      "in the format [x1,y1,x2,y2] (normalized to 0-1, retaining three decimal places):"
      f"{ex['q']}"
    )
    answer = f"bbox: [{bx1:.3f}, {by1:.3f}, {bx2:.3f}, {by2:.3f}]"

    items.append({
        "id": f"textvqa-{i:06d}",
        "image": f"images/{out_name}",
        "conversations": [
            {"from":"human","value": prompt},
            {"from":"gpt",  "value": answer}
        ],
        # 额外留存，便于可视化/检查
        "bbox_pixel_384": [nx1, ny1, nx2, ny2],
        "bbox_norm_384":  [bx1, by1, bx2, by2]
    })

os.makedirs(OUT_DIR, exist_ok=True)
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(items, f, ensure_ascii=False)
print("wrote", OUT_JSON, "with", len(items), "samples")
print("images dir:", OUT_IMG)
