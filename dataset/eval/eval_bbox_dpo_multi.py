# eval_bbox_multi.py
# 多类 × 多框评测：解析 "bbox_dict: {cat: [[x1,y1,x2,y2], ...], ...}"，
# 逐类做 IoU 贪心最大匹配，统计 TP/FP/FN/Precision/Recall/F1、mean IoU/L1，并可视化所有框。

import argparse, os, re, json, math, csv, time, copy, ast
from typing import Tuple, Optional, List, Dict, Any
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../train")))

from transformers.generation import stopping_criteria
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates

try:
    from llava.hooks.fast_dllm_hook import register_fast_dllm_hook, unregister_fast_dllm_hook
    HAS_FAST = True
except Exception:
    HAS_FAST = False

# ---------- 工具：单框解析（兼容旧数据） ----------
_PAT_SINGLE = re.compile(r'(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)', re.I)
_PAT_ALL_BOXES = re.compile(r'\[\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*\]')

def parse_any_boxes(text: str):
    if not text: return []
    out = []
    for m in _PAT_ALL_BOXES.finditer(text):
        x1, y1, x2, y2 = map(float, m.groups())
        out.append(clamp_box((x1, y1, x2, y2)))
    return out
def parse_bbox(text: str) -> Optional[Tuple[float,float,float,float]]:
    m = _PAT_SINGLE.search(text or "")
    if not m: return None
    x1,y1,x2,y2 = [float(m.group(i)) for i in range(1,5)]
    return clamp_box((x1,y1,x2,y2))

# ---------- 工具：字典解析（多类 × 多框） ----------
def norm_cat(name: str) -> str:
    if name is None: return ""
    s = str(name).strip().lower()
    s = s.replace("（","(").replace("）",")")
    return s

def clamp_box(box):
    x1,y1,x2,y2 = box
    x1 = max(0.0, min(1.0, x1)); y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2)); y2 = max(0.0, min(1.0, y2))
    if x2 < x1: x1,x2 = x2,x1
    if y2 < y1: y1,y2 = y2,y1
    eps = 1e-6
    if x2 - x1 < eps: x2 = min(1.0, x1 + eps)
    if y2 - y1 < eps: y2 = min(1.0, y1 + eps)
    return (float(x1), float(y1), float(x2), float(y2))

def parse_bbox_dict(text: str) -> Dict[str, List[Tuple[float,float,float,float]]]:
    """
    从字符串中提取 {...} 段并解析为 dict[str, list[box]]。
    允许前缀 "bbox_dict:"，以及单引号/空白等杂质。
    """
    if not text: return {}
    s = str(text)
    l = s.find("{"); r = s.rfind("}")
    if l == -1 or r == -1 or l > r:
        return {}
    inner = s[l:r+1]
    obj = None
    # 先试 JSON，再退回 ast.literal_eval
    try:
        obj = json.loads(inner)
    except Exception:
        try:
            obj = ast.literal_eval(inner)
        except Exception:
            return {}
    if not isinstance(obj, dict):
        return {}

    out: Dict[str, List[Tuple[float,float,float,float]]] = {}
    for k, v in obj.items():
        kk = norm_cat(k)
        boxes: List[Tuple[float,float,float,float]] = []
        if isinstance(v, list):
            for b in v:
                if isinstance(b, (list, tuple)) and len(b) == 4:
                    try:
                        boxes.append(clamp_box(tuple(float(x) for x in b)))
                    except Exception:
                        continue
        if boxes:
            out[kk] = boxes
    return out

# ---------- 指标 ----------
def iou(boxA, boxB) -> float:
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw * ih
    a = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    b = max(0.0, bx2-bx1) * max(0.0, by2-by1)
    union = a + b - inter + 1e-12
    return float(inter / union)

def l1_err(boxP, boxG) -> float:
    return float(sum(abs(p-g) for p,g in zip(boxP, boxG)) / 4.0)

def greedy_match(preds: List[Tuple[float,float,float,float]],
                 gts:   List[Tuple[float,float,float,float]],
                 iou_thresh: float):
    """
    计算所有 pred-gt 的 IoU，贪心取最大配对（>=阈值）直到没有可配。
    返回：
      matches: List[(pi, gi, iou)]
      unmatch_pred_idx: set
      unmatch_gt_idx: set
    """
    if not preds and not gts:
        return [], set(), set()
    ious = []
    for i,p in enumerate(preds):
        for j,g in enumerate(gts):
            ious.append((iou(p,g), i, j))
    ious.sort(reverse=True, key=lambda x: x[0])  # 大的优先
    used_p = set(); used_g = set(); matches = []
    for v, i, j in ious:
        if v < iou_thresh: break
        if i in used_p or j in used_g: continue
        used_p.add(i); used_g.add(j)
        matches.append((i,j,v))
    un_p = set(range(len(preds))) - used_p
    un_g = set(range(len(gts))) - used_g
    return matches, un_p, un_g

# ---------- 可视化（多框） ----------
def _xyxy_from_norm(box_norm, W, H):
    x1 = int(round(box_norm[0] * W))
    y1 = int(round(box_norm[1] * H))
    x2 = int(round(box_norm[2] * W))
    y2 = int(round(box_norm[3] * H))
    x1 = max(0, min(W-1, x1)); y1 = max(0, min(H-1, y1))
    x2 = max(0, min(W-1, x2)); y2 = max(0, min(H-1, y2))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2

def _draw_box(draw: ImageDraw.ImageDraw, xyxy, color, width=3):
    for w in range(width):
        draw.rectangle((xyxy[0]-w, xyxy[1]-w, xyxy[2]+w, xyxy[3]+w), outline=color)

def _put_text(draw, xy, text, color):
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text(xy, text, fill=color, font=font)

def save_vis_multi(image: Image.Image,
                   pred_dict: Dict[str, List[Tuple[float,float,float,float]]],
                   gt_dict:   Dict[str, List[Tuple[float,float,float,float]]],
                   save_path: str,
                   iou_thresh: float):
    im = image.copy()
    W, H = im.size
    dr = ImageDraw.Draw(im)
    # 先画 GT（绿），再画 Pred（红）
    for cat, boxes in gt_dict.items():
        for k, b in enumerate(boxes, 1):
            _draw_box(dr, _xyxy_from_norm(b, W, H), color=(0,255,0), width=4)
            _put_text(dr, (max(0, int(b[0]*W)+2), max(0, int(b[1]*H)+2)), f"{cat} GT#{k}", (0,255,0))
    for cat, boxes in pred_dict.items():
        for k, b in enumerate(boxes, 1):
            _draw_box(dr, _xyxy_from_norm(b, W, H), color=(255,0,0), width=3)
            _put_text(dr, (max(0, int(b[0]*W)+2), max(0, int(b[1]*H)-12)), f"{cat} P#{k}", (255,0,0))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    im.save(save_path, quality=95)

# ---------- 主流程 ----------
def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_json", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--pretrained", required=True, help="与 generate_demo 相同的 base 路径")
    ap.add_argument("--model_name", default="llava_llada")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16", choices=["bf16","fp16"])
    ap.add_argument("--projector_bin", default="", help="只训 projector 时必填")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--save_csv", default="eval_detail.csv")
    ap.add_argument("--save_sum", default="eval_summary.json")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--gen_length", type=int, default=128)
    ap.add_argument("--block_length", type=int, default=128)
    ap.add_argument("--prefix_refresh_interval", type=int, default=32)
    ap.add_argument("--use_fast_dllm", action="store_true")
    ap.add_argument("--lora_path", default="", help="LoRA 目录（含 adapter_config.json）")
    ap.add_argument("--lora_from_checkpoint", action="store_true")
    ap.add_argument("--prompt_version", default="llava_llada")
    ap.add_argument("--vis_dir", default="")
    ap.add_argument("--iou_thresh", type=float, default=0.50, help="贪心匹配的 IoU 阈值（默认 0.5）")
    
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype=="bf16" else torch.float16

    # 选择加载：LoRA 基座 or 纯基座
    if args.lora_path:
        model_path = args.lora_path
        model_base = args.pretrained
    else:
        model_path = args.pretrained
        model_base = None
    torch_dtype = "bfloat16" if args.dtype=="bf16" else "float16"
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, model_base, args.model_name,
        attn_implementation="sdpa",
        device_map="auto",             # 让 HF/accelerate 自己分配
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )
    model.eval()

    if args.projector_bin and os.path.isfile(args.projector_bin):
        sd = torch.load(args.projector_bin, map_location="cpu")
        allow = ("mm_projector", "multi_modal_projector", "vision_projector", "mm_mlp", "proj")
        sd = {k: v for k, v in sd.items() if any(t in k for t in allow)}
        # 确保前面已经 model.to(args.device)
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        if missing or unexpected:
            print("[projector] missing:", missing, "unexpected:", unexpected)

    if args.use_fast_dllm and HAS_FAST:
        register_fast_dllm_hook(model)

    data = load_json(args.data_json)
    if args.max_samples and args.max_samples>0:
        data = data[:args.max_samples]

    # 汇总统计
    total_imgs = 0
    total_gt = 0
    total_pred = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    iou_matched: List[float] = []
    l1_matched:  List[float] = []
    parse_fail = 0

    rows = []
    t0 = time.time()

    for ex in tqdm(data, ncols=100):
        img_rel = ex["image"]
        img_path = os.path.join(args.image_root, img_rel)
        user_msg = ex["conversations"][0]["value"]

        # prompt 构造
        conv_key = args.prompt_version if args.prompt_version in conv_templates else "llava_llada"
        conv = copy.deepcopy(conv_templates[conv_key])
        conv.append_message(conv.roles[0], user_msg)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # 图像处理
        image = Image.open(img_path).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [_im.to(dtype=dtype, device=args.device) for _im in image_tensor]
        image_sizes = [image.size]

        # 生成
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(args.device)
        out_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            steps=args.steps, gen_length=args.gen_length, block_length=args.block_length, tokenizer=tokenizer,
            stopping_criteria=['<|eot_id|>'],
            prefix_refresh_interval=args.prefix_refresh_interval,
            threshold=1,
        )
        pred_text = tokenizer.batch_decode(out_ids, skip_special_tokens=False)[0]
        import pdb; pdb.set_trace()
        # 解析 GT（优先从 assistant 的第二轮里解析 dict）
        # 你的 coco 多目标 JSON 中，assistant 的 value 是形如：bbox_dict: {...}
        gt_text = ex["conversations"][1]["value"]
        gt_dict  = parse_bbox_dict(gt_text)

        # 兜底：若 gt_dict 空（比如单框数据），尝试单框
        if not gt_dict:
            gt_single = parse_bbox(gt_text)
            if gt_single:
                gt_dict = {"object": [gt_single]}

        # 解析 Pred
        pred_text = tokenizer.batch_decode(out_ids, skip_special_tokens=False)[0]
        pred_dict = parse_bbox_dict(pred_text)

        # 兜底1：模型若给出单框
        if not pred_dict:
            p_single = parse_bbox(pred_text)
            if p_single:
                pred_dict = {"object": [p_single]}

        # 兜底2：模型若给出多框但无字典（DPO 风格 Count/Coordinates）
        if not pred_dict:
            all_boxes = parse_any_boxes(pred_text)
            if all_boxes:
                pred_dict = {"object": all_boxes}

        if not gt_dict:
            # 这张图没有可评估的 GT（极少见），跳过
            continue

        total_imgs += 1
        gt_count_img = sum(len(v) for v in gt_dict.values())
        pred_count_img = sum(len(v) for v in pred_dict.values())
        total_gt   += gt_count_img
        total_pred += pred_count_img

        # 逐类匹配
        tp_img = 0; fp_img = 0; fn_img = 0
        ious_img: List[float] = []
        l1s_img:  List[float] = []

        cats = set(list(gt_dict.keys()) + list(pred_dict.keys()))
        for cat in cats:
            gts = gt_dict.get(cat, [])
            prs = pred_dict.get(cat, [])
            matches, un_p, un_g = greedy_match(prs, gts, args.iou_thresh)
            tp_img += len(matches)
            fp_img += len(un_p)
            fn_img += len(un_g)
            for (pi, gi, v) in matches:
                ious_img.append(v)
                l1s_img.append(l1_err(prs[pi], gts[gi]))

        total_tp += tp_img
        total_fp += fp_img
        total_fn += fn_img
        iou_matched.extend(ious_img)
        l1_matched.extend(l1s_img)

        # 记录到 CSV
        rows.append([
            ex["id"], img_rel,
            gt_count_img, pred_count_img, tp_img, fp_img, fn_img,
            (tp_img/(tp_img+fp_img)) if (tp_img+fp_img)>0 else 0.0,  # precision
            (tp_img/(tp_img+fn_img)) if (tp_img+fn_img)>0 else 0.0,  # recall
            (np.mean(ious_img) if ious_img else 0.0),                 # mean IoU of matches
            pred_text.replace("\n"," ")[:2000]
        ])

        # 可视化
        if args.vis_dir:
            save_path = os.path.join(args.vis_dir, f"{ex['id']}_vis.jpg")
            try:
                save_vis_multi(image, pred_dict, gt_dict, save_path, args.iou_thresh)
            except Exception:
                pass

    t1 = time.time()

    precision = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    recall    = (total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
    f1        = (2*precision*recall/(precision+recall)) if (precision+recall) > 0 else 0.0
    mean_iou  = float(np.mean(iou_matched)) if iou_matched else 0.0
    mean_l1   = float(np.mean(l1_matched)) if l1_matched else 0.0

    os.makedirs(os.path.dirname(args.save_csv) or ".", exist_ok=True)
    with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id","image","gt_boxes","pred_boxes","tp","fp","fn","precision","recall","mean_IoU_matched","raw_output"])
        w.writerows(rows)

    summary = {
        "images": total_imgs,
        "total_gt_boxes": int(total_gt),
        "total_pred_boxes": int(total_pred),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "precision": round(precision,4),
        "recall": round(recall,4),
        "f1": round(f1,4),
        "mean_IoU_matched": round(mean_iou,4),
        "mean_L1_matched": round(mean_l1,4),
        "iou_thresh": args.iou_thresh,
        "time_sec": round(t1-t0,2),
        "pretrained": args.pretrained,
        "lora_path": args.lora_path
    }
    with open(args.save_sum, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("==== Summary ====")
    for k,v in summary.items():
        print(f"{k}: {v}")
    print(f"Details saved to: {args.save_csv}")
    print(f"Summary saved to: {args.save_sum}")
    if args.vis_dir:
        print(f"Visualizations saved to: {args.vis_dir}")

if __name__ == "__main__":
    main()
