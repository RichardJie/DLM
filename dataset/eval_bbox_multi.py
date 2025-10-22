# eval_bbox_multi.py (最终修正版, 兼容原始脚本框架)
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

# ==============================================================================
# ====== 核心修正: 升级解析函数以兼容新格式 ======
# ==============================================================================

def clamp_box(box: Tuple[float, ...]) -> Tuple[float, float, float, float]:
    """将坐标限制在 [0, 1] 范围内并修正顺序"""
    x1, y1, x2, y2 = box
    x1, y1 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))
    x2, y2 = max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    eps = 1e-6
    if x2 - x1 < eps: x2 = min(1.0, x1 + eps)
    if y2 - y1 < eps: y2 = min(1.0, y1 + eps)
    return float(x1), float(y1), float(x2), float(y2)

def parse_bbox_dict(text: str) -> Dict[str, List[Tuple[float, float, float, float]]]:
    """
    智能解析模型输出，兼容新旧两种格式:
    1. 新格式: 包含 'count' 或 'counts' 前缀
    2. 旧格式: 直接包含 "bbox_dict: {...}"
    3. 纯字典: "{...}"
    """
    if not text: return {}
    
    s = str(text)
    l_brace = s.find("{")
    r_brace = s.rfind("}")
    
    if l_brace == -1 or r_brace == -1 or l_brace > r_brace:
        return {}
        
    dict_str = s[l_brace : r_brace + 1]
    
    try:
        obj = ast.literal_eval(dict_str)
        if not isinstance(obj, dict):
            obj = json.loads(dict_str)
            if not isinstance(obj, dict): return {}
    except (ValueError, SyntaxError, json.JSONDecodeError):
        return {}

    out_dict: Dict[str, List[Tuple[float, float, float, float]]] = {}
    for k, v in obj.items():
        cat_key = str(k).lower().strip()
        boxes: List[Tuple[float, float, float, float]] = []
        if isinstance(v, list):
            for box_candidate in v:
                if isinstance(box_candidate, (list, tuple)) and len(box_candidate) == 4:
                    try:
                        float_box = tuple(float(coord) for coord in box_candidate)
                        boxes.append(clamp_box(float_box))
                    except (ValueError, TypeError):
                        continue
        if boxes:
            out_dict[cat_key] = boxes
            
    return out_dict

# ==============================================================================
# ====== 其他函数 (保持与您原始脚本一致) ======
# ==============================================================================

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
    if not preds and not gts:
        return [], set(), set()
    ious = []
    for i,p in enumerate(preds):
        for j,g in enumerate(gts):
            ious.append((iou(p,g), i, j))
    ious.sort(reverse=True, key=lambda x: x[0])
    used_p = set(); used_g = set(); matches = []
    for v, i, j in ious:
        if v < iou_thresh: break
        if i in used_p or j in used_g: continue
        used_p.add(i); used_g.add(j)
        matches.append((i,j,v))
    un_p = set(range(len(preds))) - used_p
    un_g = set(range(len(gts))) - used_g
    return matches, un_p, un_g

def _xyxy_from_norm(box_norm, W, H):
    x1, y1, x2, y2 = box_norm
    x1, y1 = int(round(x1 * W)), int(round(y1 * H))
    x2, y2 = int(round(x2 * W)), int(round(y2 * H))
    x1, y1 = max(0, min(W-1, x1)), max(0, min(H-1, y1))
    x2, y2 = max(0, min(W-1, x2)), max(0, min(H-1, y2))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2

def save_vis_multi(image: Image.Image,
                   pred_dict: Dict, gt_dict: Dict,
                   save_path: str, iou_thresh: float):
    im = image.copy()
    W, H = im.size
    dr = ImageDraw.Draw(im)
    try: font = ImageFont.load_default(size=15)
    except Exception: font = None
    
    for cat, boxes in gt_dict.items():
        for k, b in enumerate(boxes, 1):
            xyxy = _xyxy_from_norm(b, W, H)
            dr.rectangle(xyxy, outline=(0,255,0), width=4)
            dr.text((xyxy[0]+2, xyxy[1]+2), f"{cat} GT#{k}", fill=(0,255,0), font=font)
    for cat, boxes in pred_dict.items():
        for k, b in enumerate(boxes, 1):
            xyxy = _xyxy_from_norm(b, W, H)
            dr.rectangle(xyxy, outline=(255,0,0), width=3)
            dr.text((xyxy[0]+2, xyxy[1]-18), f"{cat} P#{k}", fill=(255,0,0), font=font)
            
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    im.save(save_path, quality=95)

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

# ==============================================================================
# ====== 主流程 (保持与您原始脚本一致的参数和调用) ======
# ==============================================================================

def main():
    # --- 参数解析 (完全恢复您原始脚本的参数) ---
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
    ap.add_argument("--prompt_version", default="llava_llada")
    ap.add_argument("--vis_dir", default="")
    ap.add_argument("--iou_thresh", type=float, default=0.50, help="贪心匹配的 IoU 阈值（默认 0.5）")
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype=="bf16" else torch.float16

    if args.lora_path:
        model_path = args.lora_path
        model_base = args.pretrained
    else:
        model_path = args.pretrained
        model_base = None

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, model_base, args.model_name,
        attn_implementation="sdpa", device_map=args.device,
        trust_remote_code=True
    )
    model.eval()

    if args.projector_bin and os.path.isfile(args.projector_bin):
        sd = torch.load(args.projector_bin, map_location="cpu")
        try: model.load_state_dict(sd, strict=False)
        except Exception: pass

    if args.use_fast_dllm and HAS_FAST:
        register_fast_dllm_hook(model)

    data = load_json(args.data_json)
    if args.max_samples and args.max_samples > 0:
        data = data[:args.max_samples]

    # --- 评估循环 (与您原始脚本完全一致) ---
    total_tp, total_fp, total_fn, total_gt, total_pred = 0, 0, 0, 0, 0
    iou_matched, l1_matched = [], []
    rows = []
    t0 = time.time()

    for ex in tqdm(data, ncols=100):
        img_rel = ex["image"]
        img_path = os.path.join(args.image_root, img_rel)
        user_msg = ex["conversations"][0]["value"]

        conv_key = args.prompt_version if args.prompt_version in conv_templates else "llava_llada"
        conv = copy.deepcopy(conv_templates[conv_key])
        conv.append_message(conv.roles[0], user_msg)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(img_path).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [_im.to(dtype=dtype, device=args.device) for _im in image_tensor]
        image_sizes = [image.size]

        # --- 生成调用 (恢复您原始脚本的调用方式) ---
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(args.device)
        with torch.inference_mode():
            out_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                steps=args.steps, gen_length=args.gen_length, block_length=args.block_length, tokenizer=tokenizer,
                stopping_criteria=['<|eot_id|>'],
                prefix_refresh_interval=args.prefix_refresh_interval,
                threshold=1,
            )
        pred_text = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
        # import pdb; pdb.set_trace()
        gt_text = ex["conversations"][1]["value"]
        gt_dict  = parse_bbox_dict(gt_text) # 使用升级后的函数
        pred_dict = parse_bbox_dict(pred_text) # 使用升级后的函数

        if not gt_dict: continue

        gt_count_img = sum(len(v) for v in gt_dict.values())
        pred_count_img = sum(len(v) for v in pred_dict.values())
        total_gt += gt_count_img
        total_pred += pred_count_img

        tp_img, fp_img, fn_img = 0, 0, 0
        ious_img, l1s_img = [], []

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

        precision = (tp_img/(tp_img+fp_img)) if (tp_img+fp_img)>0 else 0.0
        recall = (tp_img/(tp_img+fn_img)) if (tp_img+fn_img)>0 else 0.0
        mean_iou_matched = (np.mean(ious_img) if ious_img else 0.0)
        
        rows.append([
            ex.get("id", "N/A"), img_rel,
            gt_count_img, pred_count_img, tp_img, fp_img, fn_img,
            f"{precision:.4f}", f"{recall:.4f}", f"{mean_iou_matched:.4f}",
            pred_text.replace("\n"," ")[:2000]
        ])

        if args.vis_dir:
            save_path = os.path.join(args.vis_dir, f"{ex.get('id', os.path.basename(img_rel))}_vis.jpg")
            try: save_vis_multi(image, pred_dict, gt_dict, save_path, args.iou_thresh)
            except Exception: pass

    # --- 汇总与保存 (与您原始脚本一致) ---
    t1 = time.time()
    precision = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    recall    = (total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
    f1        = (2*precision*recall/(precision+recall)) if (precision+recall) > 0 else 0.0
    mean_iou  = float(np.mean(iou_matched)) if iou_matched else 0.0
    mean_l1   = float(np.mean(l1_matched)) if l1_matched else 0.0

    os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
    with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id","image","gt_boxes","pred_boxes","tp","fp","fn","precision","recall","mean_IoU_matched","raw_output"])
        w.writerows(rows)

    summary = {
        "images": len(rows),
        "total_gt_boxes": int(total_gt), "total_pred_boxes": int(total_pred),
        "tp": int(total_tp), "fp": int(total_fp), "fn": int(total_fn),
        "precision": round(precision,4), "recall": round(recall,4), "f1": round(f1,4),
        "mean_IoU_matched": round(mean_iou,4), "mean_L1_matched": round(mean_l1,4),
        "iou_thresh": args.iou_thresh, "time_sec": round(t1-t0,2),
        "pretrained": args.pretrained, "lora_path": args.lora_path
    }
    with open(args.save_sum, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n==== Summary ====")
    for k,v in summary.items(): print(f"{k}: {v}")
    print(f"\nDetails saved to: {args.save_csv}")
    print(f"Summary saved to: {args.save_sum}")
    if args.vis_dir: print(f"Visualizations saved to: {args.vis_dir}")

if __name__ == "__main__":
    main()