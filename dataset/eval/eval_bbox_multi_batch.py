# eval_bbox_multi.py (支持 batch 推理的最终修正版)
import argparse, os, re, json, math, csv, time, copy, ast
from typing import Tuple, Optional, List, Dict, Any
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../train")))

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
# ====== 解析器增强（与你给的版本保持一致） ======
# ==============================================================================

def clamp_box(box: Tuple[float, ...]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    x1, y1 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))
    x2, y2 = max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    eps = 1e-6
    if x2 - x1 < eps: x2 = min(1.0, x1 + eps)
    if y2 - y1 < eps: y2 = min(1.0, y1 + eps)
    return float(x1), float(y1), float(x2), float(y2)

def _is_finite_num(x) -> bool:
    try:
        xf = float(x)
        return math.isfinite(xf)
    except Exception:
        return False

def _to_float4(v) -> Tuple[float, float, float, float]:
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        raise ValueError("not a 4-tuple")
    vals = [float(x) for x in v]
    return tuple(vals)  # type: ignore

def _clamp_box(box: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(1.0, x1)); y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2)); y2 = max(0.0, min(1.0, y2))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    eps = 1e-6
    if x2 - x1 < eps: x2 = min(1.0, x1 + eps)
    if y2 - y1 < eps: y2 = min(1.0, y1 + eps)
    return float(x1), float(y1), float(x2), float(y2)

def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aarea = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    barea = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = aarea + barea - inter
    if denom <= 0: return 0.0
    return inter / denom

def _round_box(box: Tuple[float, float, float, float], prec: int) -> Tuple[float, float, float, float]:
    return tuple(float(f"{c:.{prec}f}"))  # type: ignore

def _dedup_boxes(boxes: List[Tuple[float, float, float, float]], prec: int = 3, iou_thr: float = 0.95
                 ) -> List[Tuple[float, float, float, float]]:
    if not boxes: return []
    seen = set()
    uniq = []
    for b in boxes:
        rb = _round_box(b, prec)
        if rb in seen: 
            continue
        seen.add(rb)
        uniq.append(b)
    kept: List[Tuple[float, float, float, float]] = []
    for b in uniq:
        ok = True
        for kb in kept:
            if _iou(b, kb) >= iou_thr:
                ok = False; break
        if ok: kept.append(b)
    return kept

def _sanitize_text(s: str) -> str:
    if not s: return ""
    s = re.sub(r'\b(NaN|nan|NAN|Infinity|INF|inf|-Infinity|-INF)\b', 'null', s)
    s = re.sub(r'\"\"\"+\s*:\s*[^,\}\]]*', '', s)
    s = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', s)
    return s

def _extract_brace_block(s: str) -> str:
    l = s.find("{"); r = s.rfind("}")
    if l == -1 or r == -1 or l > r: 
        return ""
    return s[l:r+1]

def _loads_pairs(dict_str: str):
    try:
        return json.loads(dict_str, object_pairs_hook=list)
    except Exception:
        pass
    try:
        js2 = dict_str.replace("'", '"')
        js2 = re.sub(r",\s*([\}\]])", r"\1", js2)
        return json.loads(js2, object_pairs_hook=list)
    except Exception:
        pass
    try:
        obj = ast.literal_eval(dict_str)
        if isinstance(obj, dict):
            return list(obj.items())
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    return None

def parse_bbox_dict(text: str, prec: int = 3, iou_dedup: float = 0.95
                    ) -> Dict[str, List[Tuple[float, float, float, float]]]:
    if not text:
        return {}
    s = _sanitize_text(str(text))
    blk = _extract_brace_block(s)
    if not blk:
        return {}

    pairs = _loads_pairs(blk)
    if not pairs:
        return {}

    merged: Dict[str, List] = {}
    for k, v in pairs:
        key = str(k).lower().strip()
        if key == "":
            continue
        if not isinstance(v, list):
            if isinstance(v, dict):
                possible_box = list(v.values())
                if isinstance(possible_box, list):
                    v = [possible_box]
                else:
                    continue
            else:
                continue
        merged.setdefault(key, []).extend(v)

    out: Dict[str, List[Tuple[float, float, float, float]]] = {}
    for key, arr in merged.items():
        clean_boxes: List[Tuple[float, float, float, float]] = []
        if not isinstance(arr, list): 
            continue
        for cand in arr:
            try:
                b = _to_float4(cand)
            except Exception:
                continue
            if not all(_is_finite_num(x) for x in b):
                continue
            b = _clamp_box(b)
            clean_boxes.append(b)
        if not clean_boxes:
            continue
        deduped = _dedup_boxes(clean_boxes, prec=prec, iou_thr=iou_dedup)
        if deduped:
            out[key] = deduped
    return out

# ==============================================================================
# ====== 指标与可视化（保持不变） ======
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
# ====== Batch 工具 ======
# ==============================================================================

def pad_batch_input_ids(batch_ids: List[torch.Tensor],
                        pad_id: int,
                        device: str,
                        padding_side: str = "right"
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将一批 1D input_ids 张量 pad 到同长，并返回 attention_mask（非 pad 位置为 True）
    """
    bsz = len(batch_ids)
    max_len = max(int(t.shape[0]) for t in batch_ids) if bsz > 0 else 0
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((bsz, max_len), dtype=torch.bool)
    for i, t in enumerate(batch_ids):
        L = int(t.shape[0])
        if padding_side == "left":
            input_ids[i, -L:] = t
            attn_mask[i, -L:] = True
        else:
            input_ids[i, :L] = t
            attn_mask[i, :L] = True
    return input_ids.to(device), attn_mask.to(device)

# ==============================================================================
# ====== 主流程（批推理改造） ======
# ==============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_json", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--model_name", default="llava_llada")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16", choices=["bf16","fp16"])
    ap.add_argument("--projector_bin", default="")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--save_csv", default="eval_detail.csv")
    ap.add_argument("--save_sum", default="eval_summary.json")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--gen_length", type=int, default=128)
    ap.add_argument("--block_length", type=int, default=128)
    ap.add_argument("--prefix_refresh_interval", type=int, default=32)
    ap.add_argument("--use_fast_dllm", action="store_true")
    ap.add_argument("--lora_path", default="")
    ap.add_argument("--prompt_version", default="llava_llada")
    ap.add_argument("--vis_dir", default="")
    ap.add_argument("--iou_thresh", type=float, default=0.50)
    # 新增：批大小
    ap.add_argument("--batch_size", type=int, default=8)
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

    # 保障有 pad_token_id（否则 attention_mask 将视 pad 为有效位）
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    padding_side = getattr(tokenizer, "padding_side", "right")

    if args.projector_bin and os.path.isfile(args.projector_bin):
        sd = torch.load(args.projector_bin, map_location="cpu")
        try: model.load_state_dict(sd, strict=False)
        except Exception: pass

    if args.use_fast_dllm and HAS_FAST:
        register_fast_dllm_hook(model)

    data = load_json(args.data_json)
    if args.max_samples and args.max_samples > 0:
        data = data[:args.max_samples]

    total_tp, total_fp, total_fn, total_gt, total_pred = 0, 0, 0, 0, 0
    iou_matched, l1_matched = [], []
    rows = []
    t0 = time.time()

    B = max(1, int(args.batch_size))
    conv_key = args.prompt_version if args.prompt_version in conv_templates else "llava_llada"

    # 按批推进
    for start in tqdm(range(0, len(data), B), ncols=100):
        batch = data[start:start+B]

        # === 构造 prompts & 输入 ===
        images: List[Image.Image] = []
        image_sizes: List[Tuple[int,int]] = []
        img_rels: List[str] = []
        gt_texts: List[str] = []
        ids: List[str] = []
        prompts: List[str] = []

        for ex in batch:
            img_rel = ex["image"]
            img_path = os.path.join(args.image_root, img_rel)
            image = Image.open(img_path).convert("RGB")
            images.append(image)
            image_sizes.append(image.size)
            img_rels.append(img_rel)
            gt_texts.append(ex["conversations"][1]["value"])
            ids.append(ex.get("id", "N/A"))

            user_msg = ex["conversations"][0]["value"]
            conv = copy.deepcopy(conv_templates[conv_key])
            conv.append_message(conv.roles[0], user_msg)
            conv.append_message(conv.roles[1], None)
            prompts.append(conv.get_prompt())

        # === 文本 tokens：逐条插图标记 → pad 成批 ===
        batch_ids_1d = [
            tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").to(torch.long)
            for p in prompts
        ]
        input_ids, attention_mask = pad_batch_input_ids(
            batch_ids_1d, tokenizer.pad_token_id, args.device, padding_side
        )

        # === 图像张量：一次处理整批 ===
        with torch.inference_mode():
            images_tensor = process_images(images, image_processor, model.config)
            # process_images 可能返回 Tensor 或 list（anyres 等）；统一迁移到 device/dtype
            if isinstance(images_tensor, list):
                images_tensor = [im.to(dtype=dtype, device=args.device) for im in images_tensor]
            else:
                images_tensor = images_tensor.to(dtype=dtype, device=args.device)

            # === 生成 ===
            out_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,          # 关键：告知哪里是 pad
                images=images_tensor,
                image_sizes=image_sizes,
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.block_length,
                tokenizer=tokenizer,
                stopping_criteria=['<|eot_id|>'],
                prefix_refresh_interval=args.prefix_refresh_interval,
                threshold=1,
            )

        pred_texts = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        pred_texts = [t.strip() for t in pred_texts]

        # === 逐样本评估（与你原逻辑一致） ===
        for ex_id, img_rel, gt_text, pred_text, pil_img in zip(ids, img_rels, gt_texts, pred_texts, images):
            gt_dict  = parse_bbox_dict(gt_text)
            pred_dict = parse_bbox_dict(pred_text)

            if not gt_dict:
                continue

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

            precision_img = (tp_img/(tp_img+fp_img)) if (tp_img+fp_img)>0 else 0.0
            recall_img = (tp_img/(tp_img+fn_img)) if (tp_img+fn_img)>0 else 0.0
            mean_iou_matched = (np.mean(ious_img) if ious_img else 0.0)
            
            rows.append([
                ex_id, img_rel,
                gt_count_img, pred_count_img, tp_img, fp_img, fn_img,
                f"{precision_img:.4f}", f"{recall_img:.4f}", f"{mean_iou_matched:.4f}",
                pred_text.replace("\n"," ")[:2000]
            ])

            if args.vis_dir:
                save_path = os.path.join(args.vis_dir, f"{ex_id if ex_id!='N/A' else os.path.basename(img_rel)}_vis.jpg")
                try: save_vis_multi(pil_img, pred_dict, gt_dict, save_path, args.iou_thresh)
                except Exception: pass

    # --- 汇总与保存 ---
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
        "pretrained": args.pretrained, "lora_path": args.lora_path,
        "batch_size": int(args.batch_size)
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
