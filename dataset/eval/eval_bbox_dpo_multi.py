# eval_bbox_multi.py (最终修正版, 兼容原始脚本框架)
import argparse, os, re, json, math, csv, time, copy, ast, sys
from typing import Tuple, Optional, List, Dict, Any

import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm

# 保持与你仓库一致的路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../train")))

from transformers.generation import stopping_criteria  # 保持兼容
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
# ====== 解析与后处理增强：预清洗 / 宽松解析 / 合并重复键 / 去重 / 兜底 / 按计数裁剪 ======
# ==============================================================================

_NUM_RE = r'[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?'
_QUAD_RE = rf'\[\s*{_NUM_RE}\s*,\s*{_NUM_RE}\s*,\s*{_NUM_RE}\s*,\s*{_NUM_RE}\s*\]'
_ARR_RE  = rf'\[\s*(?:{_QUAD_RE}(?:\s*,\s*{_QUAD_RE})*)?\s*\]'
# 逐类兜底提取："class": [[...], ...]
_PAIR_RE = re.compile(rf'"([A-Za-z][\w ]*)"\s*:\s*({_ARR_RE})')
_NAN_PAT = re.compile(r'\b(NaN|nan|NAN|Infinity|INF|inf|-Infinity|-INF)\b')
_COUNTS_LINE = re.compile(r'counts?\s*:\s*([^\n]+)', re.IGNORECASE)
_COUNTS_ITEM = re.compile(r'([A-Za-z][\w ]*)\s*\(\s*(\d+)\s*\)')

def normalize_class_name(name: str) -> str:
    """类名规范：小写、下划线转空格、合并多空格、trim。"""
    s = (name or "").lower().replace("_", " ")
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _is_finite_num(x) -> bool:
    try:
        xf = float(x)
        return math.isfinite(xf)
    except Exception:
        return False

def _to_float4(v) -> Tuple[float, float, float, float]:
    # 严格四元 & 可转换为 float
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        raise ValueError("not a 4-tuple")
    vals = [float(x) for x in v]  # 若有 None/str/nan 会抛异常
    return tuple(vals)  # type: ignore

def clamp_box(box: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    # [0,1] clamp
    x1 = max(0.0, min(1.0, x1)); y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2)); y2 = max(0.0, min(1.0, y2))
    # 顺序修复 + 最小宽高
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
    # 用 round 保留 prec 位小数，然后转回 float 元组
    return tuple(round(float(v), prec) for v in box)


def _dedup_boxes(boxes: List[Tuple[float, float, float, float]], prec: int = 3, iou_thr: float = 0.90
                 ) -> List[Tuple[float, float, float, float]]:
    """两阶段去重：三位小数一致 → 高阈值 IoU（默认 0.90，更激进抑制“微位移复制”）"""
    if not boxes: return []
    # 第一阶段：按三位小数一致去重（快）
    seen = set()
    uniq = []
    for b in boxes:
        rb = _round_box(b, prec)
        if rb in seen:
            continue
        seen.add(rb)
        uniq.append(b)
    # 第二阶段：高阈值 IoU 去重（贪心）
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
    # 脏 token → JSON 可解析占位
    s = _NAN_PAT.sub('null', s)
    # 清理明显的“空键”/奇怪切分：例如 """:  之类
    s = re.sub(r'\"\"\"+\s*:\s*[^,\}\]]*', '', s)
    # 去掉控制字符
    s = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', s)
    return s

def _find_bbox_dict_block(s: str) -> str:
    """优先从 'bbox_dict' 后配对大括号；若失败，退化为全局最外层 {...}。"""
    if not s: return ""
    idx = s.lower().find("bbox_dict")
    if idx != -1:
        brace_start = s.find("{", idx)
        if brace_start != -1:
            depth = 0
            for i in range(brace_start, len(s)):
                if s[i] == "{": depth += 1
                elif s[i] == "}":
                    depth -= 1
                    if depth == 0:
                        return s[brace_start:i+1]
    # 退化：全局外层
    l, r = s.find("{"), s.rfind("}")
    if l == -1 or r == -1 or l > r: return ""
    return s[l:r+1]

def _loads_pairs(dict_str: str):
    """尽量保留重复键与顺序，返回[(key, value), ...]。"""
    try:
        return json.loads(dict_str, object_pairs_hook=list)
    except Exception:
        pass
    try:
        js2 = dict_str.replace("'", '"')
        js2 = re.sub(r",\s*([\}\]])", r"\1", js2)  # 去尾逗号
        return json.loads(js2, object_pairs_hook=list)
    except Exception:
        pass
    try:
        obj = ast.literal_eval(dict_str)
        if isinstance(obj, dict): return list(obj.items())
        if isinstance(obj, list): return obj
    except Exception:
        pass
    return None

def _regex_salvage_bbox_dict(raw_text: str) -> Dict[str, List]:
    """当常规解析失败或为空时，按类名逐段兜底抢救。"""
    s = str(raw_text or "")
    found: Dict[str, List] = {}
    for m in _PAIR_RE.finditer(s):
        key = normalize_class_name(m.group(1))
        arr = m.group(2)
        # 宽松转成 py 列表
        try:
            arr_py = json.loads(arr.replace("'", '"'))
        except Exception:
            try:
                arr_py = ast.literal_eval(arr)
            except Exception:
                continue
        if isinstance(arr_py, list):
            found.setdefault(key, []).extend(arr_py)
    return found

def _parse_counts_map(raw_text: str) -> Dict[str, int]:
    """从 'count:' 或 'counts:' 行解析 {class: n}。"""
    want = {}
    m = _COUNTS_LINE.search(raw_text or "")
    if not m: return want
    counts_str = m.group(1)
    for key, num in _COUNTS_ITEM.findall(counts_str):
        want[normalize_class_name(key)] = int(num)
    return want

def _cap_by_counts(raw_text: str, pred: Dict[str, List[Tuple[float,float,float,float]]]
                   ) -> Dict[str, List[Tuple[float,float,float,float]]]:
    """按 counts 上限裁剪预测（只用于 pred，不用于 GT）。"""
    if not pred: return pred
    want = _parse_counts_map(raw_text)
    if not want: return pred
    out = {}
    for k, boxes in pred.items():
        n = want.get(k, None)
        if isinstance(boxes, list):
            out[k] = boxes[:n] if isinstance(n, int) and n >= 0 else boxes
    return out

def parse_bbox_dict(text: str, prec: int = 3, iou_dedup: float = 0.90,
                    apply_counts_cap: bool = False
                    ) -> Dict[str, List[Tuple[float, float, float, float]]]:
    """
    统一入口：
      预清洗 → 宽松解析(合并重复键) → 过滤/规范化 → 两阶段去重 → （可选）按 counts 裁剪
      若解析失败或为空 → 正则兜底抢救 → 同样清洗去重 → （可选）按 counts 裁剪
    返回: { class_name(lower&space-normalized): [(x1,y1,x2,y2), ...], ... }
    """
    if not text:
        return {}
    raw = str(text)
    s = _sanitize_text(raw)

    # 尝试常规解析
    out: Dict[str, List[Tuple[float, float, float, float]]] = {}
    blk = _find_bbox_dict_block(s)
    if blk:
        pairs = _loads_pairs(blk)
        if pairs:
            merged: Dict[str, List] = {}
            for k, v in pairs:
                key = normalize_class_name(str(k))
                if not key: continue
                if not isinstance(v, list):
                    # 容错：{"x1":..} → [x1,y1,x2,y2]
                    if isinstance(v, dict):
                        v = [list(v.values())]
                    else:
                        continue
                merged.setdefault(key, []).extend(v)

            # 过滤 & 规范化
            for key, arr in merged.items():
                if not isinstance(arr, list): continue
                clean: List[Tuple[float,float,float,float]] = []
                for cand in arr:
                    try:
                        b = _to_float4(cand)
                    except Exception:
                        continue
                    if not all(_is_finite_num(x) for x in b):
                        continue
                    clean.append(clamp_box(b))
                if clean:
                    out[key] = _dedup_boxes(clean, prec=prec, iou_thr=iou_dedup)

    # 若为空，做兜底抢救
    if not out:
        merged = _regex_salvage_bbox_dict(s)
        for key, arr in merged.items():
            if not isinstance(arr, list): continue
            clean: List[Tuple[float,float,float,float]] = []
            for cand in arr:
                try:
                    b = _to_float4(cand)
                except Exception:
                    continue
                if not all(_is_finite_num(x) for x in b):
                    continue
                clean.append(clamp_box(b))
            if clean:
                out[normalize_class_name(key)] = _dedup_boxes(clean, prec=prec, iou_thr=iou_dedup)

    # 按 counts 裁剪（仅预测时启用）
    if apply_counts_cap and out:
        out = _cap_by_counts(raw, out)

    return out

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

    # --- 评估循环 ---
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

        # --- 生成调用 ---
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
        gt_text = ex["conversations"][1]["value"]

        # ---- 解析（GT 不裁剪；Pred 裁剪 + 兜底）----
        gt_dict  = parse_bbox_dict(gt_text, apply_counts_cap=False)
        pred_dict = parse_bbox_dict(pred_text, apply_counts_cap=True)

        if not gt_dict:
            # 理论上不会出现；出现则跳过该样本
            continue

        # 统计
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
            except Exception:
                pass

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
