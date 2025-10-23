# eval_bbox_coords_repo.py
# 与 generate_demo.py 同路子加载；新增 --vis_dir：保存叠加了 pred/gt 的可视化图片

import argparse, os, re, json, math, csv, time, copy
from typing import Tuple, Optional, List
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../train")))

# ==== 与仓库 demo 一致的依赖 ====
from transformers.generation import stopping_criteria
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

try:
    from llava.hooks.fast_dllm_hook import register_fast_dllm_hook, unregister_fast_dllm_hook
    HAS_FAST = True
except Exception:
    HAS_FAST = False

# ---- 解析 "bbox: [x1, y1, x2, y2]"（允许前后缀与空白） ----
_PAT = re.compile(r'(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)', re.I)

def parse_bbox(text: str) -> Optional[Tuple[float,float,float,float]]:
    m = _PAT.search(text or "")
    if not m: return None
    x1,y1,x2,y2 = [float(m.group(i)) for i in range(1,5)]
    # clamp 到 [0,1] 且保证有效框
    x1 = max(0.0, min(1.0, x1)); y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2)); y2 = max(0.0, min(1.0, y2))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    eps = 1e-6
    if x2 - x1 < eps: x2 = min(1.0, x1 + eps)
    if y2 - y1 < eps: y2 = min(1.0, y1 + eps)
    return x1, y1, x2, y2

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

def center_dist(boxP, boxG) -> float:
    cxp, cyp = (boxP[0]+boxP[2])/2.0, (boxP[1]+boxP[3])/2.0
    cxg, cyg = (boxG[0]+boxG[2])/2.0, (boxG[1]+boxG[3])/2.0
    return float(math.hypot(cxp-cxg, cyp-cyg))

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def inject_projector_weights(model, bin_path: str):
    sd = torch.load(bin_path, map_location="cpu")
    proj = {}
    for k, v in sd.items():
        if "mm_projector" in k:
            if k.startswith("model."):
                proj[k] = v
            elif k.startswith("mm_projector."):
                proj[f"model.{k}"] = v
            else:
                if "model.mm_projector" in k:
                    proj[k] = v
    if not proj:
        print(f"[WARN] No mm_projector keys found in {bin_path}. Skipped.")
        return
    missing, unexpected = model.load_state_dict(proj, strict=False)
    if missing:
        print(f"[INFO] Missing keys when loading projector: {len(missing)} (ok if not all used)")
    if unexpected:
        print(f"[INFO] Unexpected keys when loading projector: {len(unexpected)}")


# ========== 新增：可视化相关 ==========

def _xyxy_from_norm(box_norm, W, H):
    x1 = int(round(box_norm[0] * W))
    y1 = int(round(box_norm[1] * H))
    x2 = int(round(box_norm[2] * W))
    y2 = int(round(box_norm[3] * H))
    # clamp
    x1 = max(0, min(W-1, x1)); y1 = max(0, min(H-1, y1))
    x2 = max(0, min(W-1, x2)); y2 = max(0, min(H-1, y2))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2

def _draw_box(draw: ImageDraw.ImageDraw, xyxy, color, width=3):
    # 多框线兼容旧 PIL
    for w in range(width):
        draw.rectangle((xyxy[0]-w, xyxy[1]-w, xyxy[2]+w, xyxy[3]+w), outline=color)

def _put_text(draw, xy, text, color):
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text(xy, text, fill=color, font=font)

def save_vis(image: Image.Image, pred_norm, gt_norm, save_path, iou_val=None, parse_fail=False):
    im = image.copy()
    W, H = im.size
    dr = ImageDraw.Draw(im)

    # 颜色：pred=红，gt=绿
    if gt_norm is not None:
        gx1, gy1, gx2, gy2 = _xyxy_from_norm(gt_norm, W, H)
        _draw_box(dr, (gx1,gy1,gx2,gy2), color=(0,255,0), width=4)
        _put_text(dr, (gx1+2, gy1+2), "GT", (0,255,0))

    if pred_norm is not None:
        px1, py1, px2, py2 = _xyxy_from_norm(pred_norm, W, H)
        _draw_box(dr, (px1,py1,px2,py2), color=(255,0,0), width=3)
        _put_text(dr, (px1+2, py1-12 if py1-12>0 else py1+2), "PRED", (255,0,0))

    # 标注 IoU/状态
    label = ""
    if parse_fail:
        label = "PARSE_FAIL"
    elif iou_val is not None:
        label = f"IoU={iou_val:.3f}"
    if label:
        _put_text(dr, (5, 5), label, (255,255,0))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    im.save(save_path, quality=95)

# =====================================

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
    # 在 argparse 里加
    ap.add_argument("--lora_path", default="", help="LoRA 目录（含 adapter_config.json）")
    ap.add_argument("--lora_from_checkpoint", action="store_true",
                    help="若指向 checkpoint 子目录（例如 checkpoint-910），打开此项")
    # 1) argparse 添加一个独立的 prompt_version
    ap.add_argument("--prompt_version", default="llava_llada",
                    help="对话模板名，和加载用的 model_name 解耦（例如：llava_llada）")

    # ---- 新增参数：可视化输出目录（不传则不保存）
    ap.add_argument("--vis_dir", default="", help="若提供，将把每个样本的可视化图保存到该目录")
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype=="bf16" else torch.float16
# 若提供 lora_path，则按 LoRA 方式加载；否则走普通基座加载
    if args.lora_path:
        model_path = args.lora_path
        model_base = args.pretrained   # 基座（你的 LLaDA-V 基础权重目录）
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
        inject_projector_weights(model, args.projector_bin)
        # 如需强制 projector 与 dtype 对齐，可取消下一行注释（见你之前的 dtype 报错）：
        # model.get_model().mm_projector.to(dtype)

    if args.use_fast_dllm and HAS_FAST:
        register_fast_dllm_hook(model)

    data = load_json(args.data_json)
    if args.max_samples and args.max_samples>0:
        data = data[:args.max_samples]

    ok, fail = 0, 0
    iou_list: List[float] = []; l1_list: List[float] = []; cd_list: List[float] = []
    rows = []

    t0 = time.time()
    for ex in tqdm(data, ncols=100):
        img_rel = ex["image"]
        img_path = os.path.join(args.image_root, img_rel)
        user_msg = ex["conversations"][0]["value"]

        conv_key = args.prompt_version
        #（可选）保险：若用户传了不存在的键，给个友好 fallback
        if conv_key not in conv_templates:
            # 简单兜底到 llava_llada（也可改成抛错）
            conv_key = "llava_llada"
        conv = copy.deepcopy(conv_templates[conv_key])

        conv.append_message(conv.roles[0], user_msg)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(img_path).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [_im.to(dtype=dtype, device=args.device) for _im in image_tensor]
        image_sizes = [image.size]

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

        gt = ex.get("bbox_norm_384", None)
        if gt is None:
            gt = parse_bbox(ex["conversations"][1]["value"])
        else:
            gt = tuple(gt)
        pred = parse_bbox(pred_text)

        if pred is None or gt is None:
            fail += 1
            rows.append([ex["id"], img_rel, "PARSE_FAIL", "", "", "", "", pred_text])
            # 可视化（仅画 GT 并标注失败）
            if args.vis_dir:
                save_path = os.path.join(args.vis_dir, f"{ex['id']}_vis.jpg")
                save_vis(image, None, gt, save_path, parse_fail=True)
            continue

        i = iou(pred, gt); l = l1_err(pred, gt); c = center_dist(pred, gt)
        ok += 1
        iou_list.append(i); l1_list.append(l); cd_list.append(c)
        rows.append([
            ex["id"], img_rel,
            f"{i:.4f}", f"{l:.4f}", f"{c:.4f}",
            f"[{pred[0]:.3f},{pred[1]:.3f},{pred[2]:.3f},{pred[3]:.3f}]",
            f"[{gt[0]:.3f},{gt[1]:.3f},{gt[2]:.3f},{gt[3]:.3f}]",
            pred_text
        ])
        # 保存可视化
        if args.vis_dir:
            save_path = os.path.join(args.vis_dir, f"{ex['id']}_vis.jpg")
            save_vis(image, pred, gt, save_path, iou_val=i)

    t1 = time.time()
    N = len(data)
    mean_iou = float(np.mean(iou_list)) if iou_list else 0.0
    hit_50   = float(np.mean([1.0 if v>=0.5  else 0.0 for v in iou_list])) if iou_list else 0.0
    hit_75   = float(np.mean([1.0 if v>=0.75 else 0.0 for v in iou_list])) if iou_list else 0.0
    mean_l1  = float(np.mean(l1_list)) if l1_list else 0.0
    mean_cd  = float(np.mean(cd_list)) if cd_list else 0.0

    os.makedirs(os.path.dirname(args.save_csv) or ".", exist_ok=True)
    with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id","image","IoU","L1","CenterDist","pred_norm","gt_norm","raw_output"])
        w.writerows(rows)

    summary = {
        "total": N,
        "parsed": ok,
        "parse_fail": fail,
        "mean_IoU": round(mean_iou,4),
        "IoU@0.50": round(hit_50,4),
        "IoU@0.75": round(hit_75,4),
        "mean_L1": round(mean_l1,4),
        "mean_center_dist": round(mean_cd,4),
        "time_sec": round(t1-t0,2),
        "pretrained": args.pretrained,
        "projector_bin": args.projector_bin
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
