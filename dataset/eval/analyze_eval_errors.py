#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv, re, json, ast, argparse, math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any

NUM_RE = r'[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?'
QUAD_RE = rf'\[\s*{NUM_RE}\s*,\s*{NUM_RE}\s*,\s*{NUM_RE}\s*,\s*{NUM_RE}\s*\]'
ARR_RE  = rf'\[\s*(?:{QUAD_RE}(?:\s*,\s*{QUAD_RE})*)?\s*\]'
PAIR_RE = re.compile(rf'"([A-Za-z][\w ]*)"\s*:\s*({ARR_RE})')

NAN_PAT = re.compile(r'\b(NaN|nan|NAN|Infinity|INF|inf|-Infinity|-INF)\b')
COUNTS_LINE = re.compile(r'counts?\s*:\s*([^\n]+)', re.IGNORECASE)
COUNTS_ITEM = re.compile(r'([A-Za-z][\w ]*)\s*\(\s*(\d+)\s*\)')

def is_finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False

def clamp_box(b: Tuple[float,float,float,float]) -> Tuple[float,float,float,float]:
    x1,y1,x2,y2 = [float(v) for v in b]
    x1 = max(0.0, min(1.0, x1)); y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2)); y2 = max(0.0, min(1.0, y2))
    if x2 < x1: x1,x2 = x2,x1
    if y2 < y1: y1,y2 = y2,y1
    eps = 1e-6
    if x2-x1 < eps: x2 = min(1.0, x1+eps)
    if y2-y1 < eps: y2 = min(1.0, y1+eps)
    return (x1,y1,x2,y2)

def iou(a, b) -> float:
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw,ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw*ih
    aarea = max(0.0, ax2-ax1)*max(0.0, ay2-ay1)
    barea = max(0.0, bx2-bx1)*max(0.0, by2-by1)
    denom = aarea + barea - inter
    return (inter/denom) if denom>0 else 0.0

def round_box(b, prec=3):
    return tuple(float(f"{v:.{prec}f}") for v in b)

def dedup_boxes(boxes, prec=3, iou_thr=0.90):
    if not boxes: return []
    # 快速：三位小数一致去重
    seen = set(); uniq = []
    for b in boxes:
        rb = round_box(b, prec)
        if rb in seen: 
            continue
        seen.add(rb); uniq.append(b)
    # 高阈值 IoU 去重
    kept = []
    for b in uniq:
        keep = True
        for kb in kept:
            if iou(b,kb) >= iou_thr:
                keep = False; break
        if keep: kept.append(b)
    return kept

def find_bbox_dict_block(s: str) -> str:
    # 在原文中定位 bbox_dict 后的首个 { ... }，用计数匹配括号，返回子串
    idx = s.lower().find("bbox_dict")
    if idx == -1:
        # 退化：取整段里最外层 { ... }
        l = s.find("{"); r = s.rfind("}")
        return s[l:r+1] if (l!=-1 and r!=-1 and l<r) else ""
    # 找到 bbox_dict 后的第一个 '{'
    brace_start = s.find("{", idx)
    if brace_start == -1: return ""
    depth = 0
    for i in range(brace_start, len(s)):
        if s[i] == "{": depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[brace_start:i+1]
    return ""  # 未闭合

def load_pairs_loose(dict_text: str):
    # 尝试保留重复键的解析
    try:
        return json.loads(dict_text, object_pairs_hook=list)
    except Exception:
        pass
    # 宽松处理：单引号→双引号、去尾逗号
    try:
        js2 = dict_text.replace("'", '"')
        js2 = re.sub(r",\s*([\}\]])", r"\1", js2)
        return json.loads(js2, object_pairs_hook=list)
    except Exception:
        pass
    # 兜底：ast
    try:
        obj = ast.literal_eval(dict_text)
        if isinstance(obj, dict): return list(obj.items())
        if isinstance(obj, list): return obj
    except Exception:
        pass
    return None

def parse_bbox_dict_loose(raw_text: str):
    """返回 (pred_dict, flags)；pred_dict: {class: [boxes]}, flags: set(error_tags)"""
    flags = set()
    if not raw_text: 
        flags.add("empty_text"); 
        return {}, flags

    # 标记 NaN/Inf
    if NAN_PAT.search(raw_text or ""):
        flags.add("contains_nan")

    blk = find_bbox_dict_block(raw_text)
    if not blk:
        flags.add("no_bbox_block")
        return {}, flags

    pairs = load_pairs_loose(blk)
    if not pairs:
        flags.add("malformed_bbox_dict")
        return {}, flags

    # 合并重复键
    merged = defaultdict(list)
    key_counter = Counter()
    for k, v in pairs:
        key = str(k).lower().strip()
        key_counter[key] += 1
        if isinstance(v, list):
            merged[key].extend(v)
        elif isinstance(v, dict):
            # 容错：把 {"x1":..} 误格式转列表尝试
            merged[key].append(list(v.values()))
        else:
            # 非列表值直接忽略
            pass

    # 重复键统计（忽略空键）
    for k, c in key_counter.items():
        if k and c > 1:
            flags.add("duplicate_keys")
            break

    # 过滤非法框、clamp
    out = {}
    bad_shape = False
    bad_num = False
    for k, arr in merged.items():
        if not isinstance(arr, list): 
            continue
        clean = []
        for item in arr:
            if not (isinstance(item, (list,tuple)) and len(item)==4):
                bad_shape = True; 
                continue
            try:
                b = tuple(float(x) for x in item)
            except Exception:
                bad_num = True
                continue
            if not all(is_finite(x) for x in b):
                bad_num = True
                continue
            clean.append(clamp_box(b))
        if clean:
            # 去重
            ded = dedup_boxes(clean, prec=3, iou_thr=0.90)
            if len(ded) < len(clean):
                flags.add("many_duplicates")
            out[k] = ded

    if bad_shape: flags.add("non_quad")
    if bad_num:   flags.add("non_numeric")

    if not out:
        flags.add("parsed_empty")

    # counts 对齐检查（尽力而为，不作为必须）
    m = COUNTS_LINE.search(raw_text or "")
    if m:
        want = {name.strip().lower(): int(num) 
                for name,num in COUNTS_ITEM.findall(m.group(1))}
        if want:
            mismatch = False
            for k, boxes in out.items():
                n = want.get(k, None)
                if isinstance(n, int) and n >= 0 and len(boxes) != n:
                    mismatch = True; break
            if mismatch:
                flags.add("counts_mismatch")

    return out, flags

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to eval_detail.csv")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    total = 0
    error_rows = 0
    by_type = Counter()
    examples = defaultdict(list)
    by_split = Counter()         # multi/single
    by_split_err = Counter()

    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            rid = row.get("id","N/A")
            raw = row.get("raw_output","")
            gt  = int(float(row.get("gt_boxes","0") or 0))
            pred= int(float(row.get("pred_boxes","0") or 0))

            is_multi = ("-multi" in rid)
            by_split["multi" if is_multi else "single"] += 1

            pred_dict, flags = parse_bbox_dict_loose(raw)

            # 判定是否“错误样本”
            bad = False
            reason_tags = set()

            # 规则 1：pred=0 且 gt>0
            if gt > 0 and pred == 0:
                bad = True
                reason_tags.add("pred0_with_gt")

            # 规则 2：bbox_dict 存在但解析空/结构坏/含 NaN/重复键/非四元/非数
            trigger_tags = {"no_bbox_block","malformed_bbox_dict","contains_nan","duplicate_keys","non_quad","non_numeric","parsed_empty"}
            if flags & trigger_tags:
                bad = True
                reason_tags |= (flags & trigger_tags)

            # 规则 3：重复很多（近似）或 counts 不一致
            if "many_duplicates" in flags:
                bad = True; reason_tags.add("many_duplicates")
            if "counts_mismatch" in flags:
                bad = True; reason_tags.add("counts_mismatch")

            if bad:
                error_rows += 1
                by_split_err["multi" if is_multi else "single"] += 1
                for t in sorted(reason_tags):
                    by_type[t] += 1
                    if len(examples[t]) < args.topk:
                        examples[t].append(rid)

    # 汇总输出
    err_ratio = (error_rows/total*100.0) if total else 0.0
    print("==== 错误样本统计 ====")
    print(f"总样本数: {total}")
    print(f"错误样本数: {error_rows}  ({err_ratio:.2f}%)")
    print()
    print("按类型统计（Top）:")
    for t,c in by_type.most_common():
        print(f"- {t}: {c}")
    print()
    print("按任务类型拆分:")
    for k in ["multi","single"]:
        n = by_split.get(k,0); e = by_split_err.get(k,0)
        r = (e/n*100.0) if n else 0.0
        print(f"- {k}: {e}/{n} ({r:.2f}%) 错误率")
    print()
    print("示例 ID（每类最多前几条）:")
    for t, ids in examples.items():
        print(f"[{t}] 例子: {', '.join(ids)}")

if __name__ == "__main__":
    main()
