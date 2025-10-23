import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

"""
DPO 数据集质量验证脚本
用于分析生成的 DPO 数据集，确保质量符合预期
"""

def area_norm(box):
    """计算归一化的 bbox 面积"""
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

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

def extract_bboxes_from_answer(answer_str):
    """从答案字符串中提取 bbox"""
    try:
        # 提取 bbox_dict 部分
        if "bbox_dict:" in answer_str:
            bbox_dict_str = answer_str.split("bbox_dict:")[1].strip()
            bbox_dict = json.loads(bbox_dict_str)
            all_boxes = []
            for cat, boxes in bbox_dict.items():
                all_boxes.extend(boxes)
            return all_boxes
    except Exception as e:
        print(f"Error parsing answer: {e}")
        return []
    return []

def validate_dpo_dataset(jsonl_path):
    """验证 DPO 数据集质量"""

    print("="*70)
    print("DPO 数据集质量验证")
    print("="*70)
    print(f"\n正在加载数据集: {jsonl_path}")

    # 加载数据
    samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"总样本数: {len(samples)}")

    # 统计信息
    stats = {
        'neg_modes': defaultdict(int),
        'sample_types': defaultdict(int),
        'small_object_areas': [],
        'chosen_bbox_counts': [],
        'rejected_bbox_counts': [],
        'count_differences': [],
        'mean_ious': [],
        'num_small_objects_per_sample': [],
        'num_tiny_objects_per_sample': [],
    }

    # 分析每个样本
    print("\n分析样本质量...")
    for sample in samples:
        meta = sample.get('meta', {})

        # 1. 统计负样本生成模式
        neg_mode = meta.get('neg_mode', 'unknown')
        stats['neg_modes'][neg_mode] += 1

        # 2. 统计样本类型
        sample_type = meta.get('type', 'unknown')
        stats['sample_types'][sample_type] += 1

        # 3. 统计小目标数量
        num_small = meta.get('num_small_objects', 0)
        num_tiny = meta.get('num_tiny_objects', 0)
        stats['num_small_objects_per_sample'].append(num_small)
        stats['num_tiny_objects_per_sample'].append(num_tiny)

        # 4. 提取 bbox 并分析
        chosen_boxes = extract_bboxes_from_answer(sample['chosen'])
        rejected_boxes = extract_bboxes_from_answer(sample['rejected'])

        stats['chosen_bbox_counts'].append(len(chosen_boxes))
        stats['rejected_bbox_counts'].append(len(rejected_boxes))
        stats['count_differences'].append(len(chosen_boxes) - len(rejected_boxes))

        # 5. 计算 chosen 和 rejected 的差异度（IoU）
        if chosen_boxes and rejected_boxes:
            ious = []
            for c_box in chosen_boxes:
                max_iou = 0
                for r_box in rejected_boxes:
                    max_iou = max(max_iou, iou(c_box, r_box))
                ious.append(max_iou)
            if ious:
                stats['mean_ious'].append(np.mean(ious))

        # 6. 统计小目标面积
        if 'target_area' in meta:
            stats['small_object_areas'].append(meta['target_area'])

    # ========== 输出统计结果 ==========
    print("\n" + "="*70)
    print("统计结果")
    print("="*70)

    print("\n1. 负样本生成策略分布:")
    total_samples = len(samples)
    for mode, count in sorted(stats['neg_modes'].items(), key=lambda x: -x[1]):
        percentage = (count / total_samples) * 100
        print(f"   - {mode}: {count} ({percentage:.1f}%)")

    print("\n2. 样本类型分布:")
    for stype, count in sorted(stats['sample_types'].items(), key=lambda x: -x[1]):
        percentage = (count / total_samples) * 100
        print(f"   - {stype}: {count} ({percentage:.1f}%)")

    print("\n3. 小目标统计:")
    print(f"   - 平均每样本小目标数: {np.mean(stats['num_small_objects_per_sample']):.2f}")
    print(f"   - 平均每样本极小目标数: {np.mean(stats['num_tiny_objects_per_sample']):.2f}")
    print(f"   - 小目标面积范围: {min(stats['small_object_areas']):.4f} ~ {max(stats['small_object_areas']):.4f}")
    print(f"   - 小目标平均面积: {np.mean(stats['small_object_areas']):.4f}")

    print("\n4. Bbox 数量统计:")
    print(f"   - Chosen 平均 bbox 数: {np.mean(stats['chosen_bbox_counts']):.2f}")
    print(f"   - Rejected 平均 bbox 数: {np.mean(stats['rejected_bbox_counts']):.2f}")
    print(f"   - 数量差异（chosen - rejected）:")
    print(f"     * 平均: {np.mean(stats['count_differences']):.2f}")
    print(f"     * 标准差: {np.std(stats['count_differences']):.2f}")
    print(f"     * 最小值: {min(stats['count_differences'])}")
    print(f"     * 最大值: {max(stats['count_differences'])}")

    print("\n5. Chosen vs Rejected 差异度（IoU）:")
    if stats['mean_ious']:
        print(f"   - 平均 IoU: {np.mean(stats['mean_ious']):.3f}")
        print(f"   - IoU 标准差: {np.std(stats['mean_ious']):.3f}")
        print(f"   - IoU < 0.5 的样本比例: {np.mean(np.array(stats['mean_ious']) < 0.5) * 100:.1f}%")
        print(f"   - IoU < 0.3 的样本比例: {np.mean(np.array(stats['mean_ious']) < 0.3) * 100:.1f}%")

    # ========== 质量检查 ==========
    print("\n" + "="*70)
    print("质量检查")
    print("="*70)

    issues = []

    # 检查 1：负样本策略是否多样化
    if len(stats['neg_modes']) < 3:
        issues.append("⚠️  负样本策略不够多样化，只有 {} 种策略".format(len(stats['neg_modes'])))
    else:
        print("✅ 负样本策略多样化（{} 种策略）".format(len(stats['neg_modes'])))

    # 检查 2：小目标覆盖率
    avg_small_objects = np.mean(stats['num_small_objects_per_sample'])
    if avg_small_objects < 1.0:
        issues.append("⚠️  平均每样本小目标数较少（{:.2f}），可能需要增加小目标覆盖".format(avg_small_objects))
    else:
        print("✅ 小目标覆盖充足（平均 {:.2f} 个/样本）".format(avg_small_objects))

    # 检查 3：Chosen vs Rejected 差异度
    if stats['mean_ious']:
        avg_iou = np.mean(stats['mean_ious'])
        if avg_iou > 0.8:
            issues.append("⚠️  Chosen 和 Rejected 过于相似（平均 IoU = {:.3f}），负样本可能不够困难".format(avg_iou))
        else:
            print("✅ Chosen 和 Rejected 差异度合适（平均 IoU = {:.3f}）".format(avg_iou))

    # 检查 4：数量差异
    count_diff_std = np.std(stats['count_differences'])
    if count_diff_std < 0.5:
        issues.append("⚠️  数量差异不够多样化（标准差 = {:.2f}），可能需要增加数量相关的负样本".format(count_diff_std))
    else:
        print("✅ 数量差异多样化（标准差 = {:.2f}）".format(count_diff_std))

    # 输出问题
    if issues:
        print("\n发现以下问题:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ 数据集质量良好，未发现明显问题！")

    # ========== 可视化 ==========
    print("\n正在生成可视化图表...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 图 1：负样本策略分布
    ax = axes[0, 0]
    modes = list(stats['neg_modes'].keys())
    counts = list(stats['neg_modes'].values())
    ax.bar(range(len(modes)), counts, color='steelblue')
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(modes, rotation=45, ha='right')
    ax.set_title('Negative Sample Strategy Distribution')
    ax.set_ylabel('Count')
    ax.grid(axis='y', alpha=0.3)

    # 图 2：小目标面积分布
    ax = axes[0, 1]
    if stats['small_object_areas']:
        ax.hist(stats['small_object_areas'], bins=50, color='coral', edgecolor='black')
        ax.axvline(0.03, color='red', linestyle='--', label='Tiny threshold (3%)')
        ax.axvline(0.10, color='orange', linestyle='--', label='Small threshold (10%)')
        ax.set_title('Small Object Area Distribution')
        ax.set_xlabel('Normalized Area')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(alpha=0.3)

    # 图 3：Bbox 数量对比
    ax = axes[0, 2]
    ax.scatter(stats['chosen_bbox_counts'], stats['rejected_bbox_counts'], alpha=0.3, color='green')
    max_count = max(max(stats['chosen_bbox_counts']), max(stats['rejected_bbox_counts']))
    ax.plot([0, max_count], [0, max_count], 'r--', label='Equal line')
    ax.set_title('Bbox Count: Chosen vs Rejected')
    ax.set_xlabel('Chosen Count')
    ax.set_ylabel('Rejected Count')
    ax.legend()
    ax.grid(alpha=0.3)

    # 图 4：数量差异分布
    ax = axes[1, 0]
    ax.hist(stats['count_differences'], bins=30, color='purple', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', label='No difference')
    ax.set_title('Bbox Count Difference (Chosen - Rejected)')
    ax.set_xlabel('Difference')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(alpha=0.3)

    # 图 5：IoU 分布
    ax = axes[1, 1]
    if stats['mean_ious']:
        ax.hist(stats['mean_ious'], bins=50, color='teal', edgecolor='black')
        ax.axvline(0.5, color='orange', linestyle='--', label='IoU = 0.5')
        ax.axvline(0.95, color='red', linestyle='--', label='Too similar (0.95)')
        ax.set_title('Mean IoU between Chosen and Rejected')
        ax.set_xlabel('Mean IoU')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(alpha=0.3)

    # 图 6：小目标数量分布
    ax = axes[1, 2]
    ax.hist(stats['num_small_objects_per_sample'], bins=30, color='goldenrod', edgecolor='black', alpha=0.7, label='Small objects')
    ax.hist(stats['num_tiny_objects_per_sample'], bins=30, color='darkred', edgecolor='black', alpha=0.7, label='Tiny objects')
    ax.set_title('Number of Small/Tiny Objects per Sample')
    ax.set_xlabel('Count')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = jsonl_path.replace('.jsonl', '_analysis.png')
    plt.savefig(output_path, dpi=150)
    print(f"可视化结果已保存到: {output_path}")

    print("\n" + "="*70)
    print("验证完成！")
    print("="*70)

    return stats

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python validate_dpo_data_quality.py <dpo_dataset.jsonl>")
        sys.exit(1)

    jsonl_path = sys.argv[1]
    validate_dpo_dataset(jsonl_path)