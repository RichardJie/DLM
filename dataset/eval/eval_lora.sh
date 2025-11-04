#!/bin/bash
ROOT_DIR=/root/llada
DATA_ROOT=$ROOT_DIR/dataset
VERSION=version5

CUDA_VISIBLE_DEVICES=0 python $DATA_ROOT/eval/eval_bbox_dpo_multi.py \
  --data_json  $DATA_ROOT/datasets/coco2017/val2017_split/llava_multi/coco_test_sft_version2.json \
  --image_root $DATA_ROOT/datasets/coco2017/val2017_split \
  --pretrained $ROOT_DIR/train/models/LLaDA-V \
  --lora_path  $ROOT_DIR/train/exp/llada_v_val_train_sft \
  --model_name llava_llada_lora \
  --prompt_version llava_llada \
  --device cuda:0 \
  --dtype fp16 \
  --steps 128 \
  --gen_length 128 \
  --block_length 128 \
  --prefix_refresh_interval 32 \
  --save_csv $DATA_ROOT/eval/SFT/$VERSION/eval_detail.csv \
  --save_sum $DATA_ROOT/eval/SFT/$VERSION/eval_summary.json \
  --vis_dir $DATA_ROOT/eval/SFT/$VERSION/vis_multi_object_val_train_SFT \
  --use_fast_dllm \
  --iou_thresh 0.50

/usr/bin/shutdown -h now
