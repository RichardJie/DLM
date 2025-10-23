ROOT=/root/llada
export HF_ENDPOINT=https://hf-mirror.com
experiment_dir=vis_llada_v_val_train_dpo_evident

CUDA_VISIBLE_DEVICES=0 python $ROOT/dataset/eval_bbox_multi.py \
  --data_json  $ROOT/dataset/coco2017/val2017_split/llava_multi/coco_test_sft_final_v2_en.json \
  --image_root $ROOT/dataset/coco2017/val2017_split \
  --pretrained $ROOT/train/models/LLaDA-V \
  --lora_path $ROOT/train/exp/llada_v_val_train_dpo_evident \
  --model_name llava_llada_lora \
  --prompt_version llava_llada \
  --device cuda:0 \
  --dtype fp16 \
  --steps 128 \
  --gen_length 128 \
  --block_length 128 \
  --prefix_refresh_interval 32 \
  --save_csv $ROOT/dataset/result/eval_detail_${experiment_dir}.csv \
  --save_sum $ROOT/dataset/result/eval_summary_${experiment_dir}.json \
  --vis_dir $ROOT/dataset/result/${experiment_dir} \
  --use_fast_dllm \
  --iou_thresh 0.50


# /usr/bin/shutdown -h now