DATA_ROOT=/hpc2ssd/JH_DATA/spooler/yuxuanzhao/lijungang/wujie/LLaDA-V/dataset

CUDA_VISIBLE_DEVICES=2 python $DATA_ROOT/eval_bbox_multi.py \
  --data_json  $DATA_ROOT/coco2017/llava_multi/coco_val2017_grouped_by_category.json \
  --image_root $DATA_ROOT/coco2017 \
  --pretrained /hpc2ssd/JH_DATA/spooler/yuxuanzhao/lijungang/wujie/LLaDA-V/train/models/LLaDA-8B-Instruct-HF \
  --model_name llava_llada \
  --device cuda:0 \
  --dtype fp16 \
  --steps 128 --gen_length 128 --block_length 128 --prefix_refresh_interval 32 \
  --save_csv $DATA_ROOT/eval_detail.csv \
  --save_sum $DATA_ROOT/eval_summary.json \
  --vis_dir $DATA_ROOT/vis_gt_multi_debug \
  --use_fast_dllm \
  # --projector_bin /hpc2ssd/JH_DATA/spooler/yuxuanzhao/lijungang/wujie/LLaDA-V/train/exp/llada_v_finetune/checkpoint-1092/mm_projector.bin