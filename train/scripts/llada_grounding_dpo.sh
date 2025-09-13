#!/usr/bin/env bash
# 必要的
export CUDA_HOME=/hpc2ssd/JH_DATA/spooler/yuxuanzhao/lijungang/wujie/cuda_12_1
export PATH="$CUDA_HOME/bin:$PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"

num_node=1
gpu_num=3

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29209"}
RANK=${RANK:-"0"}

LLM_VERSION="/hpc2ssd/JH_DATA/spooler/yuxuanzhao/lijungang/wujie/LLaDA-V/train/models/LLaDA-8B-Instruct-HF"
VISION_MODEL_VERSION="/hpc2ssd/JH_DATA/spooler/yuxuanzhao/lijungang/wujie/LLaDA-V/train/models/siglip2-so400m-patch14-384"

DATA_ROOT="/hpc2ssd/JH_DATA/spooler/yuxuanzhao/lijungang/wujie/LLaDA-V/dataset"
DPO_DATA="${DATA_ROOT}/coco2017/llava_multi/coco_val2017_DPO_pairs_small_focus.jsonl" 
IMAGE_ROOT="${DATA_ROOT}/coco2017"

PROMPT_VERSION="llava_llada"
BASE_RUN_NAME="llada_v_dpo"
LOG_DIR="/hpc2ssd/JH_DATA/spooler/yuxuanzhao/lijungang/wujie/LLaDA-V/log"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${BASE_RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2 stdbuf -oL -eL torchrun \
  --nproc_per_node=${gpu_num} --nnodes=${num_node} \
  --master_addr=${MASTER_ADDR} --master_port ${MASTER_PORT} --node_rank=${RANK} \
  llava/train/train_dpo.py \
  --deepspeed scripts/zero2.json \
  --model_name_or_path ${LLM_VERSION} \
  --version ${PROMPT_VERSION} \
  --data_path "${DPO_DATA}" \
  --image_folder "${IMAGE_ROOT}" \
  --vision_tower ${VISION_MODEL_VERSION} \
  --image_aspect_ratio square \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --group_by_modality_length False \
  --mm_patch_merge_type spatial_unpad \
  --bf16 True \
  --run_name ${BASE_RUN_NAME} \
  --output_dir "exp/${BASE_RUN_NAME}" \
  --num_train_epochs 2 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --save_strategy "epoch" \
  --save_total_limit 1 \
  --learning_rate 5e-6 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 8192 \
  --gradient_checkpointing True \
  --dataloader_num_workers 8 \
  --lazy_preprocess True \
  --report_to tensorboard \
  --attn_implementation sdpa \
  --dataloader_drop_last True \
  --freeze_backbone True \
  --lora_enable True \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --lora_weight_path "/hpc2ssd/JH_DATA/spooler/yuxuanzhao/lijungang/wujie/LLaDA-V/train/exp/llada_v_finetune" \
  --precompute_ref_log_probs False \
  --beta 0.1 \
  --label_smoothing 0.05 \
  2>&1 | tee -a "$LOG_FILE"
