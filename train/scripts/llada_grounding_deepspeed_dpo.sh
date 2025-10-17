#!/usr/bin/env bash
# 必要的
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export HF_ENDPOINT=https://hf-mirror.com
num_node=1
gpu_num=1

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29209"}
RANK=${RANK:-"0"}

ROOT=/root/llada

LLM_VERSION="${ROOT}/train/models/LLaDA-V"
# /root/llada/train/models/LLaDA-V/config.json 186 行   "mm_vision_tower": "/root/llada/train/models/siglip2-so400m-patch14-384",
VISION_MODEL_VERSION="${ROOT}/train/models/siglip2-so400m-patch14-384"


DPO_DATA="${ROOT}/dataset/coco2017/val2017_split/llava_multi/coco_train_dpo_final_mixed_v2_en.jsonl" 
IMAGE_ROOT="${ROOT}/dataset/coco2017/val2017_split/"

PROMPT_VERSION="llava_llada"
BASE_RUN_NAME="llada_v_val_train_dpo"
LOG_DIR="${ROOT}/log"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${BASE_RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL torchrun \
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
  --mm_use_im_patch_token False \
  --group_by_modality_length False \
  --mm_patch_merge_type spatial_unpad \
  --bf16 True \
  --run_name ${BASE_RUN_NAME} \
  --output_dir "exp/${BASE_RUN_NAME}" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --save_strategy "epoch" \
  --save_total_limit 1 \
  --learning_rate 1e-7 \
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
  --lora_weight_path "${ROOT}/train/exp/llada_v_val_train_sft" \
  --precompute_ref_log_probs False \
  --beta 0.1 \
  --label_smoothing 0.05 \
  2>&1 | tee -a "$LOG_FILE"

# /usr/bin/shutdown -h now
