#!/usr/bin/env bash
# export OMP_NUM_THREADS=8
# export NCCL_IB_DISABLE=0
# export NCCL_DEBUG=WARN
# export NCCL_DEBUG_SUBSYS=ALL

export CUDA_HOME=/hpc2ssd/JH_DATA/spooler/yuxuanzhao/lijungang/wujie/cuda_12_1
export PATH="$CUDA_HOME/bin:$PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"

num_node=1
gpu_num=2

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29199"}
RANK=${RANK:-"0"}

echo "master_addr ${MASTER_ADDR}"
echo "master_port ${MASTER_PORT}"
echo "node_rank ${RANK}"
echo "gpu_num ${gpu_num}"
echo "num_node ${num_node}"

LLM_VERSION="/hpc2ssd/JH_DATA/spooler/yuxuanzhao/lijungang/wujie/LLaDA-V/train/models/LLaDA-8B-Instruct-HF"
VISION_MODEL_VERSION="/hpc2ssd/JH_DATA/spooler/yuxuanzhao/lijungang/wujie/LLaDA-V/train/models/siglip2-so400m-patch14-384"

PROMPT_VERSION="llava_llada"
BASE_RUN_NAME="llada_v_finetune"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

# ===== 新增：日志设置（目录可按需修改）=====
LOG_DIR="/hpc2ssd/JH_DATA/spooler/yuxuanzhao/lijungang/wujie/LLaDA-V/log"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${BASE_RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
echo "Log file: $LOG_FILE"
# =========================================

# 单卡示例：调用时用 bash llada_v_finetune.sh 1 1
# 不再强制设置 CUDA_VISIBLE_DEVICES，或根据需要自己在命令行外部设置

# PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2 stdbuf -oL -eL \
# python -m debugpy --listen localhost:5678 --wait-for-client \
#   -m torch.distributed.run \
#   --nproc_per_node=${gpu_num} \
#   --nnodes=${num_node} \
#   --master_addr=${MASTER_ADDR} \
#   --master_port=${MASTER_PORT} \
#   --node_rank=${RANK} \
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1,3 stdbuf -oL -eL torchrun --nproc_per_node=${gpu_num} --nnodes=${num_node} \
--master_addr=${MASTER_ADDR} --master_port ${MASTER_PORT} --node_rank=${RANK} \
  llava/train/train_mem.py \
  --deepspeed scripts/zero2.json \
  --model_name_or_path ${LLM_VERSION} \
  --version ${PROMPT_VERSION} \
  --data_path "/hpc2ssd/JH_DATA/spooler/yuxuanzhao/lijungang/wujie/LLaDA-V/dataset/coco2017/llava_multi/coco_val2017_grouped_by_category.json" \
  --image_folder "/hpc2ssd/JH_DATA/spooler/yuxuanzhao/lijungang/wujie/LLaDA-V/dataset/coco2017" \
  --video_folder "" \
  --vision_tower ${VISION_MODEL_VERSION} \
  --image_aspect_ratio square \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --group_by_modality_length True \
  --mm_patch_merge_type spatial_unpad \
  --bf16 True \
  --run_name $BASE_RUN_NAME \
  --output_dir "exp/$BASE_RUN_NAME" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --save_strategy "epoch" \
  --save_total_limit 1 \
  --learning_rate 1e-4 \
  --mm_vision_tower_lr 2e-6 \
  --mm_projector_lr 5e-5 \
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
  --torch_compile False \
  --torch_compile_backend "inductor" \
  --dataloader_drop_last True \
  --attn_implementation sdpa \
  --use_conversation_mask False \
  --freeze_backbone True \
  --lora_enable True \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  2>&1 | tee -a "$LOG_FILE"
  # --pretrain_mm_mlp_adapter /hpc2ssd/JH_DATA/spooler/yuxuanzhao/lijungang/wujie/LLaDA-V/train/exp/llada_v_finetune_1092/checkpoint-1092/mm_projector.bin \
