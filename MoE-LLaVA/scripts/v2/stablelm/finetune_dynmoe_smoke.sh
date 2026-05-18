#!/usr/bin/env bash
set -euo pipefail

moe_mode="sparse"
num_experts=4
top_k_experts=-1
max_expert_num=${num_experts}
use_residual=False
router_aux_loss_coef=0.01

wandb_project="${WANDB_PROJECT:-dynmoe}"
wandb_entity="${WANDB_ENTITY:-Implement-DL}"
wandb_run_group="${WANDB_RUN_GROUP:-stablelm-dynmoe}"
run_suffix="$(date -u +%Y%m%d-%H%M%S)"
report_to="${REPORT_TO:-wandb}"

APP="/usr/src/app"
DATASET_DIR="${APP}/data/raw/MoE-LLaVA-unzipped"

JSON_FOLDER="${DATASET_DIR}/train_json"
IMAGE_FOLDER="${DATASET_DIR}"

cd "${APP}/DynMoE/MoE-LLaVA"
uv pip install "../DeepSpeed-0.9.5"

n_gpu=8
run_name="${RUN_NAME:-stablelm-dynmoe-smoke-${n_gpu}gpu-${run_suffix}}"

export WANDB_PROJECT="${wandb_project}"
export WANDB_RUN_GROUP="${wandb_run_group}"
if [[ -n "${wandb_entity}" ]]; then
  export WANDB_ENTITY="${wandb_entity}"
fi

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run deepspeed --num_gpus=${n_gpu} --enable_each_rank_log ./rank_logs moellava/train/train_mem.py \
  --moe_enable True \
  --num_experts ${num_experts} \
  --max_expert_num ${max_expert_num} \
  --top_k_experts ${top_k_experts} \
  --capacity_factor 1.5 \
  --moe_mode ${moe_mode} \
  --use_residual ${use_residual} \
  --router_aux_loss_coef ${router_aux_loss_coef} \
  --train_modules gate_proj up_proj down_proj wg \
  --deepspeed ./scripts/zero2.json \
  --model_name_or_path ./checkpoints/MoE-LLaVA-StableLM-Stage2 \
  --version stablelm \
  --data_path ${JSON_FOLDER}/llava_image_tune_.json ${JSON_FOLDER}/nlp_tune.json \
  --image_folder ${IMAGE_FOLDER} \
  --image_tower openai/clip-vit-large-patch14-336 \
  --image_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length False \
  --bf16 True \
  --output_dir ./checkpoints/smoke-dynmoe \
  --num_train_epochs 1 \
  --max_steps 5 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "no" \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --logging_first_step True \
  --tf32 True \
  --model_max_length 1024 \
  --gradient_checkpointing False \
  --dataloader_num_workers 0 \
  --lazy_preprocess True \
  --report_to ${report_to} \
  --run_name "${run_name}" \
  --cache_dir "./cache_dir"
