#!/bin/bash

set -o xtrace

export PRETRAINED_MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export DOCKER_IMAGE="maxrlansing/sd:kohya_ss_sdxl-latest"

export MAX_TRAIN_STEPS=20000
export OUTPUT_MODEL_NAME="max-lora-sdxl-2"

export HF_CACHE_DIR="/home/max/.cache/huggingface/"
export DATASET_DIR="/home/max/Datasets/StableDiffusionTraining/max2/cropped_resized_captioned/"
export DATASET_CONFIG="metadata-docker.toml"
export CLASS_DIR="/home/max/Datasets/Stable-Diffusion-Regularization-Images/man-generated-15"

export BASE_DIR="/home/max/Models/StableDiffusion/"
export OUTPUT_DIR="$BASE_DIR/$OUTPUT_MODEL_NAME"

mkdir -p $OUTPUT_DIR

docker run -ti --rm --runtime=nvidia --gpus all \
  -v $HF_CACHE_DIR:/root/.cache/huggingface \
  -v $DATASET_DIR:/dataset/instance \
  -v $CLASS_DIR:/dataset/class \
  -v $OUTPUT_DIR:/output \
  $DOCKER_IMAGE \
  conda run --live-stream -n torch \
  accelerate launch --num_cpu_threads_per_process 2 sdxl_train_network.py \
      --pretrained_model_name_or_path=$PRETRAINED_MODEL_NAME \
      --dataset_config=/dataset/instance/$DATASET_CONFIG \
      --output_dir=/output \
      --output_name=$OUTPUT_MODEL_NAME \
      --save_model_as=safetensors \
      --prior_loss_weight=1.0 \
      --max_train_steps=$MAX_TRAIN_STEPS \
      --learning_rate=1e-4 \
      --optimizer_type="AdamW8bit" \
      --xformers \
      --mixed_precision="fp16" \
      --no_half_vae \
      --cache_latents \
      --gradient_checkpointing \
      --save_every_n_epochs=10 \
      --network_module=networks.lora

