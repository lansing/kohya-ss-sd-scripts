export INSTANCE="sks"
export CLASS="man"
export MAX_TRAIN_STEPS=20000
export OUTPUT_MODEL_NAME="max-lora-sdxl-1"

export PRETRAINED_MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"

export DATASET_CONFIG="/home/max/Datasets/StableDiffusionTraining/max2/cropped_resized_captioned/metadata.toml"

export BASE_DIR="/home/max/Models/StableDiffusion/"
export OUTPUT_DIR="$BASE_DIR/$OUTPUT_MODEL_NAME"

accelerate launch --num_cpu_threads_per_process 1 sdxl_train_network.py  \
    --pretrained_model_name_or_path=$PRETRAINED_MODEL_NAME \
    --dataset_config=$DATASET_CONFIG \
    --output_dir=$OUTPUT_DIR \
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
    --network_module=networks.lora \
#    --cache_text_encoder_outputs \
#    --network_train_unet_only \

