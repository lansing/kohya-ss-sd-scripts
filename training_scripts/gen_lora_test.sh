#!/usr/bin/env bash

export INSTANCE="ohwx"
export CLASS="man"
export OUTPUT_MODEL_NAME="max-lora-sdxl-1024-5"

export PRETRAINED_MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"

export BASE_DIR="/home/max/Models/StableDiffusion/"
export OUTPUT_DIR="$BASE_DIR/$OUTPUT_MODEL_NAME"
export EPOCH=000005
export SEED=300

#export PROMPT="portrait photo of ($INSTANCE $CLASS:1.1), white background"
export PROMPT="portrait photo of ($INSTANCE:1.8), marlboro man, serious expression, brokeback mountain, cowboy, western shirt, leather jacket, 1978, leica, kodachrome, film grain, 90mm"
export NEG_PROMPT="--n (hdr, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, squinting, low quality"
#export NEG_PROMPT=""
#portrait photo of (ohwx man:1.1) wearing an expensive White suit, white background, fit <lora:12gb_settings-000007:1>

#(blue eyes, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), fat, text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck



accelerate launch --num_cpu_threads_per_process 1 sdxl_gen_img.py  \
    --ckpt $PRETRAINED_MODEL_NAME \
    --W 768 --H 768 \
    --no_half_vae \
    --fp16 \
    --batch_size 1 \
    --images_per_prompt 8 \
    --sampler k_euler_a \
    --steps 32 \
    --seed $SEED \
    --prompt "$PROMPT $NEG_PROMPT" \
    --network_module networks.lora \
    --network_weights $OUTPUT_DIR/$OUTPUT_MODEL_NAME-$EPOCH.safetensors \
    --network_mul 1.0 \
    --sdpa \
    --xformers




