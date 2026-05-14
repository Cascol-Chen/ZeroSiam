#!/bin/bash

# ===== 1️⃣  =====
source /home/troynsc/miniconda3/bin/activate reproduce2

export CUDA_VISIBLE_DEVICES=0

# ===== 2️⃣ config =====
MODEL=swin_tiny     # resnet50_gn_timm, vitbase_timm, vitsmall_timm, convnext_tiny, swin_tiny 
METHOD=zerosiam            # no_adapt, tent, sar, eata, deyo_come, deyo, zerosiam
EXP_TYPE=label_shifts            # label_shifts, bs1, label_shifts+bs1, incorrect_labels_k1, mix_shifts, normal

# ===== 3️⃣ set lr_scale and lr_p =====
case $MODEL in
  resnet50_gn_timm)
    LR_SCALE=10
    LR_P=100
    ;;
  vitbase_timm)
    LR_SCALE=5
    LR_P=5
    ;;
  vitsmall_timm)
    LR_SCALE=10
    LR_P=100
    ;;
  convnext_tiny)
    LR_SCALE=10
    LR_P=100
    ;;
  swin_tiny)
    LR_SCALE=10
    LR_P=50
    ;;
  *)
    echo "❌ Unknown model: $MODEL"
    exit 1
    ;;
esac

echo "======================================"
echo "Running experiment"
echo "Model:     $MODEL"
echo "Method:    $METHOD"
echo "LR_SCALE:  $LR_SCALE"
echo "LR_P:      $LR_P"
echo "EXP_TYPE:  $EXP_TYPE"
echo "======================================"

# ===== 4️⃣ 运行 =====
python3 main.py \
    --data /ssd1/pytorch_dataset/ImageNet \
    --data_corruption /ssd1/nsc/imagenet-c \
    --seed 2021 \
    --level 5 \
    --exp_type $EXP_TYPE \
    --method $METHOD \
    --lr_scale $LR_SCALE \
    --lr_p $LR_P \
    --model $MODEL \
    --output ./outputs \
    --tag1 test

echo "✅ Finished."
