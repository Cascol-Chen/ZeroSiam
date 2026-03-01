#!/bin/bash

# ===== 1️⃣  =====
source /chenyaofo/cgh/miniconda3/bin/activate reproduce1

export CUDA_VISIBLE_DEVICES=2
# export HOME=/chenyaofo/cgh/researchs/Zerosiam_Preview
export HOME=/chenyaofo/cgh

# ===== 2️⃣ config =====
MODEL=swin_tiny     # resnet50_gn_timm, vitbase_timm, vitsmall_timm, convnext_tiny, swin_tiny 
METHOD=zerosiam            # no_adapt, tent, sar, eata, deyo_come, deyo, zerosiam
EXP_TYPE=incorrect_labels_k1            # label_shifts, bs1, label_shifts+bs1, incorrect_labels_k1, mix_shifts, normal

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
    --data /chenyaofo/datasets/TTA/imagenet \
    --data_corruption /chenyaofo/datasets/TTA/imagenet-c \
    --seed 2021 \
    --level 5 \
    --exp_type $EXP_TYPE \
    --method $METHOD \
    --lr_scale $LR_SCALE \
    --lr_p $LR_P \
    --model $MODEL \
    --output /chenyaofo/cgh/cowork/cdy/research/SAR_cdy/outputs2 \
    --tag1 test \
    --tag2 test

echo "✅ Finished."