#!/bin/bash
#
# This script performs the following operations:
# 1. Downloads the Flowers dataset
# 2. Fine-tunes an Inception Resnet V2 model on the Flowers training set.
# 3. Evaluates the model on the Flowers validation set.
#
# Usage:
# cd slim
# ./slim/scripts/finetune_inception_resnet_v2_on_flowers.sh
set -e

# About base model
PRETRAINED_CHECKPOINT_DIR=~/tf/checkpoints
MODEL_NAME=inception_resnet_v2
PREPROCESSING_NAME=inception_resnet_v2_notcrop

# About dataset
DATASET_DIR=~/data/forensics/case1
DATASET_NAME=forensics_case1

# About training
TRAIN_DIR=~/forensics/${DATASET_NAME}-models/${MODEL_NAME}/${PREPROCESSING_NAME}
BATCH_SIZE=32
MAX_NUMBER_OF_STEPS=5000

# Fine-tune only the new layers for 1000 steps.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --preprocessing_name=${PREPROCESSING_NAME} \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/${MODEL_NAME}*.ckpt \
  --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
  --trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
  --max_number_of_steps=${MAX_NUMBER_OF_STEPS} \
  --batch_size=${BATCH_SIZE} \
  --save_interval_secs=300 \
  --save_summaries_secs=300
#   --learning_rate=0.01 \
#   --learning_rate_decay_type=fixed \
#   --log_every_n_steps=10 \
#   --optimizer=rmsprop \
#   --weight_decay=0.00004

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --preprocessing_name=${PREPROCESSING_NAME}
