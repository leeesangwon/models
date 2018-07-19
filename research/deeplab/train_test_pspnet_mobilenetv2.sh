#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script is used to run local test on PASCAL VOC 2012 using MobileNet-v2.
# Users could also modify from this script for their use case.
#lo
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   sh ./local_test_mobilenetv2.sh
#
#

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"
DATASET_DIR="datasets"

# Go back to original directory.
cd "${CURRENT_DIR}"

# Set up the working directories.
DATASET_FOLDER="cityscapes"
EXP_FOLDER="exp/train_on_train_set_mobilenetv2"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/${EXP_FOLDER}/export"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

# Copy locally the trained checkpoint as the initial checkpoint.
CKPT_NAME="mobilenet_v2_1.0_224"
cd "${CURRENT_DIR}"

CITYSCAPES_DATASET="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/tfrecord"

# Train 10 iterations.
NUM_ITERATIONS=90000
python "${WORK_DIR}"/train_pspnet.py \
  --logtostderr \
  --train_split="train" \
  --model_variant="mobilenet_v2" \
  --atrous_rates=1 \
  --atrous_rates=2 \
  --atrous_rates=3 \
  --atrous_rates=6 \
  --output_stride=8 \
  --train_crop_size=473 \
  --train_crop_size=473 \
  --train_batch_size=1 \
  --iter_size=16 \
  --dataset="cityscapes" \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=false \
  --tf_initial_checkpoint="${INIT_FOLDER}/${CKPT_NAME}/mobilenet_v2_1.0_224.ckpt" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${CITYSCAPES_DATASET}"

# Run evaluation. This performs eval over the full val split (1449 images) and
# will take a while.
# Using the provided checkpoint, one should expect mIOU=75.34%.
python "${WORK_DIR}"/eval_pspnet.py \
  --logtostderr \
  --eval_split="val" \
  --model_variant="mobilenet_v2" \
  --atrous_rates=1 \
  --atrous_rates=2 \
  --atrous_rates=3 \
  --atrous_rates=6 \
  --output_stride=8 \
  --eval_crop_size=1025 \
  --eval_crop_size=2049 \
  --dataset="cityscapes" \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="${CITYSCAPES_DATASET}" \
  --max_number_of_evaluations=1

# Visualize the results.
python "${WORK_DIR}"/vis_pspnet.py \
  --logtostderr \
  --vis_split="val" \
  --model_variant="mobilenet_v2" \
  --atrous_rates=1 \
  --atrous_rates=2 \
  --atrous_rates=3 \
  --atrous_rates=6 \
  --output_stride=8 \
  --vis_crop_size=1025 \
  --vis_crop_size=2049 \
  --dataset="cityscapes" \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --vis_logdir="${VIS_LOGDIR}" \
  --dataset_dir="${CITYSCAPES_DATASET}" \
  --max_number_of_iterations=1
