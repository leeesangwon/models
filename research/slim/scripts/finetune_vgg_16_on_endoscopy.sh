# This script performs the following operations:
# 1. Downloads the endoscopy dataset
# 2. Fine-tunes an vgg 16 model on the Endoscopy training set.
# 3. Evaluates the model on the Endoscopy validation set.
#
# Usage:
# cd research/slim
# bash ./scripts/finetune_vgg_16_on_endoscopy.sh

PROJECTS_DIR=/home/sangwon/Projects/Medical
START_DATE=$(date '+%F')

# About base model
PRETRAINED_CHECKPOINT_DIR=${PROJECTS_DIR}/PretrainedModel

# vgg16
MODEL_NAME='vgg_16'
PREPROCESSING_NAME='vgg_16'
EXCLUDE_SCOPES='vgg_16/fc8,vgg_16/conv1_1'
TRAINABLE_SCOPES='vgg_16/conv0_1,vgg_16/conv0_2,vgg_16/conv0_3,vgg_16/conv0_1x1,vgg_16/fc8, vgg_16/fc7, vgg_16/fc6'

# About dataset
DATA_CROSS_VAL=(0 1 2 3 4)
FOLDER_NAME=2018-01-26_3_image

# About training
TRAIN_BATCH_SIZE=32
EVAL_BATCH_SIZE=2
MAX_NUMBER_OF_STEPS=10000
EVALUATE_INTERVAL=1000
START_STEP=0

# for ((k=0; k<${#DATA_CROSS_VAL[*]}; k++))
# do
#     DATASET_DIR=${PROJECTS_DIR}/DATA/CLASSIFICATION/threeImage_0117
#     DATASET_NAME=endoscopy
#     DATASET_FILE_PATTERN=classification_data_of_3_images_${DATA_CROSS_VAL[${k}]}_%s_224_20180117.tfrecord

#     # About training
#     TRAIN_DIR=${PROJECTS_DIR}/TRAIN/CLASSIFICATION/${DATASET_NAME}-models/${MODEL_NAME}_${PREPROCESSING_NAME}/${FOLDER_NAME}/${DATA_CROSS_VAL[${k}]}
#     EVAL_DIR=${TRAIN_DIR}/eval
#     CKPT_PATH=${PRETRAINED_CHECKPOINT_DIR}/${MODEL_NAME}*.ckpt

#     for ((j=START_STEP+EVALUATE_INTERVAL; j<=MAX_NUMBER_OF_STEPS; j+=EVALUATE_INTERVAL))
#     do
#         # Fine-tune
#         python train_image_classifier.py \
#             --train_dir=${TRAIN_DIR} \
#             --dataset_name=${DATASET_NAME} \
#             --dataset_split_name=train \
#             --dataset_dir=${DATASET_DIR} \
#             --dataset_file_pattern=${DATASET_FILE_PATTERN} \
#             --model_name=${MODEL_NAME} \
#             --preprocessing_name=${PREPROCESSING_NAME} \
#             --checkpoint_path=${CKPT_PATH} \
#             --checkpoint_exclude_scopes=${EXCLUDE_SCOPES} \
#             --max_number_of_steps=${j} \
#             --batch_size=${TRAIN_BATCH_SIZE} \
#             --save_interval_secs=600 \
#             --save_summaries_secs=60 \
#             --log_every_n_steps=100 \
#             --learning_rate=0.0001 \
#             --learning_rate_decay_type=fixed \
#             --optimizer=rmsprop \
#             --weight_decay=0.00004 \
#             --trainable_scopes=${TRAINABLE_SCOPES} \
#             --ignore_missing_vars=True

#         # Run evaluation.
#         python eval_image_classifier.py \
#             --batch_size=${EVAL_BATCH_SIZE} \
#             --checkpoint_path=${TRAIN_DIR} \
#             --eval_dir=${EVAL_DIR} \
#             --dataset_name=${DATASET_NAME} \
#             --dataset_split_name=test \
#             --dataset_dir=${DATASET_DIR} \
#             --dataset_file_pattern=${DATASET_FILE_PATTERN} \
#             --model_name=${MODEL_NAME} \
#             --preprocessing_name=${PREPROCESSING_NAME}
#     done
# done

TRAIN_DIR1=TRAIN_DIR

# second step

FOLDER_NAME=2018-01-26_3_image_2nd_step
MAX_NUMBER_OF_STEPS=20000
EVALUATE_INTERVAL=1000
START_STEP=0

for ((k=0; k<${#DATA_CROSS_VAL[*]}; k++))
do
    DATASET_DIR=${PROJECTS_DIR}/DATA/CLASSIFICATION/threeImage_0117
    DATASET_NAME=endoscopy
    DATASET_FILE_PATTERN=classification_data_of_3_images_${DATA_CROSS_VAL[${k}]}_%s_224_20180117.tfrecord

    # About training
    TRAIN_DIR=${PROJECTS_DIR}/TRAIN/CLASSIFICATION/${DATASET_NAME}-models/${MODEL_NAME}_${PREPROCESSING_NAME}/${FOLDER_NAME}/${DATA_CROSS_VAL[${k}]}
    EVAL_DIR=${TRAIN_DIR}/eval
    CKPT_PATH=${PROJECTS_DIR}/TRAIN/CLASSIFICATION/${DATASET_NAME}-models/${MODEL_NAME}_${PREPROCESSING_NAME}/2018-01-26_3_image/${DATA_CROSS_VAL[${k}]}/model.ckpt-10000

    for ((j=START_STEP+EVALUATE_INTERVAL; j<=MAX_NUMBER_OF_STEPS; j+=EVALUATE_INTERVAL))
    do
        # Fine-tune
        python train_image_classifier.py \
            --train_dir=${TRAIN_DIR} \
            --dataset_name=${DATASET_NAME} \
            --dataset_split_name=train \
            --dataset_dir=${DATASET_DIR} \
            --dataset_file_pattern=${DATASET_FILE_PATTERN} \
            --model_name=${MODEL_NAME} \
            --preprocessing_name=${PREPROCESSING_NAME} \
            --checkpoint_path=${CKPT_PATH} \
            --checkpoint_exclude_scopes=${EXCLUDE_SCOPES} \
            --max_number_of_steps=${j} \
            --batch_size=${TRAIN_BATCH_SIZE} \
            --save_interval_secs=600 \
            --save_summaries_secs=60 \
            --log_every_n_steps=100 \
            --learning_rate=0.0001 \
            --learning_rate_decay_type=fixed \
            --optimizer=rmsprop \
            --weight_decay=0.00004 \
            --trainable_scopes=${TRAINABLE_SCOPES} \
            --ignore_missing_vars=True

        # Run evaluation.
        python eval_image_classifier.py \
            --batch_size=${EVAL_BATCH_SIZE} \
            --checkpoint_path=${TRAIN_DIR} \
            --eval_dir=${EVAL_DIR} \
            --dataset_name=${DATASET_NAME} \
            --dataset_split_name=test \
            --dataset_dir=${DATASET_DIR} \
            --dataset_file_pattern=${DATASET_FILE_PATTERN} \
            --model_name=${MODEL_NAME} \
            --preprocessing_name=${PREPROCESSING_NAME}
    done
done
