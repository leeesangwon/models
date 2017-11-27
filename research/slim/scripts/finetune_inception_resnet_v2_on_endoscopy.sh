# This script performs the following operations:
# 1. Downloads the endoscopy dataset
# 2. Fine-tunes an inception resnet v2 model on the Endoscopy training set.
# 3. Evaluates the model on the Endoscopy validation set.
#
# Usage:
# cd research/slim
# bash ./scripts/finetune_inception_resnet_v2_on_endoscopy.sh

export PROJECTS_DIR=/home/sangwon/Projects/Medical
export START_DATE=$(date '+%F')

# About base model
export PRETRAINED_CHECKPOINT_DIR=${PROJECTS_DIR}/PretrainedModel

# inception resnet v2
export MODEL_NAME='inception_resnet_v2'
export PREPROCESSING_NAME='inception_resnet_v2_notcrop'
export EXCLUDE_SCOPES='InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits'
export TRAINABLE_SCOPES='InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits'

# About dataset
export DATA_SUBNAMES=(A E)
export DATA_CROSS_VAL=(0 1 2 3 4)
export FOLDER_NAME=2017-11-21_20:35:59
# FOLDER_NAME=2017-11-19_19:16:57s

# About training
export TRAIN_BATCH_SIZE=32
export EVAL_BATCH_SIZE=2
export MAX_NUMBER_OF_STEPS=30000
export EVALUATE_INTERVAL=500
export START_STEP=15000

for ((i=0; i<${#DATA_SUBNAMES[*]}; i++))
do
    for ((k=0; k<${#DATA_CROSS_VAL[*]}; k++))
    do
        DATASET_DIR=${PROJECTS_DIR}/DATA/CLASSIFICATION/data1116+1117/trainval/data_${DATA_SUBNAMES[$i]}
        DATASET_NAME=endoscopy_${DATA_SUBNAMES[$i]}
        DATASET_FILE_PATTERN=cls_data_${DATA_SUBNAMES[$i]}_${DATA_CROSS_VAL[${k}]}_%s_*.tfrecord

        # About training
        TRAIN_DIR=${PROJECTS_DIR}/TRAIN/CLASSIFICATION/${DATASET_NAME}-models/${MODEL_NAME}_${PREPROCESSING_NAME}/${FOLDER_NAME}/${DATA_CROSS_VAL[${k}]}
        EVAL_DIR=${TRAIN_DIR}/eval

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
                --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/${MODEL_NAME}*.ckpt \
                --checkpoint_exclude_scopes=${EXCLUDE_SCOPES} \
                --max_number_of_steps=${j} \
                --batch_size=${TRAIN_BATCH_SIZE} \
                --save_interval_secs=600 \
                --save_summaries_secs=60 \
                --log_every_n_steps=10 \
                --learning_rate=0.0001 \
                --learning_rate_decay_type=fixed \
                --optimizer=rmsprop \
                --weight_decay=0.00004 \
                --trainable_scopes=${TRAINABLE_SCOPES}

            # Run evaluation.
            python eval_image_classifier.py \
                --batch_size=${EVAL_BATCH_SIZE} \
                --checkpoint_path=${TRAIN_DIR} \
                --eval_dir=${EVAL_DIR} \
                --dataset_name=${DATASET_NAME} \
                --dataset_split_name=validation \
                --dataset_dir=${DATASET_DIR} \
                --dataset_file_pattern=${DATASET_FILE_PATTERN} \
                --model_name=${MODEL_NAME} \
                --preprocessing_name=${PREPROCESSING_NAME}
        done
    done
done
