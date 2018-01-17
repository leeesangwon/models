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
export MODEL_NAME='mobilenet_v1'
export PREPROCESSING_NAME='inception_resnet_v2_notcrop'
export EXCLUDE_SCOPES='MobilenetV1/Logits'
# export TRAINABLE_SCOPES='InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits'

# About dataset
export DATA_SUBNAMES=(A)
export DATA_CROSS_VAL=(0)
export FOLDER_NAME=${START_DATE}_not_crop_full_image

# About training
export TRAIN_BATCH_SIZE=32
export EVAL_BATCH_SIZE=2
export MAX_NUMBER_OF_STEPS=50000
export EVALUATE_INTERVAL=500
export START_STEP=0

for ((i=0; i<${#DATA_SUBNAMES[*]}; i++))
do
    for ((k=0; k<${#DATA_CROSS_VAL[*]}; k++))
    do
        DATASET_DIR=${PROJECTS_DIR}/DATA/CLASSIFICATION/crop_bbox/trainval/data_${DATA_SUBNAMES[$i]}
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
                --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/mobilenet_v1_1.0_224.ckpt \
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
