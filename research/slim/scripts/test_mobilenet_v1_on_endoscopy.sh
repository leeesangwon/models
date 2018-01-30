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
export FOLDER_NAME=2018-01-17_not_crop_full_image

# About training
export TRAIN_BATCH_SIZE=32
export EVAL_BATCH_SIZE=2
export MAX_NUMBER_OF_STEPS=30000
export EVALUATE_INTERVAL=500
export START_STEP=0

for ((i=0; i<${#DATA_SUBNAMES[*]}; i++))
do
    for ((k=0; k<${#DATA_CROSS_VAL[*]}; k++))
    do
        # DATASET_DIR=${PROJECTS_DIR}/DATA/CLASSIFICATION/crop_bbox/test/data_${DATA_SUBNAMES[$i]}
        DATASET_DIR=${PROJECTS_DIR}/DATA/CLASSIFICATION/crop_bbox/test/data_${DATA_SUBNAMES[$i]}

        DATASET_NAME=endoscopy_${DATA_SUBNAMES[$i]}
        # DATASET_FILE_PATTERN=cls_data_${DATA_SUBNAMES[$i]}_0_%s_*.tfrecord
        DATASET_FILE_PATTERN=cls_data_${DATA_SUBNAMES[$i]}_${DATA_CROSS_VAL[${k}]}_%s_*.tfrecord

        # About training
        TRAIN_DIR=${PROJECTS_DIR}/TRAIN/CLASSIFICATION/${DATASET_NAME}-models/${MODEL_NAME}_${PREPROCESSING_NAME}/${FOLDER_NAME}/${DATA_CROSS_VAL[${k}]}
        EVAL_DIR=${TRAIN_DIR}/eval_test

        # Run evaluation.
        python eval_image_classifier_from_0_to_end.py \
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
