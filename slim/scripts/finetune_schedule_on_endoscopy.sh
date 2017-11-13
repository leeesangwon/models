# This script performs the following operations:
# 1. Downloads the Flowers dataset
# 2. Fine-tunes an InceptionV3 model on the Flowers training set.
# 3. Evaluates the model on the Flowers validation set.
#
# Usage:
# cd slim
# scripts\finetune_inception_resnet_v2_on_endoscopy.bat

PROJECTS_DIR=/home/sangwon/Projects/Medical

# About base model
PRETRAINED_CHECKPOINT_DIR=${PROJECTS_DIR}/PretrainedModel
MODEL_NAME=inception_resnet_v2
PREPROCESSING_NAME=inception_resnet_v2_notcrop

# About dataset
DATA_SUBNAMES=(A B C D E)
for ((i=0; i<${#DATA_SUBNAMES[*]}; i++))
do
    DATASET_DIR=${PROJECTS_DIR}/DATA/CLASSIFICATION/data_${DATA_SUBNAMES[$i]}
    DATASET_NAME=endoscopy_${DATA_SUBNAMES[$i]}
    DATASET_FILE_PATTERN=cls_data_${DATA_SUBNAMES[$i]}_0_%s_*.tfrecord

    # About training
    TRAIN_DIR=${PROJECTS_DIR}/TRAIN/CLASSIFICATION/${DATASET_NAME}-models/${MODEL_NAME}_${PREPROCESSING_NAME}/$(date '+%F %T')
    TRAIN_BATCH_SIZE=32
    EVAL_BATCH_SIZE=2
    MAX_NUMBER_OF_STEPS=10000
    EVALUATE_INTERVAL=250

    for ((j=EVALUATE_INTERVAL; j<=MAX_NUMBER_OF_STEPS; j+=EVALUATE_INTERVAL))
    do
        # Fine-tune only the new layers for 1000 steps.
        python train_image_classifier.py \
            --train_dir=${TRAIN_DIR} \
            --dataset_name=${DATASET_NAME} \
            --dataset_split_name=train \
            --dataset_dir=${DATASET_DIR} \
            --dataset_file_pattern=${DATASET_FILE_PATTERN} \
            --model_name=${MODEL_NAME} \
            --preprocessing_name=${PREPROCESSING_NAME} \
            --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/${MODEL_NAME}*.ckpt \
            --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
            --trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
            --max_number_of_steps=${j} \
            --batch_size=${TRAIN_BATCH_SIZE} \
            --save_interval_secs=600 \
            --save_summaries_secs=60 \
            --log_every_n_steps=10 \
            --optimizer=rmsprop \
            --weight_decay=0.00004

        # Run evaluation.
        python eval_image_classifier.py \
            --batch_size=${EVAL_BATCH_SIZE} \
            --checkpoint_path=${TRAIN_DIR} \
            --eval_dir=${TRAIN_DIR}/eval \
            --dataset_name=${DATASET_NAME} \
            --dataset_split_name=validation \
            --dataset_dir=${DATASET_DIR} \
            --model_name=${MODEL_NAME} \
            --preprocessing_name=${PREPROCESSING_NAME}
    done
done
