PROJECTS_DIR=/home/sangwon/Projects/Medical

# About base model
PRETRAINED_CHECKPOINT_DIR=${PROJECTS_DIR}/PretrainedModel
MODEL_NAME=inception_resnet_v2
PREPROCESSING_NAME=inception_resnet_v2_notcrop
# START_DATE=$(date '+%F_%T')
START_DATE=2017-11-10

# About dataset
DATA_SUBNAMES=(E)

for ((i=0; i<${#DATA_SUBNAMES[*]}; i++))
do
    DATASET_DIR=${PROJECTS_DIR}/DATA/CLASSIFICATION/data_${DATA_SUBNAMES[$i]}
    DATASET_NAME=endoscopy_${DATA_SUBNAMES[$i]}
    DATASET_FILE_PATTERN=cls_data_${DATA_SUBNAMES[$i]}_0_%s_*.tfrecord
    TRAIN_DIR=${PROJECTS_DIR}/TRAIN/CLASSIFICATION/${DATASET_NAME}-models/${MODEL_NAME}_${PREPROCESSING_NAME}/2017-11-10
    EVAL_BATCH_SIZE=2

    python eval_image_classifier_from_0_to_end.py \
        --batch_size=${EVAL_BATCH_SIZE} \
        --checkpoint_path=${TRAIN_DIR} \
        --eval_dir=${TRAIN_DIR}/eval2 \
        --dataset_name=${DATASET_NAME} \
        --dataset_split_name=validation \
        --dataset_dir=${DATASET_DIR} \
        --dataset_file_pattern=${DATASET_FILE_PATTERN} \
        --model_name=${MODEL_NAME} \
        --preprocessing_name=${PREPROCESSING_NAME}
done
