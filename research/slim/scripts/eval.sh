PROJECTS_DIR=/home/sangwon/Projects/Medical
START_DATE=$(date '+%F')

# About base model
MODEL_NAME=inception_resnet_v2
PREPROCESSING_NAME=inception_resnet_v2_notcrop
FOLDER_NAME=${START_DATE}_lastlayer

# About dataset
DATA_SUBNAMES=(A)
DATA_CROSS_VAL=(0 1 2 3 4)

EVAL_BATCH_SIZE=2

for ((i=0; i<${#DATA_SUBNAMES[*]}; i++))
do
    for ((k=0; k<${#DATA_CROSS_VAL[*]}; k++))
    do    
        DATASET_DIR=${PROJECTS_DIR}/DATA/CLASSIFICATION/data1116+1117/trainval/data_${DATA_SUBNAMES[$i]}
        DATASET_NAME=endoscopy_${DATA_SUBNAMES[$i]}
        DATASET_FILE_PATTERN=cls_data_${DATA_SUBNAMES[$i]}_${DATA_CROSS_VAL[${k}]}_%s_*.tfrecord
        
        TRAIN_DIR=${PROJECTS_DIR}/TRAIN/CLASSIFICATION/${DATASET_NAME}-models/${MODEL_NAME}_${PREPROCESSING_NAME}/${FOLDER_NAME}/${DATA_CROSS_VAL[${k}]}
        EVAL_DIR=${TRAIN_DIR}/eval2

        python eval_image_classifier_from_0_to_end.py \
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
