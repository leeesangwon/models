# train_schedule.sh
OBJ_DTC_PATH='/home/sangwon/Projects/tensorflow/models/research/object_detection'
TRAIN_PATH='/home/sangwon/Projects/Medical/TRAIN/DETECTION'
TRAIN_MODEL=rfcn_resnet101
# hori-flip + random-crop
START_DATE=$(date '+%F')
EXP_NAME=$START_DATE
DATA_LIST=(0 1 2 3 4)
for ((i=0; i<${#DATA_LIST[*]}; i++))
do
    python ./train.py \
        --logtostderr \
        --train_dir=${TRAIN_PATH}/${TRAIN_MODEL}/${EXP_NAME}/${DATA_LIST[$i]}/ \
        --pipeline_config_path=${OBJ_DTC_PATH}/config/${TRAIN_MODEL}_medical_${DATA_LIST[$i]}.config
    cp ${OBJ_DTC_PATH}/config/${TRAIN_MODEL}_medical_${DATA_LIST[$i]}.config \
	    ${TRAIN_PATH}/${TRAIN_MODEL}/${EXP_NAME}/${TRAIN_MODEL}_medical_${DATA_LIST[$i]}.config
done
