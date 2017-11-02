# train_schedule.sh
OBJ_DTC_PATH='/path/to/tensorflow/models/object_detection'
# hori-flip + random-crop
EXP_NAME='7'
DATA_LIST=(A C B D)
for ((i=0; i<${#DATA_LIST[*]}; i++))
do
    python ./train.py \
        --logtostderr \
        --train_dir=${OBJ_DTC_PATH}/train_rfcn_medical/${EXP_NAME}/${DATA_LIST[$i]}/ \
        --pipeline_config_path=${OBJ_DTC_PATH}/train_rfcn_medical/${EXP_NAME}/rfcn_resnet101_medical_${DATA_LIST[$i]}.config
done