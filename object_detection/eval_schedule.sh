OBJ_DTC_PATH='/home/sangwon/Projects/tensorflow/models/object_detection'
TRAIN_DIR=${OBJ_DTC_PATH}/train_rfcn_medical
EXP_NAME='7'
DATA_LIST=(D)
for ((i=0; i<${#DATA_LIST[*]}; i++))
do
    python ./eval.py --logtostderr \
        --checkpoint_dir=${TRAIN_DIR}/${EXP_NAME}/${DATA_LIST[$i]}/ \
        --eval_dir=${TRAIN_DIR}/${EXP_NAME}/${DATA_LIST[$i]}/eval2 \
        --pipeline_config_path=${TRAIN_DIR}/${EXP_NAME}/rfcn_resnet101_medical_${DATA_LIST[$i]}.config
done

