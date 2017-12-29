# train_schedule.sh
OBJ_DTC_PATH='/home/sangwon/Projects/tensorflow/models/research/object_detection'
TRAIN_PATH='/home/sangwon/Projects/Medical/TRAIN/DETECTION'
TRAIN_MODEL=rfcn_resnet101
# hori-flip + random-crop
EXP_NAME=2017-12-28
DATA_LIST=(0 1 2 3 4)
for ((i=0; i<${#DATA_LIST[*]}; i++))
do
    python ./eval.py --logtostderr \
        --checkpoint_dir=${TRAIN_PATH}/${TRAIN_MODEL}/${EXP_NAME}/${DATA_LIST[$i]}/ \
        --eval_dir=${TRAIN_PATH}/${TRAIN_MODEL}/${EXP_NAME}/${DATA_LIST[$i]}/eval_test \
        --pipeline_config_path=${TRAIN_PATH}/${TRAIN_MODEL}/${EXP_NAME}/${TRAIN_MODEL}_medical_${DATA_LIST[$i]}.config \
        # --debug
done

