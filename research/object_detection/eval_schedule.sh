# train_schedule.sh
OBJ_DTC_PATH='/home/sangwon/Projects/tensorflow/models/research/object_detection'
TRAIN_PATH='/home/sangwon/Projects/Medical/TRAIN/DETECTION'
TRAIN_MODEL=ssd_mobilenet_v1
# hori-flip + random-crop
EXP_NAME=2017-11-17_20:45:14
DATA_LIST=(A)
for ((i=0; i<${#DATA_LIST[*]}; i++))
do
    python ./eval.py --logtostderr \
        --checkpoint_dir=${TRAIN_PATH}/${TRAIN_MODEL}/${EXP_NAME}/${DATA_LIST[$i]}/ \
        --eval_dir=${TRAIN_PATH}/${TRAIN_MODEL}/${EXP_NAME}/${DATA_LIST[$i]}/eval \
        --pipeline_config_path=${TRAIN_PATH}/${TRAIN_MODEL}/${EXP_NAME}/${TRAIN_MODEL}_medical_${DATA_LIST[$i]}.config
done

