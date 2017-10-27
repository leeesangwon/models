python ./train.py \
    --logtostderr \
    --train_dir=`pwd`/train_rfcn_medical/4/A/ \
    --pipeline_config_path=`pwd`/rfcn_resnet101_medical_A.config
python ./train.py --logtostderr \
    --train_dir=`pwd`/train_rfcn_medical/4/C/ \
    --pipeline_config_path=`pwd`/rfcn_resnet101_medical_C.config
