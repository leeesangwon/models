:: This script performs the following operations:
:: 1. Downloads the Flowers dataset
:: 2. Fine-tunes an InceptionV3 model on the Flowers training set.
:: 3. Evaluates the model on the Flowers validation set.
::
:: Usage:
:: cd slim
:: scripts\finetune_inception_resnet_v2_on_endoscopy.bat

:: About base model
set PRETRAINED_CHECKPOINT_DIR=\tmp\checkpoints
set MODEL_NAME=inception_resnet_v2
set PREPROCESSING_NAME=inception_resnet_v2_notcrop

:: About dataset
set DATASET_DIR=\tmp\data\forensics\case3
set DATASET_NAME=forensics_case3

:: About training
set TRAIN_DIR=\tmp\%DATASET_NAME%-models\%MODEL_NAME%\%PREPROCESSING_NAME%
set BATCH_SIZE=32
set MAX_NUMBER_OF_STEPS=20000

:: Fine-tune only the new layers for 1000 steps.
python train_image_classifier.py ^
    --train_dir=%TRAIN_DIR% ^
    --dataset_name=%DATASET_NAME% ^
    --dataset_split_name=train ^
    --dataset_dir=%DATASET_DIR% ^
    --model_name=%MODEL_NAME% ^
    --preprocessing_name=%PREPROCESSING_NAME% ^
    --checkpoint_path=%PRETRAINED_CHECKPOINT_DIR%\%MODEL_NAME%*.ckpt ^
    --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits ^
    --trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits ^
    --max_number_of_steps=%MAX_NUMBER_OF_STEPS% ^
    --batch_size=%BATCH_SIZE% ^
    REM --learning_rate=0.0001 ^
    REM --learning_rate_decay_type=fixed ^
    REM --save_interval_secs=60 ^
    REM --save_summaries_secs=60 ^
    REM --log_every_n_steps=10 ^
    REM --optimizer=rmsprop ^
    REM --weight_decay=0.00004

:: Run evaluation.
python eval_image_classifier.py ^
    --checkpoint_path=%TRAIN_DIR% ^
    --eval_dir=%TRAIN_DIR% ^
    --dataset_name=%DATASET_NAME% ^
    --dataset_split_name=validation ^
    --dataset_dir=%DATASET_DIR% ^
    --model_name=%MODEL_NAME% ^
    --preprocessing_name=%PREPROCESSING_NAME%