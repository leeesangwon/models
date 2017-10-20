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
set DATASET_DIR=\tmp\data\forensics\case2
set DATASET_NAME=forensics_case2

:: About training
set TRAIN_DIR=\tmp\%DATASET_NAME%-models\%MODEL_NAME%\%PREPROCESSING_NAME%\trainable_scopes
set BATCH_SIZE=32
set MAX_NUMBER_OF_STEPS=5000

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
    --save_interval_secs=300 ^
    --save_summaries_secs=300

:: Run evaluation.
python eval_image_classifier.py ^
    --checkpoint_path=%TRAIN_DIR% ^
    --eval_dir=%TRAIN_DIR% ^
    --dataset_name=%DATASET_NAME% ^
    --dataset_split_name=validation ^
    --dataset_dir=%DATASET_DIR% ^
    --model_name=%MODEL_NAME% ^
    --preprocessing_name=%PREPROCESSING_NAME%

    :: Fine-tune only the new layers for 1000 steps.
REM python train_image_classifier.py ^
REM     --train_dir=%TRAIN_DIR%/all ^
REM     --dataset_name=%DATASET_NAME% ^
REM     --dataset_split_name=train ^
REM     --dataset_dir=%DATASET_DIR% ^
REM     --model_name=%MODEL_NAME% ^
REM     --preprocessing_name=%PREPROCESSING_NAME% ^
REM     --checkpoint_path=%TRAIN_DIR% ^
REM     --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits ^
REM     --trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits ^
REM     --max_number_of_steps=%MAX_NUMBER_OF_STEPS% ^
REM     --batch_size=%BATCH_SIZE% ^
REM     --save_interval_secs=300 ^
REM     --save_summaries_secs=300 ^
REM     REM --learning_rate=0.0001 ^
REM     REM --learning_rate_decay_type=fixed ^
REM     REM --log_every_n_steps=10 ^
REM     REM --optimizer=rmsprop ^
REM     REM --weight_decay=0.00004

REM :: Run evaluation.
REM python eval_image_classifier.py ^
REM     --checkpoint_path=%TRAIN_DIR%/all ^
REM     --eval_dir=%TRAIN_DIR%/all ^
REM     --dataset_name=%DATASET_NAME% ^
REM     --dataset_split_name=validation ^
REM     --dataset_dir=%DATASET_DIR% ^
REM     --model_name=%MODEL_NAME% ^
REM     --preprocessing_name=%PREPROCESSING_NAME%