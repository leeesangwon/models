:: This script performs the following operations:
:: 1. Downloads the Flowers dataset
:: 2. Fine-tunes an InceptionV3 model on the Flowers training set.
:: 3. Evaluates the model on the Flowers validation set.
::
:: Usage:
:: cd slim
:: scripts\finetune_inception_resnet_v2_on_endoscopy.bat

setlocal ENABLEDELAYEDEXPANSION

set PROJECTS_DIR=D:\Projects\Medical
set START_DATE=2018-01-22

:: About base model
set PRETRAINED_CHECKPOINT_DIR=%PROJECTS_DIR%\PretrainedModel
REM set CKPT_PATH=%PRETRAINED_CHECKPOINT_DIR%\mobilenet_v1_1.0_224.ckpt
set CKPT_PATH=%PROJECTS_DIR%\TRAIN\CLASSIFICATION\endoscopy-models\mobilenet_v1_\2018-01-22_3_Image_jysun_2nd_step

set MODEL_NAME=mobilenet_v1

:: About dataset
set DATA_CROSS_VAL=(0)
set FOLDER_NAME=%START_DATE%_3_Image_jysun_3rd_step

:: About training
set TRAIN_BATCH_SIZE=32
set EVAL_BATCH_SIZE=2
set MAX_NUMBER_OF_STEPS=20000
set EVALUATE_INTERVAL=1000
set START_STEP=11000

set DATASET_DIR=C:\Projects\Medical\DATA\CLASSIFICATION\threeImage_0117
set DATASET_NAME=endoscopy

FOR %%i IN %DATA_CROSS_VAL% do (
    for /L %%j IN (%START_STEP%, %EVALUATE_INTERVAL%, %MAX_NUMBER_OF_STEPS%) do (
        :: Fine-tune
        python train_image_classifier.py ^
            --train_dir=%PROJECTS_DIR%\TRAIN\CLASSIFICATION\%DATASET_NAME%-models\%MODEL_NAME%_%PREPROCESSING_NAME%\%FOLDER_NAME%\%%i ^
            --dataset_name=%DATASET_NAME% ^
            --dataset_split_name=train ^
            --dataset_dir=%DATASET_DIR% ^
            --dataset_file_pattern=classification_data_of_3_images_%%i_%%s_224_20180117.tfrecord ^
            --model_name=%MODEL_NAME% ^
            --checkpoint_path=%CKPT_PATH%\%%i\model.ckpt-10000 ^
            --max_number_of_steps=%%j ^
            --batch_size=%TRAIN_BATCH_SIZE% ^
            --save_interval_secs=600 ^
            --save_summaries_secs=60 ^
            --log_every_n_steps=10 ^
            --learning_rate=0.00001 ^
            --learning_rate_decay_type=fixed ^
            --optimizer=rmsprop ^
            --weight_decay=0.00004 ^
            --ignore_missing_vars=True

        :: Run evaluation.
        python eval_image_classifier.py ^
            --batch_size=%EVAL_BATCH_SIZE% ^
            --checkpoint_path=%PROJECTS_DIR%\TRAIN\CLASSIFICATION\%DATASET_NAME%-models\%MODEL_NAME%_%PREPROCESSING_NAME%\%FOLDER_NAME%\%%i ^
            --eval_dir=%PROJECTS_DIR%\TRAIN\CLASSIFICATION\%DATASET_NAME%-models\%MODEL_NAME%_%PREPROCESSING_NAME%\%FOLDER_NAME%\%%i\eval ^
            --dataset_name=%DATASET_NAME% ^
            --dataset_split_name=test ^
            --dataset_dir=%DATASET_DIR% ^
            --dataset_file_pattern=classification_data_of_3_images_%%i_%%s_224_20180117.tfrecord ^
            --model_name=%MODEL_NAME% 
    )
)