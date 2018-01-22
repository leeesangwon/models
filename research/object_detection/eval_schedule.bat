:: eval_schedule.bat
set OBJ_DTC_PATH=C:\Projects\tf\models\research\object_detection
set TRAIN_PATH=D:\Projects\Medical\TRAIN\DETECTION
set TRAIN_MODEL=rfcn_resnet101
:: hori-flip + random-crop
set EXP_NAME=2018-01-04_medical_5fold

set DATA_LIST=(0 1 2 3 4)

FOR %%i IN %DATA_LIST% do (
    python ./eval.py --logtostderr ^
        --checkpoint_dir=%TRAIN_PATH%/%TRAIN_MODEL%/%EXP_NAME%/%%i/ ^
        --eval_dir=%TRAIN_PATH%/%TRAIN_MODEL%/%EXP_NAME%/%%i/eval_detecification2 ^
        --pipeline_config_path=%TRAIN_PATH%/%TRAIN_MODEL%/%EXP_NAME%/%TRAIN_MODEL%_medical_%%i.config
)
