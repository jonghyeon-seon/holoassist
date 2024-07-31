#!/bin/bash

python run_net.py --cfg configs/fine_action_recognition.yaml PATH_TO_DATA_DIR <PATH_TO_DATA_DIR> LABEL_DIR data_2221 OUTPUT_DIR <OUTPUT_DIR> NUM_GPUS 4 TRAIN.BATCH_SIZE 32  SOLVER.BASE_LR 0.01 SOLVER.WARMUP_START_LR 0.005 TASKS rgb,hands-left,hands-right