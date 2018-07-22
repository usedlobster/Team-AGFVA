#!/bin/bash

ITER="200"
PREFIX="real"
TRAIN_DIR="./models/train_$PREFIX_$ITER"
CONFIG_PATH="ssd_mobilenet_v1_coco_sim$PREFIX_$ITER.config"
MODEL_OUT="./fine_tuned_model_sim$PREFIX_$ITER"

rm -rf $TRAIN_DIR
python train.py --logtostderr --train_dir=$TRAIN_DIR --pipeline_config_path=$CONFIG_PATH
rm -rf $MODEL_OUT
python export_inference_graph.py --input_type image_tensor --pipeline_config_path $CONFIG_PATH --trained_checkpoint_prefix $TRAIN_DIR/model.ckpt-200 --output_directory $MODEL_OUT
