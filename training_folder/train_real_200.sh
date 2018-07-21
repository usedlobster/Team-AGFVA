#!/bin/bash

rm -rf ./models/train_real_200
python train.py --logtostderr --train_dir=./models/train_real_200 --pipeline_config_path=ssdlite_mobilenet_v2_coco_real_200.config

rm -rf ./fine_tuned_model_real_200
python export_inference_graph.py --input_type image_tensor --pipeline_config_path ./ssdlite_mobilenet_v2_coco_real_200.config --trained_checkpoint_prefix ./models/train_real_200/model.ckpt-200 --output_directory ./fine_tuned_model_real_200
