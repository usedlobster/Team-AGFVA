#!/bin/bash

rm -r ./models/train_sim_200
python3.6 train.py --logtostderr --train_dir=./models/train_sim_200 --pipeline_config_path=ssdlite_mobilenet_v2_coco_sim_200.config

rm -r ./fine_tuned_model_sim_200
python3.6 export_inference_graph.py --input_type image_tensor --pipeline_config_path ./ssdlite_mobilenet_v2_coco_sim_200.config --trained_checkpoint_prefix ./models/train_sim_200/model.ckpt-200 --output_directory ./fine_tuned_model_sim_200
