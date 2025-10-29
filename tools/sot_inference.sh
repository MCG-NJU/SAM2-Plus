#bin/bash

export PYTHONPATH=$PYTHONPATH:./

path_to_dataset=""
model_config="configs/sam2.1/sam2.1_hiera_b+.yaml"
checkpoint_path="./checkpoints/sam2.1/sam2.1_hiera_base_plus.pt"
checkpoint_name=$(basename "$checkpoint_path")
path_to_output="./outputs/SAM-2.1/${checkpoint_name%.pt}/SOT/"


## Example
python ./tools/sot_inference.py \
--sam2_cfg ${model_config} \
--sam2_checkpoint ${checkpoint_path} \
--dataset_name customdataset \
--dataset_dir ./examples \
--output_box_dir ${path_to_output}/customdataset \
--skip_exist_result

python ./benchmarks/sot_benchmark/evaluation.py \
--dataset_name customdataset \
--dataset_dir ./examples \
--output_box_dir ${path_to_output}/customdataset


## Tracking-Any-Granularity val
python ./tools/sot_inference.py \
--sam2_cfg ${model_config} \
--sam2_checkpoint ${checkpoint_path} \
--dataset_name tracking_any_granularity_val \
--dataset_dir ${path_to_dataset}/Tracking-Any-Granularity/valid \
--output_box_dir ${path_to_output}/tracking_any_granularity_val \
--skip_exist_result

python ./benchmarks/sot_benchmark/evaluation.py \
--dataset_name tracking_any_granularity_val \
--dataset_dir ${path_to_dataset}/Tracking-Any-Granularity/valid \
--output_box_dir ${path_to_output}/tracking_any_granularity_val