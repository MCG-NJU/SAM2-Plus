#bin/bash

export PYTHONPATH=$PYTHONPATH:./

path_to_dataset=""
model_config="configs/sam2.1/sam2.1_hiera_b+_predmasks_decoupled_MAME.yaml"
checkpoint_path="./checkpoints/SAM2-Plus/checkpoint_phase123.pt"
checkpoint_name=$(basename "$checkpoint_path")
path_to_output="./outputs/SAM2-Plus/${checkpoint_name%.pt}/PT/"
radius=5
sigma=2


## Tracking-Any-Granularity val
python ./tools/pt_inference_plus.py \
--sam2_cfg ${model_config} \
--sam2_checkpoint ${checkpoint_path} \
--dataset_name tracking_any_granularity_val \
--dataset_dir ${path_to_dataset}/Tracking-Any-Granularity/valid \
--output_pt_dir ${path_to_output}/tracking_any_granularity_val \
--skip_exist_result \
--radius $radius --sigma $sigma

python ./benchmarks/pt_benchmark/evaluation_method.py \
--dataset_name tracking_any_granularity_val \
--metrics_type tapvid \
--dataset_dir ${path_to_dataset}/Tracking-Any-Granularity/valid \
--groundtruth_npz_dir ${path_to_dataset}/Tracking-Any-Granularity/valid/Points \
--pred_npz_dir ${path_to_output}/tracking_any_granularity_val/result_point