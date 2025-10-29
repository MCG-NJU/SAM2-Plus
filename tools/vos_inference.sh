#bin/bash

export PYTHONPATH=$PYTHONPATH:./

path_to_dataset=""
model_config="configs/sam2.1/sam2.1_hiera_b+.yaml"
checkpoint_path="./checkpoints/sam2.1/sam2.1_hiera_base_plus.pt"
checkpoint_name=$(basename "$checkpoint_path")
path_to_output="./outputs/SAM-2.1/${checkpoint_name%.pt}/VOS/"


## Example
python ./tools/vos_inference.py \
--sam2_cfg ${model_config} \
--sam2_checkpoint ${checkpoint_path} \
--base_video_dir ./examples/JPEGImages/ \
--input_mask_dir ./examples/Annotations/ \
--output_mask_dir ${path_to_output}/examples/Annotations

python sav_dataset/sav_evaluator.py \
  --gt_root ./examples/Annotations/ \
  --pred_root ${path_to_output}/examples/Annotations/ \
  --strict \
  --num_processes `nproc`



## Tracking-Any-Granularity val
python ./tools/vos_inference.py \
--sam2_cfg ${model_config} \
--sam2_checkpoint ${checkpoint_path} \
--base_video_dir ${path_to_dataset}/Tracking-Any-Granularity/valid/JPEGImages/ \
--input_mask_dir ${path_to_dataset}/Tracking-Any-Granularity/valid/Annotations/ \
--video_list_file ${path_to_dataset}/Tracking-Any-Granularity/ImageSets/valid.txt \
--output_mask_dir ${path_to_output}/tracking_any_granularity_val_pred_pngs/Annotations

python sav_dataset/sav_evaluator.py \
  --gt_root ${path_to_dataset}/Tracking-Any-Granularity/valid/Annotations/ \
  --pred_root ${path_to_output}/tracking_any_granularity_val_pred_pngs/Annotations \
  --video_list_file ${path_to_dataset}/Tracking-Any-Granularity/ImageSets/valid.txt \
  --strict \
  --num_processes `nproc`