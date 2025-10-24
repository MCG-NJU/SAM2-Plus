import os
import argparse
from benchmarks.sot_benchmark.datasets import construct_dataset
from benchmarks.sot_benchmark.utils import get_and_save_results

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_name",
    type=str,
    required=True,
    help="dataset name",
)
parser.add_argument(
    "--dataset_dir",
    type=str,
    required=True,
    help="directory containing videos (as JPEG files) to run VOS prediction on",
)
parser.add_argument(
    "--output_box_dir",
    type=str,
    required=True,
    help="directory to save the output masks (as PNG files)",
)
args = parser.parse_args()

dataset = construct_dataset(args.dataset_name, args.dataset_dir)

# Analysis and get results
eval_result = get_and_save_results(
    track_result_dir=os.path.join(args.output_box_dir, "result_bbox"),
    dataset=dataset,
    result_file_path=os.path.join(args.output_box_dir, "result.txt"),
)
print(eval_result)