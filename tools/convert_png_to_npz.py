import os
import argparse
import numpy as np
from PIL import Image
from os import path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def pack_pngs_to_npz(src_dir: str, out_npz_path: str) -> bool:
    """
    Packs all PNG files from a source directory into a single compressed NPZ file.
    """
    png_files = sorted([f for f in os.listdir(src_dir) if f.endswith(".png")])
    if not png_files:
        return False

    data_to_save = {}
    for png_file in png_files:
        try:
            arr = np.array(Image.open(path.join(src_dir, png_file)))
            unique_values = np.unique(arr)
            assert len(unique_values) <= 2, \
                f"Expected a binary mask, but found {len(unique_values)} unique values in {path.join(src_dir, png_file)}"
            if len(unique_values) <= 2 and np.all(np.isin(unique_values, [0, 1, 255])):
                arr = (arr > 0).astype(bool)
            key = path.splitext(png_file)[0]
            data_to_save[key] = arr
        except Exception as e:
            print(f"\nWarning: Could not process file {png_file} in {src_dir}. Error: {e}")
            continue

    if not data_to_save:
        return False

    os.makedirs(path.dirname(out_npz_path), exist_ok=True)
    np.savez_compressed(out_npz_path, **data_to_save)
    return True

def process_single_video(task):
    """
    Worker function to process one video directory. Designed for parallel execution.
    """
    src_video_dir, output_npz_file = task
    if path.exists(output_npz_file):
        return f"[SKIP] {output_npz_file}"

    success = pack_pngs_to_npz(src_video_dir, output_npz_file)
    if success:
        return f"[SUCCESS] {output_npz_file}"
    else:
        return f"[WARN] No PNGs in {src_video_dir}"

def convert_structure(src_root: str, dst_root: str, num_workers: int, quiet: bool):
    """
    Converts a directory of video folders to NPZ files in parallel.
    """
    if not path.isdir(src_root):
        print(f"Error: Source directory not found at '{src_root}'")
        return

    tasks = []
    for video_id in sorted(os.listdir(src_root)):
        src_video_dir = path.join(src_root, video_id)
        if path.isdir(src_video_dir):
            output_npz_file = path.join(dst_root, f"{video_id}.npz")
            tasks.append((src_video_dir, output_npz_file))

    if not tasks:
        print("No video directories found to process.")
        return

    print(f"Found {len(tasks)} video directories. Starting conversion with {num_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_single_video, tasks), total=len(tasks)))
    if not quiet:
        for res in results:
            print(res)

def main():
    parser = argparse.ArgumentParser(
        description="Convert directories of PNG masks into single NPZ files in parallel."
    )
    parser.add_argument(
        "--src_root", 
        required=True, 
        help="Source root directory containing video subfolders (e.g., 'Annotations')."
    )
    parser.add_argument(
        "--dst_root", 
        required=True, 
        help="Destination directory for the output .npz files (e.g., 'Masks')."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel processes to use. Defaults to the number of CPU cores."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output messages."
    )
    args = parser.parse_args()

    convert_structure(args.src_root, args.dst_root, args.num_workers, args.quiet)
    print("\nConversion process finished. Saved NPZ files to:", args.dst_root)

if __name__ == "__main__":
    main()