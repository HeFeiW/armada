# check the structure of a zarr dataset
import zarr
import argparse
from pathlib import Path

def check_zarr_dataset(zarr_path: Path):
    """Check the structure of a zarr dataset."""
    if not zarr_path.exists():
        print(f"Zarr path {zarr_path} does not exist.")
        return

    try:
        zarr_dataset = zarr.open(str(zarr_path), mode='r')
        print(f"Successfully opened zarr dataset at {zarr_path}")
        print("Dataset structure:")
        print(zarr_dataset.tree())
        # 读取meta中的episode_ends,划分对应的episodes，并打印每个episode最后一帧的tcp_pose
        if 'meta' in zarr_dataset and 'episode_ends' in zarr_dataset['meta']:
            episode_ends = zarr_dataset['meta']['episode_ends'][:]
            print(f"Found episode_ends in meta: {episode_ends}")
            for i, end_idx in enumerate(episode_ends):
                tcp_pose = zarr_dataset['data']['tcp_pose'][end_idx - 1]  # 获取每个episode最后一帧的tcp_pose
                print(f"Episode {i}: last frame index={end_idx - 1}, tcp_pose={tcp_pose}")
        else:
            print("No episode_ends found in meta.")
    except Exception as e:
        print(f"Error opening zarr dataset: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the structure of a zarr dataset.")
    parser.add_argument("zarr_path", type=Path, help="Path to the zarr dataset")
    args = parser.parse_args()

    check_zarr_dataset(args.zarr_path)