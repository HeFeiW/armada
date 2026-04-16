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
    except Exception as e:
        print(f"Error opening zarr dataset: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the structure of a zarr dataset.")
    parser.add_argument("zarr_path", type=Path, help="Path to the zarr dataset")
    args = parser.parse_args()

    check_zarr_dataset(args.zarr_path)