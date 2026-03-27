from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.screenspot import load_screenspot_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-download ScreenSpot data needed for the benchmark.")
    parser.add_argument(
        "--repo-id",
        default="lscpku/ScreenSpot",
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--parquet-path",
        default="data/test-00000-of-00002.parquet",
        help="Parquet file within the dataset repo.",
    )
    parser.add_argument(
        "--revision",
        help="Optional Hugging Face revision.",
    )
    args = parser.parse_args()

    samples = load_screenspot_samples(
        repo_id=args.repo_id,
        parquet_path=args.parquet_path,
        revision=args.revision,
    )
    print(f"downloaded {len(samples)} ScreenSpot samples")


if __name__ == "__main__":
    main()
