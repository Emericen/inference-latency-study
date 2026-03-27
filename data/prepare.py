from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.screenspot import prepare_bucketed_images


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local ScreenSpot payload buckets for the benchmark.")
    parser.add_argument(
        "--config",
        default="configs/screenspot_payload_buckets.yaml",
        help="Config YAML that defines the ScreenSpot bucket cases.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/prepared/screenspot_payload_buckets",
        help="Directory for prepared local JPEGs and manifest.",
    )
    parser.add_argument(
        "--per-bucket",
        type=int,
        default=12,
        help="How many prepared images to store per bucket.",
    )
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cases = config["cases"]
    if not cases:
        raise ValueError(f"No cases found in config: {config_path}")

    first_payload = cases[0]["payload"]
    repo_id = first_payload["screenspot_repo_id"]
    parquet_path = first_payload["screenspot_parquet_path"]
    revision = first_payload.get("screenspot_revision")
    bucket_targets = [int(case["payload"]["target_image_bytes"]) for case in cases]

    output_dir = ROOT / args.output_dir
    manifest_path = output_dir / "manifest.json"
    prepare_bucketed_images(
        repo_id=repo_id,
        parquet_path=parquet_path,
        revision=revision,
        output_dir=output_dir,
        manifest_path=manifest_path,
        bucket_targets=bucket_targets,
        per_bucket=args.per_bucket,
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
