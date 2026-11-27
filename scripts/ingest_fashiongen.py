"""Build the catalogue on S3 from the Fashion-Gen HDF5 file.

The Fashion-Gen training set (`fashiongen_256_256_train.h5`, ~14 GB, from Kaggle:
`kaggle datasets download -d bothin/fashiongen-validation`) holds aligned datasets:
256x256 RGB images under `input_image` and one value per item for the metadata
fields. This script writes

- one PNG per item to `<out-dir>/images/{shard:03d}/{idx}.png`, uploaded to the S3
  keys in `settings.catalogue.image_key_template` (folders of 1000 by index), and
- the metadata Parquet (`index_2`, price, the string attributes and each item's S3
  key) to `<out-dir>/metadata.parquet`, uploaded to `settings.catalogue.metadata_key`.

That is the layout the app reads (`frag.catalogue.store`). Restored from the deleted
`exploration/dataset_wrangling/split_hdf5.py` (`git show 54890dd^:...`).

Uploads skip keys that already exist (`head_object`), so a re-run uploads only what is
missing. An existing metadata Parquet is never replaced without
`--overwrite-metadata`. Needs the `ingest` dependency group:
`uv sync --group ingest`, then `uv run python scripts/ingest_fashiongen.py --help`.
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.config import Config
from botocore.exceptions import ClientError
from loguru import logger as log
from PIL import Image

from _common import confirm, start

try:
    import h5py
except ImportError:  # the `ingest` group is optional
    h5py = None

IMAGES_KEY = "input_image"
INDEX_KEY = "index_2"
PRICE_KEY = "input_msrpUSD"
STRING_ATTRIBUTES = [
    "input_brand",
    "input_category",
    "input_composition",
    "input_department",
    "input_gender",
    "input_name",
    "input_season",
    "input_subcategory",
    "input_description",
]
STRING_CODEC = "latin-1"  # the encoding of Fashion-Gen's byte strings
IMAGE_BATCH = 1024  # images read from the HDF5 file at a time

# us-east-1 list prices, for the estimate printed before uploading
S3_PUT_PER_1000 = 0.005
S3_HEAD_PER_1000 = 0.0004
S3_GB_MONTH = 0.023


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("hdf5", type=Path, help="path to the Fashion-Gen HDF5 file")
    parser.add_argument(
        "out_dir", type=Path, help="local folder for the PNGs and the Parquet"
    )
    parser.add_argument(
        "--s3-bucket", help="target bucket (default: settings.catalogue.s3_bucket)"
    )
    which = parser.add_mutually_exclusive_group()
    which.add_argument(
        "--ids-file", type=Path, help="text file with one item index per line"
    )
    which.add_argument("--limit", type=int, help="only the first N items")
    parser.add_argument("--workers", type=int, default=8, help="threads for PNG/S3")
    parser.add_argument(
        "--compression", default="zstd", help="Parquet compression codec"
    )
    parser.add_argument(
        "--no-upload", action="store_true", help="write the local files only"
    )
    parser.add_argument(
        "--overwrite-metadata",
        action="store_true",
        help="replace an existing metadata Parquet on S3",
    )
    parser.add_argument("--yes", action="store_true", help="skip the confirmation")
    return parser.parse_args()


def selected_ids(file: "h5py.File", args: argparse.Namespace) -> np.ndarray:
    """The item indexes to process, sorted."""
    total = len(file[INDEX_KEY])
    if args.ids_file:
        ids = np.array(sorted({int(x) for x in args.ids_file.read_text().split()}))
        if ids.size and (ids[0] < 0 or ids[-1] >= total):
            msg = f"ids must be in 0..{total - 1}"
            raise ValueError(msg)
        return ids
    return np.arange(min(args.limit or total, total))


def write_metadata(
    file: "h5py.File",
    ids: np.ndarray,
    key_template: str,
    path: Path,
    compression: str,
) -> None:
    """Write the metadata Parquet for `ids` (one row each, in index order)."""

    def decode(values: np.ndarray) -> list[str]:
        return [v.decode(STRING_CODEC) for v in np.ravel(values)]

    columns = {
        "s3_keys": [key_template.format(shard=i // 1000, idx=i) for i in ids],
        INDEX_KEY: np.ravel(file[INDEX_KEY][:])[ids].tolist(),
        PRICE_KEY: np.ravel(file[PRICE_KEY][:])[ids].tolist(),
    }
    for attr in STRING_ATTRIBUTES:
        columns[attr] = decode(file[attr][:][ids])
    # index_2 is the sequential dataset index the app looks items up by
    if columns[INDEX_KEY] != ids.tolist():
        msg = f"{INDEX_KEY} is not aligned with the dataset order"
        raise ValueError(msg)
    pq.write_table(pa.table(columns), path, compression=compression)
    log.info("Wrote {} metadata rows to {}.", len(ids), path)


def extract_images(
    file: "h5py.File", ids: np.ndarray, out_dir: Path, workers: int
) -> None:
    """Write each selected image as `<out_dir>/images/{shard:03d}/{idx}.png`."""

    def save(item: tuple[int, np.ndarray]) -> None:
        idx, pixels = item
        path = out_dir / "images" / f"{idx // 1000:03d}" / f"{idx}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels.astype(np.uint8)).save(path)

    images = file[IMAGES_KEY]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start_at in range(0, len(ids), IMAGE_BATCH):
            batch = ids[start_at : start_at + IMAGE_BATCH]
            first, span = int(batch[0]), int(batch[-1] - batch[0] + 1)
            if span <= 2 * len(batch):
                # dense: one contiguous read is much faster than scattered indexes
                block = images[first : first + span]
                pixels = [block[i - first] for i in batch]
            else:
                # sparse (an --ids-file): read only the selected rows (sorted)
                pixels = list(images[batch.tolist()])
            list(pool.map(save, zip(batch.tolist(), pixels, strict=True)))
            log.info("Extracted {}/{} images.", start_at + len(batch), len(ids))


def _exists(s3: object, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response["Error"]["Code"] in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise
    return True


def upload_images(  # noqa: PLR0913  the bucket layout and paths, explicitly
    s3: object,
    bucket: str,
    key_template: str,
    ids: np.ndarray,
    out_dir: Path,
    workers: int,
) -> None:
    """Upload the PNGs whose keys don't exist yet."""

    def upload(idx: int) -> bool:
        key = key_template.format(shard=idx // 1000, idx=idx)
        if _exists(s3, bucket, key):
            return False
        path = out_dir / "images" / f"{idx // 1000:03d}" / f"{idx}.png"
        s3.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": "image/png"})
        return True

    uploaded = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, was_uploaded in enumerate(pool.map(upload, ids.tolist()), start=1):
            uploaded += was_uploaded
            if n % 1000 == 0 or n == len(ids):
                log.info("Checked {}/{} images; uploaded {}.", n, len(ids), uploaded)


def main() -> None:
    """Extract, write and upload the catalogue."""
    args = parse_args()
    settings = start("ingest")
    if h5py is None:
        sys.exit("h5py is missing: run `uv sync --group ingest` first.")
    catalogue = settings.catalogue
    bucket = args.s3_bucket or catalogue.s3_bucket
    args.out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = args.out_dir / "metadata.parquet"

    with h5py.File(args.hdf5, "r") as file:
        ids = selected_ids(file, args)
        log.info("{} items selected.", len(ids))
        write_metadata(
            file, ids, catalogue.image_key_template, parquet_path, args.compression
        )
        extract_images(file, ids, args.out_dir, args.workers)
    if args.no_upload:
        return

    size_gb = (
        sum(p.stat().st_size for p in (args.out_dir / "images").rglob("*.png")) / 1e9
    )
    estimate = (
        len(ids) * (S3_PUT_PER_1000 + S3_HEAD_PER_1000) / 1000 + size_gb * S3_GB_MONTH
    )
    question = (
        f"Upload up to {len(ids)} images ({size_gb:.2f} GB) and the metadata to "
        f"s3://{bucket}. At most about ${estimate:.2f} in requests plus "
        f"${size_gb * S3_GB_MONTH:.2f}/month in storage (existing keys are skipped)."
    )
    if not confirm(question, assume_yes=args.yes):
        sys.exit("Not confirmed; nothing uploaded.")

    s3 = boto3.client(
        "s3",
        region_name=settings.aws_region,
        config=Config(max_pool_connections=args.workers * 2),
    )
    upload_images(
        s3, bucket, catalogue.image_key_template, ids, args.out_dir, args.workers
    )
    if _exists(s3, bucket, catalogue.metadata_key) and not args.overwrite_metadata:
        log.warning(
            "s3://{}/{} exists; not replaced (use --overwrite-metadata).",
            bucket,
            catalogue.metadata_key,
        )
        return
    s3.upload_file(str(parquet_path), bucket, catalogue.metadata_key)
    log.info("Uploaded the metadata to s3://{}/{}.", bucket, catalogue.metadata_key)


if __name__ == "__main__":
    main()
