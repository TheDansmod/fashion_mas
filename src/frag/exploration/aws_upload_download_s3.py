import logging

import s3fs
import h5py
import numpy as np
import boto3
from botocore.exceptions import ClientError
from PIL import Image
from dependency_injector.wiring import inject, Provide
from frag.config.container import Container

log = logging.getLogger(__name__)


def list_all_buckets():
    log.info("listing all buckets")
    s3_client = boto3.client("s3")
    response = s3_client.list_buckets()
    log.info("Existing buckets:")
    for bucket in response["Buckets"]:
        log.info(f"\t{bucket['Name']}")


# @inject
def create_bucket(cfg):
    bucket_name = cfg.exploration.s3_bucket_name
    try:
        s3_client = boto3.client("s3")
        s3_client.create_bucket(Bucket=bucket_name)
    except ClientError as e:
        log.exception("Some client error while trying to create bucket")
        return False
    return True


def upload_hdf5_to_s3_bucket(cfg):
    s3_client = boto3.client("s3")
    file_path = cfg.exploration.sample_hdf5_file_path
    bucket = cfg.exploration.s3_bucket_name
    object_name = cfg.exploration.sample_hdf5_s3_path
    try:
        s3_client.upload_file(file_path, bucket, object_name)
        log.info(f"Uploaded {file_path} → s3://{bucket}/{object_name}")
        return True
    except ClientError as e:
        log.error(f"Upload failed: {e}")
        return False


def get_item_by_index(cfg, index=3):
    # s3fs reads credentials from ~/.aws/credentials automatically
    fs = s3fs.S3FileSystem(anon=False)

    bucket = cfg.exploration.s3_bucket_name
    s3_key = cfg.exploration.sample_hdf5_s3_path
    img_dset_name = cfg.exploration.hdf5_images_dataset_name
    dsc_dset_name = cfg.exploration.hdf5_descriptions_dataset_name
    img_path = cfg.exploration.random_access_image_path.format(index=index)
    # Open the remote HDF5 file — no full download happens here
    with fs.open(f"s3://{bucket}/{s3_key}", "rb") as s3_file:
        with h5py.File(s3_file, "r") as f:
            total = f[img_dset_name].shape[0]
            log.info(f"Dataset has {total} items")

            if not (0 <= index < total):
                raise IndexError(f"Index {index} out of range (0–{total - 1})")

            # Only the relevant HDF5 chunk is fetched from S3 — not the whole file
            image = Image.fromarray(
                f[img_dset_name][index]
            )  # numpy array, e.g. (H, W, C)
            # TODO: when using actual dataset - you need to do f[dsc_dset_name][index][0] here
            description = f[dsc_dset_name][index]

            if isinstance(description, (bytes, np.bytes_)):
                # TODO: when using actual dataset - you need to use latin-1 here
                description = description.decode("utf-8")
    image.save(img_path)
    log.info("saved image")
    log.info(description)


if __name__ == "__main__":
    pass
    # from frag.

    # file code
    # get_item_by_index(cfg, index=2)
