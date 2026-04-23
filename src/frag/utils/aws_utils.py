"""AWS Related Utility Functions."""

import boto3
from botocore.exceptions import ClientError
from loguru import logger as log
from dependency_injector.wiring import inject, Provide as PV

from frag.config.container import Container

cfg = Container.config.provided

def get_all_buckets() -> list[str]:
    """Get all bucket names."""
    log.info("listing all buckets")
    s3_client = boto3.client("s3")
    response = s3_client.list_buckets()
    buckets = []
    log.debug("Existing buckets:")
    for bucket in response["Buckets"]:
        log.debug(f"\t{bucket['Name']}")
        buckets.append(bucket["Name"])
    return buckets

@inject
def create_bucket_if_not_exist(
    bucket_name: str = PV[cfg.data.aws_fashion_gen.s3_bucket_name],
) -> bool:
    if bucket_name in get_all_buckets():
        log.debug("bucket {} already exists.", bucket_name)
        return True
    try:
        s3_client = boto3.client("s3")
        s3_client.create_bucket(Bucket=bucket_name)
    except ClientError as e:
        log.exception("Some client error while trying to create bucket")
        return False
    return True

def s3_object_exists(bucket_name: str, object_key: str) -> bool:
    s3_client = boto3.client("s3")

    try:
        s3_client.head_object(Bucket=bucket_name, Key=object_key)
        return True
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")

        if error_code in ("404", "NoSuchKey", "NotFound"):
            return False

        # If you don't have permission to know whether it exists,
        # S3 may return 403 instead of 404.
        raise

@inject
def upload_hdf5_to_s3_bucket(
    file_path: str = PV[cfg.data.fashion_gen.hdf5_path],
    bucket: str = PV[cfg.data.aws_fashion_gen.s3_bucket_name],
    object_name: str = PV[cfg.data.aws_fashion_gen.dataset_object_name],
) -> bool:
    s3_client = boto3.client("s3")
    if s3_object_exists(bucket_name=bucket, object_key=object_name):
        log.debug("Object {} already exists.", object_name)
        return True
    try:
        s3_client.upload_file(file_path, bucket, object_name)
        log.info(f"Uploaded {file_path} → s3://{bucket}/{object_name}")
        return True
    except ClientError as e:
        log.error(f"Upload failed: {e}")
        return False

def upload_fashion_gen_to_s3():
    create_bucket_if_not_exist()
    log.debug("Created bucket. starting upload.")
    upload_hdf5_to_s3_bucket()

