# the regular hdf5 file with s3fs did not work out - the usage is quite slow
# so i have decided to split the 14GB file into individual files and gather
# the metadata into a single parquet file

# this file was run by importing the main method of this file into cli_main.py
# and then calling it from the main function in cli_main.py. Be careful and ensure
# that the setup and teardown for the container is executed.

import random
from concurrent.futures import ThreadPoolExecutor

import h5py
import boto3
import botocore
import boto3.s3.transfer as s3transfer
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from loguru import logger as log
from dependency_injector.wiring import inject, Provide as PV

from frag.config.container import Container

cfg = Container.config.provided

@inject
def create_metadata_file(
    hdf5_path: str = PV[cfg.data.fashion_gen.hdf5_path],
    string_attributes: list[str] = PV[cfg.data.fashion_gen.string_attributes],
    metadata_path: str = PV[cfg.exploration.fashion_gen_metadata_path],
    compression: str = PV[cfg.exploration.parquet_compression],
    codec: str = PV[cfg.data.fashion_gen.string_codec],
    num_datapoints: int = PV[cfg.data.fashion_gen.num_datapoints],
):
    columns = dict()
    vec_decode = np.vectorize(pyfunc=lambda x: x.decode(codec))
    s3_keys = []
    for idx in range(num_datapoints):
        s3_keys.append(f"images/{idx // 1000:03d}/{idx}.png")
    columns["s3_keys"] = s3_keys
    with h5py.File(hdf5_path, "r") as file:
        columns["index_2"] = file["index_2"][:].tolist()
        columns["input_msrpUSD"] = np.ravel(file["input_msrpUSD"][:]).tolist()
        for attr in string_attributes:
            columns[attr] = vec_decode(np.ravel(file[attr][:])).tolist()
    log.debug("obtained columns")
    df = pd.DataFrame(columns)
    log.debug("Created dataframe")
    log.debug(df.info(verbose=True))
    table = pa.Table.from_pandas(df)
    pq.write_table(table, metadata_path, compression=compression)
    log.debug("written table")

def _save_image(args):
    img_idx, numpy_image, save_folder = args
    img = Image.fromarray(numpy_image)
    img.save(f"{save_folder}/{img_idx}.png")


@inject
def extract_images(
    batch_size: int = PV[cfg.exploration.image_extraction_batch_size],
    save_folder: str = PV[cfg.exploration.image_extraction_folder_path],
    hdf5_path: str = PV[cfg.data.fashion_gen.hdf5_path],
    num_datapoints: int = PV[cfg.data.fashion_gen.num_datapoints],
    max_workers: int = PV[cfg.exploration.image_extraction_max_workers],
    images_key: str = PV[cfg.data.fashion_gen.images_key],
):
    with h5py.File(hdf5_path, "r") as file:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for start in range(0, num_datapoints, batch_size):
                end = min(start + batch_size, num_datapoints)
                batch = file[images_key][start: end].astype(np.uint8)
                executor.map(_save_image, [(start + i, batch[i], save_folder) for i in range(len(batch))])
                log.debug("done from {} to {} of {}", start, end, num_datapoints)

@inject
def upload_images_to_s3(
    max_pool_size: int = PV[cfg.exploration.s3_upload_max_pool_size],
    num_datapoints: int = PV[cfg.data.fashion_gen.num_datapoints],
    bucket_name: str = PV[cfg.data.aws_fashion_gen.s3_bucket_name],
    save_folder: str = PV[cfg.exploration.image_extraction_folder_path],
):
    session = boto3.session.Session()
    botocore_config = botocore.config.Config(max_pool_connections=max_pool_size)
    s3client = session.client("s3", config=botocore_config)
    transfer_config = s3transfer.TransferConfig(use_threads=True, max_concurrency=max_pool_size)
    s3t = s3transfer.create_transfer_manager(s3client, transfer_config)

    for idx in range(0, num_datapoints):
        key = f"images/{idx // 1000:03d}/{idx}.png"
        file_path = f"{save_folder}/{idx}.png"
        s3t.upload(file_path, bucket_name, key)
        if idx % 1000 == 0:
            log.debug("done {} of {}; {}%", idx + 1, num_datapoints, ((idx + 1) * 100) / num_datapoints)
    # blocks until all uploads finish
    s3t.shutdown()

@inject
def misc_explorer(
    hdf5_path: str = PV[cfg.data.fashion_gen.hdf5_path],
    num_datapoints: int = PV[cfg.data.fashion_gen.num_datapoints],
):
    for i in range(50):
        idx = random.randint(0, num_datapoints)
        log.debug(f"bucket_name/images/{idx // 1000:03d}/{idx}.png")

def main():
    upload_to_s3()
