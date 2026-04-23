from typing import Any
import base64

from loguru import logger as log
from dependency_injector.wiring import inject, Provide as PV

from frag.config.container import Container

cfg = Container.config.provided

async def get_fashion_gen_data(
    fetch_index,
    num_datapoints: int = PV[cfg.data.fashion_gen.num_datapoints],
    prices_key: str = PV[cfg.data.fashion_gen.prices_key],
    categories_key: str = PV[cfg.data.fashion_gen.categories_key],
    descriptions_key: str = PV[cfg.data.fashion_gen.descriptions_key],
    s3_client = PV[Container.s3_client.provided],
    bucket_name: str = PV[cfg.data.aws_fashion_gen.s3_bucket_name],
    metadata_lookup: dict[int, Any] = PV[Container.metadata_lookup.provided],
):
    """Get data from the fashion-gen dataset in dictionary format.

    We will be extracting all the attributes in the dataset in a dictionary. What
    data to fetch is determined by the from_idx (included) and to_idx (excluded)
    values. If there is no data within the provided bounds then we return empty
    dictionary.

    Returns:
        data (dict): This dictionary contains the datapoints from the requested
            indices from the fashion-gen dataset. Each key in the dictionary
            corresponds to the name of a dataset in the fashion-gen hdf5 file.
            The value of each key in the dictionary varies depending on requirement.
            If we are sending back images, they are send back as numpy ndarrays. If
            we are sending back string values, they are lists of strings. If we are
            sending back floats, they are lists of floats.
    """
    data = dict()
    if not 0 <= fetch_index < num_datapoints:
        return data
    image_key = f"images/{fetch_index // 1000:03d}/{fetch_index}.png"
    response = s3_client.get_object(Bucket=bucket_name, Key=image_key)
    data["image"] = base64.b64encode(response["Body"].read()).decode("utf-8")
    data["price"] = metadata_lookup[fetch_index][prices_key]
    data["category"] = metadata_lookup[fetch_index][categories_key]
    data["description"] = metadata_lookup[fetch_index][descriptions_key]
    data["id"] = fetch_index
    return data

