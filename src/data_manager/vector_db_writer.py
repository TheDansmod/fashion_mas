"""Read the fashion-gen dataset, generate embeddings, and write them to vector db."""

import h5py
import numpy as np
import open_clip
import torch
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models
from loguru import logger as log
from dependency_injector.wiring import inject, Provide as PV

from src.config.container import Container

cfg = Container.config.provided


class FashionSigLIPEmbedding:
    """Create and return multi-modal embeddings."""

    def __init__(self, cfg):
        """Initialise device, model, image and text processor, and batch size.

        None of the attributes are intended for external use. So all of them start
        with _.
        """
        self._device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(self._device_type)
        self._model, _, self._preprocess_val = open_clip.create_model_and_transforms(
            cfg.data.vector_db.embedding_model
        )
        self._tokenizer = open_clip.get_tokenizer(cfg.data.vector_db.embedding_model)
        self._model.to(self._device)
        self._embed_batch_size = cfg.data.data_processing.embedding_batch_size

    def get_image_embedding_batch(self, images):
        """Generates embeddings for an array of images.

        We iterate through the images and give them to the model to encode in batches.

        Args:
            images (numpy ndarray): Array of images of shape [B, H, W, C].

        Returns:
            results (list[list[float]]): It returns a list of embeddings. Each embedding
                is a python list of floats length
                cfg.data.data_processing.embedding_size.
        """
        results = []
        for i in range(0, images.shape[0], self._embed_batch_size):
            batch = images[i : i + self._embed_batch_size]
            tensor_list = [self._preprocess_val(Image.fromarray(img)) for img in batch]
            batched_image_tensor = torch.stack(tensor_list, dim=0).to(self._device)
            with torch.no_grad(), torch.amp.autocast(device_type=self._device_type):
                image_features = (
                    self._model.encode_image(batched_image_tensor, normalize=True)
                    .cpu()
                    .tolist()
                )
            results.extend(image_features)
        return results

    def get_text_embedding_batch(self, texts):
        """Generates embeddings for a list of texts.

        We iterate through the texts and give them to the model to encode in batches.

        Args:
            texts (list[str]): A list of string texts.

        Returns:
            results (list[list[float]]): It returns a list of embeddings. Each embedding
                is a python list of floats of length
                cfg.data.data_processing.embedding_size.
        """
        results = []
        for i in range(0, len(texts), self._embed_batch_size):
            batch = texts[i : i + self._embed_batch_size]
            batched_texts = self._tokenizer(batch).to(self._device)
            with torch.no_grad(), torch.amp.autocast(device_type=self._device_type):
                text_features = (
                    self._model.encode_text(batched_texts, normalize=True)
                    .cpu()
                    .tolist()
                )
            results.extend(text_features)
        return results

    def get_paired_embedding_batch(self, images, texts):
        """Generates embeddings for batched images and texts and returns them paired.

        If the images and the texts on corresponding indices are related (eg: one is
        the description of the other, we often want the embeddings to also be
        together). That is what this function does. It is mostly a utility function.

        Args:
            images (numpy ndarray): Array of images of shape [B, H, W, C].
                B = len(texts)
            texts (list[str]): A list of string texts. len(texts) = B

        Returns:
            results (list[tuple[list[float], list[float]]]): Returns a list of pairs
                of embeddings. Each emebdding is a list of floats of length given by
                cfg.data.data_processing.embedding_size.
        """
        img_results = self.get_image_embedding_batch(images)
        text_results = self.get_text_embedding_batch(texts)
        return list(zip(img_results, text_results))


def get_vector_db_client(cfg):
    """Create the vector db client,and collection, and returns the client.

    Depending on the setup the existing collection might be deleted and a new
    collection created. Or if the collection already exists and the recreate flag is
    not enabled, then the existing collection is fetched.
    """
    client = QdrantClient(
        url=cfg.data.vector_db.vector_store_network_path,
        prefer_grpc=cfg.data.vector_db.prefer_grpc,
    )

    recreate = cfg.data.vector_db.recreate
    collection_name = cfg.data.vector_db.collection_name
    image_vectors_name = cfg.data.vector_db.image_vectors_name
    text_vectors_name = cfg.data.vector_db.text_vectors_name
    embedding_size = cfg.data.data_processing.embedding_size

    if recreate and client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        log.info(f"Deleted existing collection '{collection_name}'.")

    vectors_on_disk = cfg.data.vector_db.vectors_on_disk
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                image_vectors_name: models.VectorParams(
                    size=embedding_size,
                    distance=models.Distance.DOT,
                    on_disk=vectors_on_disk,
                ),
                text_vectors_name: models.VectorParams(
                    size=embedding_size,
                    distance=models.Distance.COSINE,
                    on_disk=vectors_on_disk,
                ),
            },
            on_disk_payload=cfg.data.vector_db.payload_on_disk,
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=cfg.data.vector_db.indexing_threshold
            ),
            hnsw_config=models.HnswConfigDiff(on_disk=cfg.data.vector_db.hnsw_on_disk),
        )
        log.info(f"Created new collection '{collection_name}'...")
    else:
        log.info(f"Collection '{collection_name}' already exists. Appending.")

    return client


def get_fashion_gen_data(cfg, from_idx, to_idx):
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
    images_key = cfg.data.fashion_gen.images_key
    prices_key = cfg.data.fashion_gen.prices_key
    index_key = cfg.data.fashion_gen.index_key
    num_datapoints = cfg.data.fashion_gen.num_datapoints
    codec = cfg.data.fashion_gen.string_codec
    if from_idx >= num_datapoints or from_idx >= to_idx:
        return data
    else:
        from_idx = max(0, from_idx)
    if to_idx >= num_datapoints:
        to_idx = num_datapoints
    vec_decode = np.vectorize(pyfunc=lambda x: x.decode(codec))
    with h5py.File(cfg.data.fashion_gen.hdf5_path, "r") as file:
        data[images_key] = file[images_key][from_idx:to_idx].astype("uint8")
        data[prices_key] = np.ravel(file[prices_key][from_idx:to_idx]).tolist()
        data[index_key] = file[index_key][from_idx:to_idx].tolist()  # don't need ravel
        for key in cfg.data.fashion_gen.string_attributes:
            data[key] = vec_decode(np.ravel(file[key][from_idx:to_idx])).tolist()
    return data


def _iter_data_points(cfg, embedder):
    image_vec_name = cfg.data.vector_db.image_vectors_name
    text_vec_name = cfg.data.vector_db.text_vectors_name

    images_key = cfg.data.fashion_gen.images_key
    descriptions_key = cfg.data.fashion_gen.descriptions_key
    prices_key = cfg.data.fashion_gen.prices_key
    index_key = cfg.data.fashion_gen.index_key

    num_points = cfg.data.fashion_gen.num_datapoints
    fetch_batch_sz = cfg.data.data_processing.data_fetch_batch_size

    for from_idx in range(0, num_points, fetch_batch_sz):
        to_idx = min(from_idx + fetch_batch_sz, num_points)
        data = get_fashion_gen_data(cfg, from_idx, to_idx)

        img_descr_pairs = embedder.get_paired_embedding_batch(
            data[images_key], data[descriptions_key]
        )
        log.debug(
            f"Got embedding {from_idx} to {to_idx} out of {num_points} datapoints."
        )
        for idx, (img_vec, text_vec) in enumerate(img_descr_pairs):
            # construct the payload
            string_payload = {
                key: data[key][idx] for key in cfg.data.fashion_gen.string_attributes
            }
            payload = {
                prices_key: data[prices_key][idx],
                index_key: data[index_key][idx],
                **string_payload,
            }
            # construct the named vectors
            named_vectors = {image_vec_name: img_vec, text_vec_name: text_vec}
            # construct the point struct from the payload and named vectors
            yield models.PointStruct(
                id=data[index_key][idx], vector=named_vectors, payload=payload
            )


def populate_vector_db(cfg):
    """Read fashion-gen HDF5 data, generate embeddings, write it into the vector db.

    We fetch some datapoints (cfg.data.data_processing.data_fetch_batch_size) from the
    fashion-gen hdf5 dataset file (so as not to fill up RAM), generate the embeddings
    for the images and descriptions, and insert these and the other attributes as
    metadata into the vector database.
    """
    # preliminaries
    embedder = FashionSigLIPEmbedding(cfg)
    client = get_vector_db_client(cfg)

    # insert the points into the collection
    client.upload_points(
        collection_name=cfg.data.vector_db.collection_name,
        points=_iter_data_points(cfg, embedder),
        batch_size=cfg.data.data_processing.upload_batch_size,
        parallel=cfg.data.data_processing.upload_parallel,
        wait=False,
    )
