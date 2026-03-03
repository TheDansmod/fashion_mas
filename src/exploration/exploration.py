"""Explore various things to see how they work before incorporation.

1. We are going to explore how qwen3-vl can be used to describe images.
2. Figure out how data is stored in the h5 file for fashion-gen, and
"""

import logging

log = logging.getLogger(__name__)


def test_qwen(cfg):
    import hydra
    from langchain_core.messages import HumanMessage
    from src.utils.common_utils import encode_image
    """See how qwen local model describes an image."""
    provider = hydra.utils.instantiate(cfg.models.vlm_agent)
    model = provider(
        model=cfg.models.vlm_agent.name, temperature=cfg.models.vlm_agent.temp
    )
    image_data = encode_image(cfg.misc.test_image_path)
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{image_data}",
                },
                {
                    "type": "text",
                    "text": "Describe what you see in the image in great detail",
                },
            ]
        )
    ]
    result = model.invoke(messages)
    log.info(type(result))
    log.info(result.content)


def test_fashion_gen(cfg):
    import h5py
    import random
    from PIL import Image
    """Explore the fashion-gen dataset.

    See how it is setup, how to navigate it, and what content it has.
    """
    with h5py.File(cfg.data.fashion_gen.hdf5_path, "r") as file:
        num_images = file["index"].shape[0]
        idx = random.randint(0, num_images - 1)
        # img = Image.fromarray(file["input_image"][idx].astype("uint8"))
        # img.save(cfg.misc.random_image_save_path)
        price = file["input_msrpUSD"][idx].item()
        log.info(f"Number of images: {num_images}")
        log.info(f"input_msrpUSD {price}")
        for key in cfg.data.fashion_gen.string_attributes:
            value = file[key][idx][0].decode(cfg.data.fashion_gen.string_codec)
            log.info(f"{key} {value}")
        for key in file.keys():
            log.info(f"{key}\t{file[key].dtype}")


class FashionSigLIPEmbedding():
    import torch
    import open_clip
    def __init__(self, cfg):
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model, _, self._preprocess_val = open_clip.create_model_and_transforms(cfg.data.vector_db.embedding_model)
        self._tokenizer = open_clip.get_tokenizer(cfg.data.vector_db.embedding_model)
        self._model.to(self._device)
        self._embed_batch_size = 32

    def get_image_embedding_batch(self, images):
        """images: numpy ndarray of shape [B, H, W, C].

        results is a list of lists where the internal list is of shape 768.
        """
        results = []
        for i in range(0, images.shape[0], self._embed_batch_size):
            batch = images[i : i + self._embed_batch_size]
            tensor_list = [self._preprocess_val(Image.fromarray(img)) for img in images]
            batched_image_tensor = torch.stack(tensor_list, dim=0).to(self._device)
            with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
                image_features = self._model.encode_image(batched_image_tensor, normalize=True).cpu().tolist()
            results.extend(image_features)
        return results

    def get_text_embedding_batch(self, texts):
        """texts is a list of strings.

        results is a list of lists where the internal list is of shape 768.
        """
        results = []
        for i in range(0, len(texts), self._embed_batch_size):
            batch = texts[i : i + self._embed_batch_size]
            batched_texts = self._tokenizer(batch).to(self._device)
            with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
                text_features = self._model.encode_text(batched_texts, normalize=True).cpu().tolist()
            results.extend(text_features)
        return results
    
    def get_paired_embedding_batch(self, images, texts):
        """images is a numpy ndarray of shape [B, H, W, C].

        texts is a list of strings. results is a list of 2-tuples of lists. Each list
        in the tuple is of len 768.
        """
        img_results = self.get_image_embedding_batch(images)
        text_results = self.get_text_embedding_batch(texts)
        return list(zip(img_results, text_results))


def test_qdrant(cfg):
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import (
        Distance,
        PointStruct,
        VectorParams,
    )
    import numpy as np
    import h5py
    import random
    """Explore how I can insert the data from the HDF5 file into Qdrant Vector DB.

    First we load the embedding model from hugging face transformers (the demo
    code for this was in the page for the embedding model).
    """

    embedding_model = cfg.data.vector_db.embedding_model
    client = QdrantClient(path=cfg.data.vector_db.vector_store_path)

    collection_name = cfg.data.vector_db.collection_name
    image_vectors_name = cfg.data.vector_db.image_vectors_name
    text_vectors_name = cfg.data.vector_db.text_vectors_name
    embedding_size = cfg.data.data_processing.embedding_size
    recreate = cfg.data.vector_db.recreate

    if recreate and client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        log.info(f"Deleted existing collection '{collection_name}'.")

    if not client.collection_exists(collection_name):
        log.info(f"Creating collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                image_vectors_name: VectorParams(size=embedding_size, distance=Distance.DOT),
                text_vectors_name: VectorParams(size=embedding_size, distance=Distance.COSINE),
            }
        )
    else:
        log.info(
            f"Collection '{collection_name}' already exists. Appending documents..."
        )

    vec_decode = np.vectorize(pyfunc=lambda x: x.decode('latin-1'))
    with h5py.File(cfg.data.fashion_gen.hdf5_path, "r") as file:
        img = file["input_image"][100: 200].astype("uint8")
        descriptions = vec_decode(np.ravel(file["input_description"][100: 200])).tolist()
    embedder = FashionSigLIPEmbedding(cfg)
    points = []
    for idx, (img_vec, text_vec) in enumerate(embedder.get_paired_embedding_batch(img, descriptions)):
        struct = PointStruct(id=idx, vector={image_vectors_name: img_vec, text_vectors_name: text_vec}, payload={"description": descriptions[idx]})
        points.append(struct)
    client.upsert(collection_name=collection_name, points=points)
    query = [random.random() for i in range(768)]
    # client.query_points returns a qdrant_client.http.models.models.QueryResponse
    # it contains a attribute called points which is a list of ScoredPoints
    # ScoredPoints has attributes id, version, score, payload (dict), vector, shard_key, order_value
    query_response = client.query_points(collection_name=collection_name, query=query, using=image_vectors_name, limit=2)
    log.info("Matched values:")
    for scored_points in query_response.points:
        log.info(scored_points.payload['description'])
