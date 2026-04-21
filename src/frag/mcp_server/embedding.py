
import torch
import open_clip
from dependency_injector.wiring import inject, Provide as PV

from frag.config.container import Container

cfg = Container.config.provided

class FashionSigLIPEmbedding:
    """Create and return multi-modal embeddings."""

    @inject
    def __init__(
        self,
        embedding_model: str = PV[cfg.data.vector_db.embedding_model],
        embedding_batch_size: str = PV[cfg.data.data_processing.embedding_batch_size],
    ):
        """Initialise device, model, image and text processor, and batch size.

        None of the attributes are intended for external use. So all of them start
        with _.
        """
        self._device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(self._device_type)
        self._model, _, self._preprocess_val = open_clip.create_model_and_transforms(
            embedding_model
        )
        self._tokenizer = open_clip.get_tokenizer(embedding_model)
        self._model.to(self._device)
        self._embed_batch_size = embedding_batch_size

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
