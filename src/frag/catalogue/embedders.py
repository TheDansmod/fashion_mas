"""Query-text embedders; each must match the vectors of its collection.

Only the text path exists: the catalogue vectors were embedded once, offline
(`scripts/build_vector_db.py`).
"""

import asyncio
import json
from typing import Protocol

import boto3
from botocore.config import Config
from loguru import logger as log

from frag.settings import Settings


class Embedder(Protocol):
    """Turns a query text into a vector of length `dim`."""

    dim: int

    async def embed_text(self, text: str) -> list[float]:
        """Embed one text."""
        ...


class NovaEmbedder:
    """Amazon Nova Multimodal Embeddings on Bedrock (the `nova_fashion_gen` collection).

    Every `embed_text` call is a billed Bedrock request.
    """

    def __init__(self, model_id: str, dim: int, purpose: str, region: str) -> None:
        """Create the Bedrock runtime client (no requests are made here)."""
        self.dim = dim
        self._model_id = model_id
        self._purpose = purpose
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(retries={"mode": "adaptive", "total_max_attempts": 5}),
        )

    def _embed(self, text: str) -> list[float]:
        body = {
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": self._purpose,
                "embeddingDimension": self.dim,
                "text": {"value": text, "truncationMode": "END"},
            },
        }
        response = self._client.invoke_model(
            body=json.dumps(body),
            modelId=self._model_id,
            accept="application/json",
            contentType="application/json",
        )
        return json.loads(response["body"].read())["embeddings"][0]["embedding"]

    async def embed_text(self, text: str) -> list[float]:
        """Embed `text` with one Bedrock `InvokeModel` call."""
        return await asyncio.to_thread(self._embed, text)


class SigLIPEmbedder:
    """Marqo-FashionSigLIP text tower, run locally (the `fashion_gen` collection).

    torch and open_clip are imported here rather than at module level so that
    deployments using Nova never need them (they are the optional `siglip` group).
    """

    def __init__(self, model_name: str) -> None:
        """Load the model once per process; this takes a few seconds."""
        import open_clip  # noqa: PLC0415  optional dependency, see class docstring
        import torch  # noqa: PLC0415  optional dependency, see class docstring

        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("loading SigLIP model {} on {}", model_name, self._device)
        model, _, _ = open_clip.create_model_and_transforms(model_name)
        self._model = model.to(self._device).eval()
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self.dim = len(self._embed("warm-up"))
        log.info("SigLIP model ready ({}-d)", self.dim)

    def _embed(self, text: str) -> list[float]:
        tokens = self._tokenizer([text]).to(self._device)
        with self._torch.no_grad():
            features = self._model.encode_text(tokens, normalize=True)
        return features[0].cpu().tolist()

    async def embed_text(self, text: str) -> list[float]:
        """Embed `text` in a worker thread (CPU-bound)."""
        return await asyncio.to_thread(self._embed, text)


def build_embedder(settings: Settings) -> Embedder:
    """Build the embedder for `settings.embedding.backend` (may load a model)."""
    emb = settings.embedding
    if emb.backend == "nova":
        return NovaEmbedder(
            model_id=emb.nova_model_id,
            dim=emb.nova_dim,
            purpose=emb.nova_purpose,
            region=settings.aws_region,
        )
    return SigLIPEmbedder(emb.siglip_model)
