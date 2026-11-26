"""Query-text embedding with Amazon Nova Multimodal Embeddings (Bedrock)."""

import json

import boto3
from botocore.config import Config
from loguru import logger as log


class AWSEmbedder:
    """Embeds text with Nova; the vectors match the `nova_fashion_gen` collection."""

    def __init__(
        self,
        embedding_size: int,
        model_id: str,
        region_name: str,
        purpose: str = "GENERIC_INDEX",
    ) -> None:
        """Create the Bedrock runtime client (no calls are made here)."""
        config = Config(retries={"total_max_attempts": 5, "mode": "adaptive"})
        self.client = boto3.client(
            "bedrock-runtime", region_name=region_name, config=config
        )
        self.embedding_size = embedding_size
        self.model_id = model_id
        self.purpose = purpose

    def _get_embedding(self, request_body: dict) -> list[float]:
        try:
            response = self.client.invoke_model(
                body=json.dumps(request_body),
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json",
            )
        except Exception:
            log.exception("Failed to get an embedding from the Nova model.")
            raise
        resp_body = json.loads(response["body"].read())
        return resp_body["embeddings"][0]["embedding"]

    def get_text_embedding_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed each text (one Bedrock call per text)."""
        return [
            self._get_embedding(
                {
                    "taskType": "SINGLE_EMBEDDING",
                    "singleEmbeddingParams": {
                        "embeddingPurpose": self.purpose,
                        "embeddingDimension": self.embedding_size,
                        "text": {"value": text, "truncationMode": "END"},
                    },
                }
            )
            for text in texts
        ]
