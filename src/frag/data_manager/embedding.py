import json

import boto3
from loguru import logger as log
from botocore.config import Config

class AWSEmbedder:
    def __init__(self, embedding_size, model_id):
        config = Config(retries={"total_max_attempts": 5, "mode": "adaptive"})
        self.client = boto3.client("bedrock-runtime", config=config)
        self.embedding_size = embedding_size
        self.model_id = model_id

    def _get_embedding(self, request_body):
        try:
            response = self.client.invoke_model(
                body=json.dumps(request_body),
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json",
            )
        except Exception:
            log.exception(
                "There was an exception while getting embedding from the nova model."
            )
            raise

        resp_body = json.loads(response["body"].read())
        embedding = resp_body["embeddings"][0]["embedding"]
        return embedding

    def get_text_embedding_batch(self, texts):
        results = []
        for text in texts:
            request_body = {
                "taskType": "SINGLE_EMBEDDING",
                "singleEmbeddingParams": {
                    "embeddingPurpose": "GENERIC_INDEX",
                    "embeddingDimension": self.embedding_size,
                    "text": {"value": text, "truncationMode": "END"},
                },
            }
            results.append(self._get_embedding(request_body))
        return results
