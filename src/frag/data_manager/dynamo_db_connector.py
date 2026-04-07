import boto3
from botocore.config import Config

def get_dynamodb_resource(cfg) -> boto3.resources.base.ServiceResource:
    """
    Returns a DynamoDB resource.

    - In development: connects to DynamoDB Local via DYNAMODB_ENDPOINT_URL.
    - In production:  omit DYNAMODB_ENDPOINT_URL entirely; boto3 routes to AWS.
    
    Credentials are sourced from environment variables, which boto3 picks
    up automatically (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION).
    IAM roles (on EC2/ECS/Lambda) are also resolved automatically in production.
    """
    config = Config(
        retries={"max_attempts": 3, "mode": "adaptive"},
        connect_timeout=5,
        read_timeout=10,
    )
    if cfg.app_env == cfg.dev_env:
        kwargs = dict(config=config)
        kwargs["endpoint_url"] = cfg.data.dynamo_db.endpoint_url   # None in production
        kwargs["aws_access_key_id"] = "local"
        kwargs["aws_secret_access_key"] = "local"
    return boto3.resource("dynamodb", **kwargs)
