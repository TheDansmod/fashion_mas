"""Application settings: one pure, idempotent Pydantic settings tree.

Rules (see cleanup.md §3.1):
- Constructing `Settings` only reads environment variables and `.env`; no other I/O and
  no side effects, so `Settings() == Settings()` for the same environment.
- Nothing here is instantiated at import time. Use `get_settings()` (one cached instance
  per process) or construct `Settings(...)` directly in tests.
- Paths are plain strings; the code that uses a path checks or creates it.
- Only settings the application reads itself are declared. Variables read directly by
  libraries (LangSmith, Chainlit/Cognito OAuth, Ollama, FastMCP, Hugging Face, boto3
  credentials) are deliberately absent.

Nested sections are set from the environment with a double underscore, e.g.
`CHECKPOINTER__BACKEND=dynamodb` or `LLM__PROVIDER=ollama`.
"""

from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class _Section(BaseModel):
    # extra="forbid" so that a mistyped nested env var (e.g. CHECKPOINTER__BACKND)
    # fails loudly instead of being silently ignored
    model_config = ConfigDict(frozen=True, extra="forbid")


# Per-provider defaults for the fields left unset. `ollama` (local and free) is the
# default provider; set LLM__PROVIDER=bedrock for Claude Haiku on AWS.
_LLM_PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "ollama": {
        # tested end to end on an 8 GB GPU (cleanup.md, Phase 7 notes); pull it with
        # `ollama pull qwen3.5:4b`
        "model_id": "qwen3.5:4b",
        # a local server needs no client-side throttling
        "rate_limit_rps": None,
    },
    "bedrock": {
        "model_id": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        # kept below the account quota; token-per-minute limits are not modelled
        "rate_limit_rps": 0.8,
    },
}


class LLMSettings(_Section):
    """The chat / vision LLM used by every agent node and by the evaluation judge.

    `model_id` and `rate_limit_rps` default per provider (`_LLM_PROVIDER_DEFAULTS`),
    so switching `provider` alone gives a working configuration.
    """

    provider: Literal["bedrock", "ollama"] = "ollama"
    model_id: str
    temperature: float = 0.6
    # client-side request rate limit (LangChain InMemoryRateLimiter: a token bucket);
    # None disables it
    rate_limit_rps: float | None
    # ollama only: turn a thinking model's reasoning on or off (None = the model's
    # default). Off by default: small local models give empty or invalid structured
    # output with it on, and take much longer
    reasoning: bool | None = False
    rate_limit_max_bucket: float = 1.0
    rate_limit_check_every_s: float = 1.0

    @model_validator(mode="before")
    @classmethod
    def _provider_defaults(cls, data: Any) -> Any:  # noqa: ANN401  pydantic passes raw input
        if not isinstance(data, dict):
            return data
        provider = data.get("provider", cls.model_fields["provider"].default)
        defaults = _LLM_PROVIDER_DEFAULTS.get(provider, {})
        return {**defaults, **data}


class EmbeddingSettings(_Section):
    """Query-time text embedder; must match the vectors in the selected collection."""

    # "siglip": Marqo-FashionSigLIP run locally (needs the `siglip` dependency group);
    # "nova": Amazon Nova on Bedrock (billed per query)
    backend: Literal["siglip", "nova"] = "siglip"
    siglip_model: str = "hf-hub:Marqo/marqo-fashionSigLIP"
    nova_model_id: str = "amazon.nova-2-multimodal-embeddings-v1:0"
    # Nova supports 3072, 1024, 384 and 256; the existing collection uses 1024
    nova_dim: int = 1024
    nova_purpose: str = "GENERIC_INDEX"


class CatalogueSettings(_Section):
    """Where the product catalogue (images + metadata) lives on S3, and its fields."""

    s3_bucket: str = "frag-fashion-gen-dataset"
    metadata_key: str = "metadata/fashion_gen_metadata.parquet"
    # images are sharded into folders of 1000 by index
    image_key_template: str = "images/{shard:03d}/{idx}.png"
    # index_2 is aligned with the sequential dataset index (unlike `index`)
    index_field: str = "index_2"
    category_field: str = "input_category"
    description_field: str = "input_description"
    price_field: str = "input_msrpUSD"


class VectorDBSettings(_Section):
    """Qdrant connection and collection selection."""

    # str rather than HttpUrl: the Qdrant client expects a plain string
    url: str = "http://localhost:6333"
    # gRPC is faster; REST is more convenient for debugging
    prefer_grpc: bool = True
    image_vector_name: str = "images"
    # one collection per embedding backend - vectors are not interchangeable
    collections: dict[str, str] = Field(
        default_factory=lambda: {"siglip": "fashion_gen", "nova": "nova_fashion_gen"}
    )


class ToolsSettings(_Section):
    """How the agent reaches the catalogue tools."""

    # "local": in-process tools; "mcp": tools served by the MCP server process
    mode: Literal["local", "mcp"] = "local"
    mcp_url: str = "http://localhost:9000/mcp"
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 9000


class CheckpointerSettings(_Section):
    """LangGraph checkpointer; switch backends with CHECKPOINTER__BACKEND only."""

    backend: Literal["memory", "sqlite", "postgres", "dynamodb"] = "sqlite"
    sqlite_path: str = "data/pipeline_checkpoints.db"
    # matches the `postgres` compose profile; override with CHECKPOINTER__POSTGRES_DSN
    postgres_dsn: str = "postgresql://frag:frag@localhost:5432/frag"
    postgres_pool_size: int = 10
    # must match the table name used when deploying the CloudFormation stack
    dynamodb_table: str = "langgraph-checkpoints"
    # dynamodb only: entries expire after a day
    ttl_seconds: int = 60 * 60 * 24
    compress: bool = True


class ChatHistorySettings(_Section):
    """Chainlit chat-history persistence (DynamoDB table + S3 bucket for elements)."""

    # required by the Chainlit app (checked there); optional so that other entrypoints
    # and tests can build Settings without them
    dynamodb_table: str | None = None
    s3_bucket: str | None = None


class AgentSettings(_Section):
    """Behaviour of the LangGraph agent."""

    # used when the user does not say how many recommendations they want
    default_num_recommendations: int = 3
    # retries in the modifier node to get the right number of item descriptions
    num_recommendation_attempts: int = 3
    # how many times the critique node may send the recommendations back
    max_num_critiques: int = 3
    # recommended catalogue images are written here (unique file names)
    temp_dir: str = "data/temp_images"
    # if set, the compiled graph is drawn here (uses the mermaid.ink web API)
    draw_graph_path: str | None = None


class EvalWeights(_Section):
    """Relative weights of the judge criteria; they need not sum to 1."""

    visual_grounding: float = 0.2
    item_suitability: float = 0.3
    completeness_coverage: float = 0.5


class EvalSettings(_Section):
    """LLM-as-a-judge evaluation inputs and outputs."""

    queries_path: str = "data/evaluation_data/eval_queries.json"
    images_dir: str = "data/evaluation_data/eval_images"
    # each evaluation writes a self-contained <runs_dir>/<UTC time>/ directory
    runs_dir: str = "data/evaluation_runs"
    weights: EvalWeights = Field(default_factory=EvalWeights)


class LogSettings(_Section):
    """Loguru logging; file names get a per-process service suffix."""

    # None -> DEBUG in development/staging, INFO in production
    level: str | None = None
    console: bool = True
    human_readable_file: bool = True
    machine_readable_file: bool = True
    file_dir: str = "logs"
    # loguru rotation/retention/compression syntax
    rotation: str = "1 day"
    retention: str = "30 days"
    compression: str = "zip"
    library_levels: dict[str, str] = Field(
        default_factory=lambda: {
            "httpx": "WARNING",
            "urllib3": "WARNING",
            "mcp": "WARNING",
            "httpcore": "WARNING",
            "botocore": "WARNING",
            "langgraph_checkpoint_aws": "WARNING",
        }
    )


class Settings(BaseSettings):
    """Root settings object."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        # .env is shared with libraries that read their own variables
        extra="ignore",
        frozen=True,
    )

    app_env: Literal["development", "staging", "production"] = "development"
    aws_region: str = "us-east-1"

    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    catalogue: CatalogueSettings = Field(default_factory=CatalogueSettings)
    vector_db: VectorDBSettings = Field(default_factory=VectorDBSettings)
    tools: ToolsSettings = Field(default_factory=ToolsSettings)
    checkpointer: CheckpointerSettings = Field(default_factory=CheckpointerSettings)
    chat_history: ChatHistorySettings = Field(default_factory=ChatHistorySettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    evaluation: EvalSettings = Field(default_factory=EvalSettings)
    logging: LogSettings = Field(default_factory=LogSettings)

    # local token-usage tracking; None disables it
    token_usage_csv: str | None = "data/token_usage.csv"  # noqa: S105  a file path, not a secret

    @property
    def collection(self) -> str:
        """Qdrant collection matching the configured embedding backend."""
        return self.vector_db.collections[self.embedding.backend]

    @property
    def is_dev(self) -> bool:
        """Whether this is a development environment."""
        return self.app_env == "development"

    @property
    def log_level(self) -> str:
        """Effective log level."""
        if self.logging.level:
            return self.logging.level
        return "INFO" if self.app_env == "production" else "DEBUG"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings (read once, then cached)."""
    return Settings()
