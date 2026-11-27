# FRAG: The Fashion Recommendation Agent
A multi-modal, multi-agent RAG framework for getting clothing item recommendations from a product catalogue.

# Demo
https://github.com/user-attachments/assets/cc40f339-4bd2-4075-aea9-9388405977f5

# Features
1. A stateful multi-agent pipeline using LangGraph with conditional branching and specialised nodes for intent parsing, visual feature extraction, retrieval, self-correction, and explanation.
2. A **self-correcting RAG** loop inspired by Self-RAG, where a critique agent evaluates retrieval quality and feeds structured correction signals back into the generation stage."
3. Uses a domain-specific multi-modal embedding model (marqo-fashionSigLIP) for cross-modal semantic search, enabling image-to-image and text-to-image retrieval across a 260k-item vector database - which acts as an analogue to the product catalogue.
4. A production-grade **MCP server** (with streamable HTTP transport) exposing a Qdrant vector database as LLM-consumable tools, following the Model Context Protocol standard."
5. Incorporates stateful multi-turn conversation with cross-turn image reference tracking and SQLite-backed checkpoint persistence for session resumability.
6. Async-first agentic backend with real-time streaming updates to a **Chainlit UI**, including per-session token usage monitoring for API rate limit management, along with **LangSmith observability** for enhanced debugging.
7. Evaluation pipeline which uses the **LLM-as-a-judge framework** to evaluate the agent across multiple different query types including "items that go well with X", "same item, different style", "items that look good on me", and "aesthetic from image"; then judges the output of the agent on various criteria like visual grounding, individual item suitability, completeness, and coverage; then assigns a weighted final score and produces the mean and standard deviation of all final scores across the entire evaluation dataset.
8. Current Evaluation Score: **9.37 ± 0.78 / 10.00**

# Agent Orchestration Diagram
<p align="center">
  <img width="155" height="422" alt="fashion_agent_diagram" src="https://github.com/user-attachments/assets/79ef3f6a-9411-4232-b967-7272b5946b9c" />
</p>

# Install and Run Instructions
1. Clone git repo
2. Run `uv sync --all-groups` from root directory (or pick groups: `dev`, `siglip` for the local Marqo-FashionSigLIP embedder - CPU-only torch, `local-llm` for the default local LLM (Ollama, `ollama pull qwen3.5:4b`), `ingest` for the HDF5 ingestion script)
3. Chainlit does not play well with symlinked / hardlinked installs. Reinstall just chainlit as real files (re-run this whenever chainlit gets reinstalled):
    `uv sync --all-groups --reinstall-package chainlit --link-mode copy`
    - If you use `UV_LINK_MODE=symlink`: `uv cache clean` / `uv cache prune` deletes the files the venv links to and silently breaks it. Repair with `uv sync --reinstall --all-groups` followed by the chainlit command above.
    - The AWS CLI is not a project dependency. Install v2 system-wide (e.g. `sudo pacman -S aws-cli-v2` on Arch, or AWS's official installer).
4. Copy `.env.example` to `.env` and fill in what you need. The defaults use the local LLM (Ollama, `qwen3.5:4b`) and the local SigLIP embedder. Start Ollama with the memory settings in `cleanup.md` (Phase 7 notes, `LLAMA_ARG_CACHE_RAM=0`).
5. Install podman / docker and podman-compose / docker-compose, and run `podman-compose up -d` to start Qdrant. It serves `data/qdrant_storage/` (override with `QDRANT_STORAGE_DIR`), which holds both collections: `fashion_gen` (SigLIP, the default) and `nova_fashion_gen` (Nova).
6. Run `podman-compose down` to stop the containers when done.
7. The catalogue images and metadata are read from S3, so AWS credentials must be configured (`aws configure`).
8. Optional: serve the catalogue tools from a separate process with `uv run frag mcp` (port 9000) and set `TOOLS__MODE=mcp`. By default the agent uses the tools in-process.
9. Start the Chainlit UI with `uv run chainlit run app.py` (http://localhost:8000; sign-in is through AWS Cognito), or chat in the terminal with `uv run frag chat`.
10. Evaluation: `uv run frag eval` runs the agent over the 60 evaluation queries, then has the LLM judge the outputs. Each run is a self-contained directory `data/evaluation_runs/<UTC time>/` (`config.json` with the models and setup, `results.jsonl`, `judgement.jsonl`, `summary.json` with completed/total and mean ± std per group, and copies of the images). `--limit N` runs only the first N queries; `--skip-judge` only runs the agent; `--skip-run` judges a stored run (`--run-dir`, default the latest); `--sleep S` waits between queries. The judge uses the agent's LLM unless `EVALUATION__JUDGE_LLM__PROVIDER` (and optionally `..._MODEL_ID`) is set, so the agent can run locally while only the judging uses Bedrock (about $0.30 for 60 queries); running the agent on Bedrock as well costs about $3–5.
11. Data pipeline scripts (run by path, e.g. `uv run python scripts/qdrant_admin.py info`): `scripts/ingest_fashiongen.py` (HDF5 to PNGs + Parquet on S3), `scripts/build_vector_db.py` (Nova embeddings into Qdrant; billed, asks first) and `scripts/qdrant_admin.py` (info, sample, snapshot, restore, copy-subset).
12. To run the test(s), do `uv run pytest` (offline). The slow model-quality tests run the real agent on the local Ollama model, Qdrant and S3: `uv run pytest -m ollama` (see `tests/model/`). For a coverage report: `uv run pytest --cov=frag --cov-report=term-missing:skip-covered --cov-report=html:reports/htmlcov`


# AWS resources
Every AWS resource the app uses (catalogue and chat-elements buckets, the two DynamoDB tables, the Cognito login) is owned by a CloudFormation stack in `infra/cloudformation/` (`frag-<env>-storage`, `frag-<env>-auth`). It's changed only through `infra/scripts/`, which shows a change set and waits for a typed confirmation. See `docs/cloudformation.md` for the workflow and `docs/aws_inventory.md` for what exists and why.
- `infra/scripts/deploy.sh [layer] [Param=Value ...] [--dry-run]`: set up or update the stacks.
- `infra/scripts/write_env_from_stack.sh [--write]`: put table, bucket and Cognito values (including the client secret) into `.env`.
- `infra/scripts/create_user.sh <email>`: sign-up is admin-only, so users are created this way.
- `infra/scripts/teardown.sh <layer>`: delete a layer; it refuses stacks that hold retained data.

**Data uploads** go only through `scripts/ingest_fashiongen.py` (catalogue images and metadata Parquet) and, once the deployment plan adds it, the Qdrant snapshot upload. Nothing else writes catalogue data to S3.

# AWS Deployment Considerations
1. KNOWN: Embedding model - lives on the EC2 t3.large - can use Amazon Nova Embedding - but quality might be lower
2. KNOWN: MCP server - AWS Lambda + API Gateway - each tool is a lambda function - use API Gateway to be called by agent - The core flow would be: API Gateway HTTP API → Lambda (FastMCP + Mangum) → External services. Might not actually need API Gateway since you are invoking the lambda internally from EC2
3. Agent Orchestration - this runs on the EC2 t3.large
4. Chainlit UI - this runs on the EC2 t3.large - points 3. 4. are being suggested to run together EC2 t3.small - which will cost around 15 / month - for always on. But perhaps you can have it on demand in some way - if you are willing to wait for the launch time. - also need to see how to setup static IP - that might cost extra
6. KNOWN: Qdrant Docker Container - Can run on ECS Fargate 0.25vCPU and 1 GB RAM (or EC2 t4g.micro) - it will have 90 second startup time and 1-2 second query fetch time
7. KNOWN: Qdrant Vector DB - this can live on an EBS, although EFS is better (more support with Fargate) but more expensive
8. KNOWN: Fashion-Gen 14 GB database - use a single file (hdf5) entry on AWS S3 Standard - this supports random access with an object - see exploration folder
9. KNOWN: Sqlite checkpointer db - use dynamo DB
10. KNOWN: LLM Calls - AWS Bedrock - this is a very in-demand service - might have to use Mistral Medium directly for now since Bedrock does not have this model and others cost money - there is a simple ChatBedrockConverse API on LangChain which allows easy switching from Mistral to any Bedrock model
11. .env - AWS Secrets Manager / Parameter Store

Can use Qdrant Cloud Free Tier to host the docker container and the Qdrant DB - they have 0.5vCPU, 1 GB RAM, 4 GB disk space free forever

## Changes to MCP Server
1. Need to switch out code components that assume local stuff
```python
# Replace FashionSigLIPEmbedding with:
class RemoteEmbedder:
    def get_text_embedding_batch(self, texts):
        response = httpx.post("https://your-embedding-api/embed", json={"texts": texts})
        return response.json()["embeddings"]

# Replace mcp.run() at the bottom with:
from mangum import Mangum
app = mcp.http_app()   # FastMCP exposes the underlying Starlette app
handler = Mangum(app)  # this is your Lambda handler
```


TODO: you have added psycopg[binary] as a dependency - but the binary is not recommended for production - please libpq on the system in a production environment.


# Contribution Guidelines
I have connected JIRA to github so please make use of the integration
1. Pick up a JIRA ticket (story, task, sub-task) or create it if it does not exist (likely at the start) - create a branch named `<prefix>/<project-name>-<ticket-id>-short-description` (eg: `bugfix/FRAG-23-fix-graph-generation-bug`) - our project name is `FRAG`. If you don't use the right branch name, you will get a message due to a Github Action
2. JIRA also allows you to create a github branch directly from JIRA - it also auto-suggests a name based on the ticket name.
2. Every commit to that branch should start with `<project-name>-<ticket-id>` in the message (eg: `FRAG-42 Add google OAuth callback`)
3. Use Smart Commit Syntax to automatically update JIRA through the commit.
4. Open a PR with title `<project-name>-<ticket-id>: <description>` when ready for review. For reviews that only impact the `exploration` folder, I will simply approve. Any other project code should be reviewed seriously. In the start we might have lower scrutiny, while the project skeleton takes shape.
5. Once the PR is merged, JIRA issue gets closed and the full history is tracked there.
6. Ensure the email you use to commit to github is the one of the emails that is registered in JIRA for the project, else it will be silently ignored.
7. Please mostly create branches from develop (at least for now) rather than main - main is for releases

### Gitflow prefixes
These are the prefixes which are allowed in git or github branches
| Prefix   | Purpose                                                          | Branches from | Merges into               |
| -------- | ---------------------------------------------------------------- | ------------- | ------------------------- |
| feature/ | Developing new features or functionalities                       | develop       | develop            |
| bugfix/  | Fixing bugs found during development (non-production)            | develop       | develop |
| release/ | Preparing a new version for production (version bumps, final QA) | develop       | main + develop |
| hotfix/  | Urgent, critical patches applied directly to production          | main          | main + develop |
| support/ | Long-term support branches for maintaining older versions        | main          | Stays isolated 
| refactor/ | Code improvements, restructuring, cleanup - no external change  | develop       | develop |
| chore/   | Maintenance tasks eg updating dependencies, configs, or tooling  | develop       | develop |
| docs/    | Documentation-only changes (READMEs, wikis, inline comments)     | develop       | develop |
| test/    | Adding / updating tests without changing production code         | develop       | develop |
| ci/      | Changes to CICD pipeline  Github Actions, Jenkins etc            | develop       | develop |

### Smart Commit Syntax
1. Syntax (for commits): `<ISSUE_KEY> <ignored_text> #<command 1> <command 1 arguments> #<command 2> <command 2 arguments> ...`
2. Three key commands:
    1. `#comment <comment text>` - add comment to jira issue
    2. `#time <value> <optional worklog comment>` - add time you worked on the task (`w` weeks, `d` days, `h` hours, `m` minutes)
    3. `#<transition-name>` - to move the ticket to a different flow state (eg: `#to-do`, `#in-progress`, `#done`, `#cancelled` - these are the only valid ones)

