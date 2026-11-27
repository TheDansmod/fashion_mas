# FRAG: The Fashion Recommendation Agent
A multi-modal agentic RAG application that recommends clothing items from a product catalogue, from a text request, a photo, or both.

# Demo
https://github.com/user-attachments/assets/cc40f339-4bd2-4075-aea9-9388405977f5

# Features
1. **A stateful LangGraph agent** with conditional routing between specialised nodes: intake, quantity and item-type extraction, intent parsing, visual feature extraction from uploaded images, retrieval query generation, recommendation, critique and explanation. Each user message is one graph run; the conversation state carries over between turns.
2. **A self-correcting RAG loop**, inspired by Self-RAG: a critique node checks the recommendations against the request and sends structured feedback back to the retrieval step, up to a configurable number of times. Item IDs the LLM returns are validated against what the tools actually returned.
3. **Two embedding backends over the same 260k-item Fashion-Gen catalogue:**
   - `siglip` (local default): Marqo-FashionSigLIP embeds queries locally and searches the `fashion_gen` Qdrant collection (260k items).
   - `nova` (AWS): Amazon Nova Multimodal Embeddings on Bedrock, against the `nova_fashion_gen` collection (~10k items).
   Images and metadata (Parquet) are always read from S3.
4. **Two LLM providers**, switched with one setting: Claude Haiku 4.5 on **AWS Bedrock**, or a local vision model on **Ollama** (`qwen3.5:4b` tested) for free development.
5. **An MCP server** (FastMCP, streamable HTTP) that exposes the catalogue as tools: semantic search, product categories, item lookup. The agent can use the same tools in-process instead.
6. **Multi-turn conversations** that track images across turns, with **four interchangeable LangGraph checkpointers** (memory, SQLite, Postgres, DynamoDB) selected by one setting.
7. **A Chainlit UI** with streamed progress, Cognito sign-in (admin-created users only), persistent chat history (DynamoDB + S3), token usage recorded per message (`data/token_usage.csv`), and optional **LangSmith** tracing.
8. **LLM-as-a-judge evaluation** over 60 queries in four groups ("items that go well with X", "same item, different style", "items that look good on me", "aesthetic from image"). The judge scores visual grounding, item suitability and completeness. Each run records the models used, the completion rate (`completed/total`), and the mean ± std of the weighted score per group.
9. **Current evaluation score: pending re-run.** The earlier figure (9.37 ± 0.78) was the average of only the 31 of 60 queries that completed, so it isn't reported here. The completion rate will be reported together with the score.
10. **Infrastructure as code:** every AWS resource is owned by a CloudFormation stack and changed only through reviewed change sets (`infra/`).

# Agent Orchestration Diagram
<p align="center">
  <img width="155" height="422" alt="fashion_agent_diagram" src="https://github.com/user-attachments/assets/79ef3f6a-9411-4232-b967-7272b5946b9c" />
</p>

# Install and Run Instructions

### 1. Install
1. Clone the repo, then from the root directory run:
   `uv sync --group siglip --group local-llm`
   Add `--group dev` for tests and lint, or `--group ingest` for the HDF5 ingestion script; `--all-groups` installs everything. The `siglip` group (CPU-only torch) is only needed for the local SigLIP embedder, and the `local-llm` group only for Ollama.
2. Chainlit doesn't work from a symlinked or hard-linked install. Reinstall just Chainlit as real files, and re-run this whenever Chainlit gets reinstalled:
   `uv sync --all-groups --reinstall-package chainlit --link-mode copy`
   - If your shell sets `UV_LINK_MODE=symlink`: `uv cache clean` or `uv cache prune` deletes the files the venv links to, which silently breaks it. Repair it with `uv sync --reinstall --all-groups`, followed by the Chainlit command above.
3. Install the AWS CLI v2 system-wide (it isn't a project dependency; e.g. `sudo pacman -S aws-cli-v2`, or AWS's installer), and configure credentials with a named profile or `aws configure sso`. The catalogue is read from S3 in every mode.
4. Install podman or docker with compose.

### 2. Configure
1. Copy `.env.example` to `.env` and pick a preset: **local development** (Ollama + SigLIP + SQLite, the default) or **AWS** (Bedrock + Nova + DynamoDB).
2. Fill in the AWS resource names and Cognito values from the CloudFormation stacks:
   `infra/scripts/write_env_from_stack.sh --write`
   This also writes the Cognito client secret, which it never prints. If the stacks don't exist yet, see [AWS resources](#aws-resources).
3. Create the Chainlit auth secret with `uv run chainlit create-secret` and put it in `.env` as `CHAINLIT_AUTH_SECRET`.
4. Create a login for yourself: `infra/scripts/create_user.sh <email>`. Sign-up is admin-only.

### 3. Start the services
1. Qdrant: `podman compose up -d`. It serves `data/qdrant_storage/` (override with `QDRANT_STORAGE_DIR`), which holds both collections: `fashion_gen` (SigLIP) and `nova_fashion_gen` (Nova). To create the storage folder: `mkdir -p data/qdrant_storage/`. Restore a collection with `scripts/qdrant_admin.py restore`.
2. Optional Postgres checkpointer: `podman compose --profile postgres up -d`, and set `CHECKPOINTER__BACKEND=postgres`.
3. Local LLM: `ollama pull qwen3.5:4b`, then start Ollama with its prompt cache in system RAM turned off. By default it can grow to 8 GiB, and with image-heavy agent calls it once filled 14 GB of RAM and froze the machine. The tested command, with a memory cap as a safety net, is:
   ```bash
   LLAMA_ARG_CACHE_RAM=0 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_CONTEXT_LENGTH=16384 \
     systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 ollama serve
   ```
   On an 8 GB GPU, `qwen3.5:4b` runs fully on the GPU, using about 3.7 GB of VRAM at a 16k context. Keep `LLM__REASONING=false`, the default: with thinking on, small models return empty or invalid structured output.
4. Stop the containers with `podman compose down`.

### 4. Run
- **Chainlit UI:** `uv run chainlit run app.py`, then open http://localhost:8000 and sign in through Cognito.
- **Terminal chat:** `uv run frag chat`.
- **MCP server (optional):** `uv run frag mcp` (port 9000, health check at `/health`). Set `TOOLS__MODE=mcp` so the agent uses it; by default the tools run in-process.
- **Evaluation:** `uv run frag eval` runs the agent over the 60 evaluation queries, then has the LLM judge the outputs.
  - Each run is a self-contained directory, `data/evaluation_runs/<UTC time>/`, containing:
    - `config.json`: the models and setup
    - `results.jsonl` and `judgement.jsonl`
    - `summary.json`: completed/total and mean ± std per group
    - copies of the images
  - Options:
    - `--limit N` runs only the first N queries.
    - `--skip-judge` runs only the agent.
    - `--skip-run` judges a stored run (`--run-dir`; the default is the latest).
    - `--sleep S` waits S seconds between queries.
  - The judge uses the agent's LLM unless `EVALUATION__JUDGE_LLM__PROVIDER` (and optionally `..._MODEL_ID`) is set. That way the agent can run locally while only the judging uses Bedrock, which costs about $0.30 for 60 queries. Running the agent on Bedrock as well costs about $3–5.
- **Tests:**
  - `uv run pytest` runs offline.
  - `uv run pytest -m ollama` runs the slow model-quality tests: the real agent on the local Ollama model, Qdrant and S3 (see `tests/model/`).
  - For a coverage report: `uv run pytest --cov=frag --cov-report=term-missing:skip-covered --cov-report=html:reports/htmlcov`.
- **Lint:** `uv run ruff check` and `uv run cfn-lint infra/cloudformation/*.yaml`.
- **Data pipeline scripts** (run by path, e.g. `uv run python scripts/qdrant_admin.py info`):
  - `scripts/ingest_fashiongen.py`: HDF5 to PNGs + Parquet on S3.
  - `scripts/build_vector_db.py`: Nova embeddings into Qdrant; billed, and asks first. It's kept as a record; the collections already exist.
  - `scripts/qdrant_admin.py`: `info`, `sample`, `snapshot`, `restore`, `copy-subset`.

# AWS resources
Every AWS resource the app uses is owned by a CloudFormation stack in `infra/cloudformation/` (`frag-<env>-storage`, `frag-<env>-auth`): the catalogue and chat-elements buckets, the two DynamoDB tables, and the Cognito login. These resources are changed only through `infra/scripts/`, which shows a change set and waits for a typed confirmation. See `docs/cloudformation.md` for the workflow and `docs/aws_inventory.md` for what exists and why.
- `infra/scripts/deploy.sh [layer] [Param=Value ...] [--dry-run]`: set up or update the stacks.
- `infra/scripts/write_env_from_stack.sh [--write]`: put table, bucket and Cognito values (including the client secret) into `.env`.
- `infra/scripts/create_user.sh <email>`: create a user (sign-up is admin-only).
- `infra/scripts/teardown.sh <layer>`: delete a layer; it refuses stacks that hold retained data.

**Data uploads** go only through `scripts/ingest_fashiongen.py` (catalogue images and metadata Parquet); nothing else writes catalogue data to S3.

# Contribution Guidelines
I have connected JIRA to GitHub, so please make use of the integration.
1. Pick up a JIRA ticket (story, task, sub-task), or create one if it doesn't exist. Create a branch named `<prefix>/<project-name>-<ticket-id>-short-description` (e.g. `bugfix/FRAG-23-fix-graph-generation-bug`); our project name is `FRAG`. A GitHub Action flags branches that don't follow this pattern.
2. JIRA can also create the GitHub branch for you, and suggests a name based on the ticket name.
3. Start every commit message on that branch with `<project-name>-<ticket-id>` (e.g. `FRAG-42 Add google OAuth callback`).
4. Use Smart Commit Syntax to update JIRA automatically through the commit.
5. When ready for review, open a PR titled `<project-name>-<ticket-id>: <description>`.
6. Once the PR is merged, the JIRA issue is closed and the full history is tracked there.
7. Commit with an email address that is registered in JIRA for the project; otherwise the commits are silently ignored.
8. Create branches from `main`. PRs merge into `main`.

### Branch prefixes
These are the prefixes allowed in branch names. All branches start from `main` and merge back into `main`.
| Prefix    | Purpose                                                          |
| --------- | ---------------------------------------------------------------- |
| feature/  | Developing new features or functionalities                       |
| bugfix/   | Fixing bugs found during development (non-production)            |
| release/  | Preparing a new version for production (version bumps, final QA) |
| hotfix/   | Urgent, critical patches applied directly to production          |
| support/  | Long-term support branches for maintaining older versions (these stay isolated) |
| refactor/ | Code improvements, restructuring, cleanup; no external change    |
| chore/    | Maintenance tasks, e.g. updating dependencies, configs or tooling |
| docs/     | Documentation-only changes (READMEs, wikis, inline comments)     |
| test/     | Adding or updating tests without changing production code        |
| ci/       | Changes to the CI/CD pipeline (GitHub Actions, Jenkins, etc.)    |

### Smart Commit Syntax
1. Syntax (for commits): `<ISSUE_KEY> <ignored_text> #<command 1> <command 1 arguments> #<command 2> <command 2 arguments> ...`
2. Three key commands:
    1. `#comment <comment text>`: add a comment to the JIRA issue.
    2. `#time <value> <optional worklog comment>`: log time worked on the task (`w` weeks, `d` days, `h` hours, `m` minutes).
    3. `#<transition-name>`: move the ticket to a different state. The only valid ones are `#to-do`, `#in-progress`, `#done` and `#cancelled`.
