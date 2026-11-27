# FRAG Cleanup and Refactor Instructions

These instructions are for an agent doing the cleanup. They come from a full read of the
repository on 2026-11-26 and were revised the same day with the user's decisions (§1).
Nothing was executed and no AWS calls were made while writing this.

`docs/frag_aws_completion_plan.md` (referred to below as "the plan"; currently at
`data/frag_aws_completion_plan.md`) covers the deployment work. This document gets the
codebase small, correct and explicit enough that the plan can be carried out on top of
it. Where the two overlap, this document does the code-level part only. Where the user's
decisions here contradict the plan (for example the plan's "delete the full-size
buckets"), **this document wins**, and Phase 15 updates the plan to match.

---

## 0. Rules for the executing agent

### 0.1 Git

No commits, no pushes, no branch changes, no GitHub actions. The user commits.

### 0.2 AWS usage policy (read carefully)

The account is personal. The goal is **zero or near-zero spend**. The account has some
sign-up credits, but **credits don't count as "free"**: the baseline is the default AWS
free tier, and anything that would go beyond it needs the user's approval first, even if
credits would cover it.
Region: `us-east-1`. Use the credentials already configured on the machine (default
profile or `AWS_PROFILE`). Never create, print or copy credentials.

**A. Allowed without asking: read-only control-plane calls.** These are free and
change nothing:
- `sts get-caller-identity`
- S3: `ls`, `list-objects-v2` (keep it to a few pages; listing 260k keys is about
  260 LIST requests, roughly $0.001, which is fine to do *once*), `head-object`,
  `get-bucket-*` (location, CORS, policy, versioning)
- DynamoDB: `list-tables`, `describe-table`, `describe-time-to-live`
- Cognito: `list-user-pools`, `describe-user-pool`, `list-user-pool-clients`,
  `describe-user-pool-client`, `describe-user-pool-domain`
- CloudFormation: `list-stacks`, `describe-stacks`, `describe-stack-resources`,
  `get-template`, `validate-template`. `cfn-lint` is local and always allowed.
- IAM: `list-roles`, `get-role`, `list-role-policies`, `get-role-policy`,
  `list-attached-role-policies`
- Bedrock control plane: `list-foundation-models`, `list-inference-profiles`,
  `get-foundation-model-availability`
- SSM: `describe-parameters` (names only; **never** `get-parameter --with-decryption`)
- Budgets: `describe-budgets`

**B. Allowed without asking: small data-plane reads and writes at test scale.**
Each costs a fraction of a cent:
- S3 `GetObject` of the metadata Parquet (a few MB) and of individual catalogue images.
  Egress to the internet is free up to 100 GB/month across the account, and GETs cost
  $0.0004 per 1,000. Budget for at most **a few hundred image GETs per session**.
- DynamoDB reads and writes from running the app or tests against the existing
  checkpointer and Chainlit tables, at most **a few thousand requests per session**.
  Note: both tables are `PAY_PER_REQUEST` (on-demand). The DynamoDB free tier
  (25 RCU/WCU) applies only to *provisioned* capacity. On-demand isn't free, but at
  test scale it costs about $0.000001 per request, so this is fine. Storage up to 25 GB
  is free either way.
- Cognito sign-ins through the Chainlit UI. The first 10k monthly active users are
  free.

**C. Ask the user first, every time, with an estimate of the cost.** This applies to
anything above the default free tier, whether or not credits would pay for it.
- **Any Bedrock runtime call** (`InvokeModel`, `Converse`, and therefore running the
  agent with `llm.provider=bedrock` or `embedding.backend=nova`, or running the eval
  against Bedrock). Bedrock is billed per token or per embedding, and there is no
  free tier. Prefer `llm.provider=ollama` (§1.6, local and free) for development runs.
- **Any create, update or delete of an AWS resource**, even a free one:
  CloudFormation deploy, change set execution, resource import or stack deletion;
  creating buckets or tables; `PutObject`/`DeleteObject`; IAM changes; SSM
  `put-parameter`; Cognito changes. Per §1.8, these must always be done by running a
  committed script or template, never as an ad-hoc command. Ask before running the
  script.

**D. Never.**
- Delete or overwrite existing data or resources: buckets, objects, tables, stacks,
  user pools, roles. Removal candidates are listed for the user (Phase 14.4).
- Start anything with an hourly cost (EC2, NAT gateway, Elastic IP, ALB, ECR images,
  RDS).
- Call Cost Explorer (`ce:*` costs $0.01 per request) or touch billing settings.
- Use `--with-decryption`, or print the values of secrets or `.env`.

**E. Log every AWS call.** Keep a running list of every AWS CLI or SDK call you make
outside normal app or test runs: the command, why you made it, and whether it was A, B
or C. Put the list in the final report.

### 0.3 Local data

Never delete anything under `data/`, `.env`, `.chainlit/` or `.venv/` contents without
asking. In particular, **`data/qdrant_storage/` holds the Marqo-SigLIP collection
(`fashion_gen`), which is the intended local mode** (§1.4).

### 0.4 Linting

Run `uv run ruff check` (the whole repo, no path arguments). Phase 1.8 switches ruff to
`select = ["ALL"]`. When a rule should legitimately be bypassed, add it to `ignore` or
`per-file-ignores` **with a one-line justification comment**, or use an inline
`# noqa: XXX  <reason>` for one-off cases. Don't blanket-ignore a rule just to make the
count go down.

### 0.5 Verification after every phase

- `uv run ruff check` must show no new violations in files touched in that phase.
- From Phase 4 onwards, `env -i PATH="$PATH" HOME="$HOME" uv run python -c "import frag.cli, frag.agent.graph, frag.mcp_server.server, frag.evaluation.run"`
  must succeed with an empty environment, using whatever module names exist at that
  point. This proves that imports have no side effects (§3).
- From Phase 5 onwards, `uv run pytest` must pass with no network.

### 0.6 Working method

- Work through the phases in order and tick the checkboxes. If a step turns out to be
  wrong, record why under it instead of forcing it through.
- Behaviour must stay the same unless a step says it changes (bug fixes, or
  simplifications that are explicitly listed).
- Write plain, accurate docstrings and comments. Several current ones are inaccurate
  ("SQLite-backed vector database reader", "multidimensional state vector", "Random
  docstring."); replace them as you touch each function.

---

## 1. User decisions (settled)

| # | Decision |
|---|---|
| 1.1 | **Remove `dependency-injector`.** Keep the Pydantic configs, but make them pure and idempotent (no I/O beyond reading env, no instances built at import). Side effects only happen in entrypoints, under the rules in §3. |
| 1.2 | **Keep uv's link mode as configured** (`UV_LINK_MODE=symlink` in the user's shell). Don't add `link-mode = "copy"`. The README keeps the Chainlit `--link-mode copy` workaround. |
| 1.3 | **AWS usage** follows §0.2. |
| 1.4 | **Two embedding backends, both kept.** `siglip` is the local default: Marqo-FashionSigLIP runs locally for query text, against the `fashion_gen` collection in `data/qdrant_storage` (768-d, about 260k points). `nova` is the AWS mode: Amazon Nova Multimodal Embeddings on Bedrock, against the full ~10k Nova collection (decided 2026-11-26: use all of it rather than a 5k subset; no new embeddings are needed and the extra S3/RAM cost is a fraction of a cent per month). **This is feasible.** Qdrant point IDs and `payload.index_2` match the S3 image keys (`images/{idx//1000:03d}/{idx}.png`) and the Parquet index, so the SigLIP collection works unchanged with S3 images and metadata. In both modes, the LLM is remote. |
| 1.5 | **Catalogue images and metadata Parquet always come from S3.** Cognito auth and the Chainlit chat history (DynamoDB + S3) are always on, locally included. There's no local-filesystem catalogue backend. **Consequence:** the full 260k-image bucket must be **kept** (the local SigLIP mode needs it), which reverses the plan's "delete the full-size buckets". |
| 1.6 | **LLM providers: `bedrock` (Claude Haiku 4.5) and `ollama` (local, free, for development).** Gemini was dropped (2026-11-26) because its API wanted billing set up; Mistral is dropped too. `langchain-ollama` is the optional `local-llm` group. |
| 1.7 | **Checkpointer:** toggling between `memory`, `sqlite`, `postgres` and `dynamodb` should be as seamless as possible, through one setting. |
| 1.8 | **Every AWS resource is created by committed code** (CloudFormation templates, plus scripts that deploy them or upload data). No one-off console or CLI setup. Resources that already exist but were made by hand get codified (Phase 13). |
| 1.9 | **Remove the Bedrock batch-inference code and the JSONL files entirely.** Batch inference is blocked by an account limitation, so embeddings are produced with individual `InvokeModel` calls, and the AWS catalogue stays small. |
| 1.10 | **One graph run per user message** instead of interrupt and resume (Phase 7.6), since it's simpler with the same functionality. |
| 1.11 | **Ruff: start from `ALL`**, and add justified exceptions (§0.4). |
| 1.12 | **No new embeddings are generated.** The SigLIP collection (`fashion_gen`) and the Nova collection (`nova_fashion_gen`) already exist. Only the SigLIP *query-text* path is needed at runtime. The vector-DB build script is kept only as a reproducible record and isn't run. The AWS catalogue is the whole existing Nova collection (~10k points, decision 1.4); `qdrant_admin.py copy-subset` stays as a tool but no 5k cut is needed. |
| 1.13 | **Cognito sign-up is admin-only** (`AllowAdminCreateUserOnly: true`). |
| 1.14 | **Keep the Rich console logging.** It's per process and loguru sinks are thread-safe, so it's fine with several processes, each writing to its own console. |
| 1.15 | **Branching and merging are the user's.** PRs go into `main`. The agent may add GitHub Actions CI (Phase 16) as long as it involves no AWS integration and costs nothing. It doesn't change the existing branch-naming, commit-message or PR-title workflows. |

---

## 2. Target layout

```
app.py                          # Chainlit entry (Chainlit needs a file path)
compose.yaml                    # qdrant (always) + postgres (profile "postgres")
pyproject.toml / uv.lock / ruff.toml / .env.example / README.md
cleanup.md                      # this file (delete when done)
docs/
  frag_aws_completion_plan.md   # moved from data/ (data/ is git-ignored)
  aws_inventory.md              # Phase 13.1: every AWS resource and the code that owns it
  cloudformation.md             # notes salvaged from exploration/chainlit_auth_persistence/README.md
infra/
  cloudformation/               # moved from cloudformation/, extended in Phase 13
  scripts/deploy.sh             # aws cloudformation package + deploy (asks for confirmation)
  scripts/teardown.sh           # empties owned buckets, deletes stacks (asks for confirmation)
scripts/
  ingest_fashiongen.py          # HDF5 -> PNGs + Parquet -> S3 (restored from git history)
  build_vector_db.py            # embeds items (siglip | nova) and upserts to Qdrant
  qdrant_admin.py               # info / sample / snapshot / restore / copy-subset
  profile_resources.py          # optional, from exploration/profile_ram_cpu_util/profile_03.py
src/frag/
  settings.py                   # single Pydantic Settings tree + get_settings()
  prompts.py                    # prompt strings as module constants
  runtime.py                    # open_resources(): builds llm/checkpointer/catalogue/tools
  logging_setup.py
  llm.py                        # build_llm()
  checkpointer.py               # open_checkpointer()
  tokens.py                     # token-usage tracking
  messages.py                   # build_message() + image helpers
  cli.py                        # `frag chat | eval | mcp`
  catalogue/
    models.py                   # CatalogueItem dataclass
    store.py                    # S3ImageStore + load_metadata()
    embedders.py                # Embedder protocol, SigLIPEmbedder, NovaEmbedder, build_embedder()
    vector_db.py                # QdrantConnector (ids + scores only)
    service.py                  # Catalogue: search(), get(), categories
  tools/
    base.py                     # ToolsClient protocol + content-block helpers
    local.py                    # LangChain tools over Catalogue
    mcp_client.py               # MCP-backed ToolsClient
  mcp_server/server.py          # build_server(catalogue) -> FastMCP
  agent/
    state.py                    # AgentState
    schemas.py                  # LLM output schemas used by the agent
    graph.py                    # FashionAgent
  ui/
    node_updates.py             # NODE_META
    chainlit_patches.py         # the two Chainlit monkeypatches
  evaluation/
    schemas.py                  # RecommendationEvaluation
    run.py                      # run eval set, judge, score
tests/
  fixtures/shirt.jpg
  fakes.py
  unit/...
  integration/test_agent_graph.py
```

The exact layout is a suggestion. What matters is that scripts stay out of the
`src/frag` package, `utils/` stops being a grab-bag, and there's one module per concern.

---

## 3. Configuration and side effects across processes (design rules)

The app runs as **several independent processes**: the Chainlit app (the agent), the
MCP server, and sometimes `frag eval` or a script at the same time. Nothing is shared
in memory between them. These rules keep that tidy.

### 3.1 Configuration is pure and idempotent

- `Settings` (a Pydantic `BaseSettings`) reads environment variables and `.env`, and
  does nothing else: no network, no file creation, no logging setup.
  `Settings() == Settings()` for the same environment. It's `frozen=True`.
- `get_settings()` is `@lru_cache`d, so there's one instance per process. Tests call
  `get_settings.cache_clear()` or construct `Settings(...)` directly.
- **No module-level instances** of `Settings`, clients, `FastMCP`, loggers with sinks,
  or anything else that does I/O. Today `ChainlitPersistenceConfig()`,
  `PostgresCheckpointerConfig()` and `DynamoDBCheckpointerConfig()` are built as
  class-attribute defaults at import time, which reads `.env` at import and requires
  `POSTGRES_*` to be set. That goes away. Only the root `Settings` reads the
  environment, and its nested sections are plain `BaseModel`s.
- **No path validation in config.** Use `str`/`Path`, never `FilePath`/`DirectoryPath`
  with `validate_default=True`. The code that uses a path checks or creates it.

### 3.2 Side effects happen only in entrypoints, and are idempotent

The entrypoints are `app.py`, `frag.cli:main` (`chat`, `eval`, `mcp`), each script
under `scripts/`, and the pytest fixtures. Each one runs this sequence:

1. `load_dotenv()`. It's idempotent and doesn't override variables that are already
   set. It's needed so LangSmith, Chainlit, boto3 and the Ollama client see
   their variables in `os.environ`.
2. `settings = get_settings()`
3. `setup_logging(settings.logging, service="agent" | "mcp" | "eval" | "<script>")`.
   It must be idempotent: keep the handler IDs it added in a module-level variable,
   remove those (and only those) before adding again, and never call a bare
   `logger.remove()`.
4. `async with open_resources(settings, role=...) as res:` opens clients, connections
   and the checkpointer, and closes them on exit, including on error.
5. Chainlit monkeypatches: `chainlit_patches.apply()`. Plain attribute assignment is
   idempotent; also guard it with a module flag.

### 3.3 Rules for files written by more than one process

| Resource | Who writes | Rule |
|---|---|---|
| Log files (dev only; prod logs JSON to stdout) | every process | One file per `service` (`logs/frag-agent.log`, `logs/frag-mcp.log`, …). Never let two processes share a rotating file. |
| `data/token_usage.csv` | agent, eval | One append of short rows per session end. Add `service` and `thread_id` columns. Small appends are effectively atomic on local filesystems, so no locking is needed. |
| Temporary recommended images | agent, eval | Unique `uuid4` file names in `settings.agent.temp_dir`. Eval copies the images it needs into its own run directory (Phase 9.3), so the temp directory can be cleaned freely. |
| SQLite checkpointer file | agent only | `frag eval` and the tests default to the `memory` checkpointer, since they don't need persistence. If two agent processes ever share it, `AsyncSqliteSaver` uses SQLite locking; keep WAL mode, which is the library default. |
| Qdrant | MCP server or local tools (read), scripts (write) | The Qdrant server handles concurrency. Scripts that write must take `--collection` explicitly and confirm before a recreate. |

---

## Phase 1: Repository hygiene (no behaviour change)

- [x] **1.1 Delete `src/frag/exploration/`.**
  - **Salvage these first:**
    - `chainlit_auth_persistence/README.md` content goes into `docs/cloudformation.md`.
    - `dataset_wrangling/explore_nova_qdrant.py` and `create_nova_qdrant_snapshot.py`
      move unchanged to `scripts/legacy_qdrant_info.py` and
      `scripts/legacy_qdrant_snapshot.py`. Phase 10.3 merges them into
      `scripts/qdrant_admin.py` and deletes them.
    - `profile_ram_cpu_util/profile_03.py` optionally becomes
      `scripts/profile_resources.py`.
  - **Then delete these, which are superseded or broken:**
    - `uvicorn/*` (it imports Hydra and functions that no longer exist)
    - `wired_config.py` and `langgraph_injection_verification.py` (DI experiments)
    - `chainlit_auth_persistence/app.py` and `chainlit_table.json` (superseded by
      `app.py` and the YAML template)
    - `sample_dataset.h5` (a 4.9 MB tracked binary)
    - `saved_image_10.png`
- [x] **1.2 Delete `src/frag/config/exploration.py`** and the `exploration` field. It
  validates hard-coded `/mnt/windows/...` paths at import time.
- [x] **1.3 Delete `src/frag/utils/aws_utils.py`.** It creates a bucket and uploads the
  HDF5 file, which the S3-native layout has superseded. Nothing imports it.
- [x] **1.4 Delete `src/frag/utils/userdata.sh`.** It has `<YOUR_USERNAME>`
  placeholders, and the plan's compute template replaces it.
- [x] **1.5 Delete `src/frag/utils/aws_generate_embeddings.py`** (decision 1.9),
  along with every setting and env var that only it uses: `records_per_jsonl_file`,
  `jsonl_folder_path`, `embedding_s3_bucket_name`, `jsonl_s3_key_lambda`,
  `jsonl_s3_folder_name`, `embedding_output_folder_s3`,
  `bedrock_batch_inference_role_name`, `bedrock_batch_s3_access_policy_name`,
  `batch_embedding_job_prefix`, `AWS_ACCOUNT_ID` and
  `AWS_BEDROCK_BATCH_INFERENCE_ROLE_ARN`. The AWS leftovers (the embeddings bucket and
  `BedrockBatchInferenceRole`) are listed for the user in Phase 13.1; don't touch them.
  - `EnvSettings` uses `extra="forbid"`, so removing those two fields while `.env`
    still defines them would make startup fail. Switch it to `extra="ignore"` now
    (Phase 4.1 does this anyway), and tell the user they can remove the two variables
    from `.env`. The agent doesn't edit `.env`.
  - Remove `"frag.exploration"` from the `Container` wiring package list, or wiring
    fails on import.
- [x] **1.6 Move `cloudformation/` to `infra/cloudformation/`.** Move
  `data/frag_aws_completion_plan.md` to `docs/`, since `data/` is git-ignored and the
  plan currently isn't versioned.
- [x] **1.7 Clean up `.gitignore`.** Remove the stale entries: `/outputs/` (Hydra),
  `src/exploration/*`, `config/kaggle.json`, and the comment saying `.env` keys are
  "enumerated in main.py". Add `.files/`, `reports/`, `.ruff_cache/` and
  `.pytest_cache/`.
- [x] **1.8 Ruff: switch to `select = ["ALL"]`** (decision 1.11). Start `ruff.toml`
  with only these unavoidable exceptions, each commented:
  ```toml
  line-length = 88
  [lint]
  select = ["ALL"]
  ignore = [
      "D203",   # conflicts with D211 (blank line before class docstring); keep D211
      "D213",   # conflicts with D212 (summary line position); keep D212, matches google style
      "COM812", # conflicts with `ruff format` trailing-comma handling
      "ISC001", # conflicts with `ruff format` implicit string concatenation
  ]
  [lint.per-file-ignores]
  "tests/**" = [
      "S101",   # pytest uses bare assert
      "D",      # test functions are self-describing
      "PLR2004" # magic values are the point of assertions
  ]
  "scripts/**" = [
      "T201",   # scripts print progress to stdout by design
  ]
  [lint.pydocstyle]
  convention = "google"
  ```
  Then run `uv run ruff check --statistics` and record the baseline count by rule in
  your report. **Don't fix everything now.** Fix violations in each file as later
  phases touch it, and decide rule by rule whether a bypass is justified. Candidates
  likely to need a justified repo-wide ignore: `ANN401` (`Any` in LangChain callback
  signatures), `FBT001`/`FBT002` (boolean positional settings flags), `TRY003` (long
  messages in `ValueError`), `EM101`/`EM102`, and `G004` (only if loguru f-strings are
  kept; prefer loguru's `{}` formatting instead). By Phase 14, `uv run ruff check`
  must be clean.
- [x] **1.9 Fix `pyproject.toml` metadata.** Give it a real `description`, and replace
  `[tool.coverage.run] omit` (`src/frag/exploration/*`) with `scripts/*`.

---

**Phase 1 done (2026-11-26).** Notes:
- Baseline after the switch to `ALL`: **993 violations**. The biggest groups are
  E501 (184), ANN001 (161), ANN201 (91), D103 (43), I001 (32), D102 (21),
  LOG015 (21), Q000 (20) and E402 (17). That's 125 auto-fixable. There are also two
  real bugs flagged as F821: 3.1 (`img_path`) and 3.6 (`populate_vector_db`).
- Exceptions added beyond the initial four: `EXE002`, because the NTFS mount reports
  every file as executable; and `INP001` for `tests/**` (importlib mode) and
  `scripts/**` (standalone scripts).
- `data/frag_aws_completion_plan.md` is now `docs/frag_aws_completion_plan.md`, which
  makes it tracked. **Its last section (after the `====` line) contains personal notes
  about the resume and the job application.** The user should trim it before
  committing.
- `.env` still defines `AWS_ACCOUNT_ID` and `AWS_BEDROCK_BATCH_INFERENCE_ROLE_ARN`.
  They're harmless now that `EnvSettings` uses `extra="ignore"`, and the user can
  remove them.

## Phase 2: Dependencies and environment

- [x] **2.1 Dependencies** in `pyproject.toml`:
  - **Remove:**
    - `awscli`: a heavy CLI tool, not a library. Install it with
      `uv tool install awscli` or the system package, and document that in the README.
    - `langchain-mistralai` (decision 1.6)
    - `dependency-injector` (remove at the end of Phase 4)
  - **Add the packages that are imported but not declared:** `numpy`, `tqdm`,
    `langgraph-checkpoint-sqlite`, `aiosqlite`, `langgraph-checkpoint-postgres` and
    `psycopg-pool`. Keep `psycopg[binary]` for local development, and leave the plan's
    production note about building against `libpq` to the plan.
  - ~~**Add** `langchain-google-genai`~~, superseded: Ollama is used instead (Phase 5 notes). If Phase 4.2 shows Gemini can't take
    images in tool results, replace it with an optional group
    `local-llm = ["langchain-ollama"]`.
  - **Add optional groups:**
    - `siglip = ["torch", "open-clip-torch"]`, needed only when
      `embedding.backend=siglip`. The AWS image must not install it. Since no bulk
      embedding will be done (decision 1.12), pin torch to the **CPU wheel index** with
      `[tool.uv.sources]`: query-time text embedding is fast enough on CPU, and the
      wheel is about 10× smaller than the CUDA build.
    - `ingest = ["h5py"]`, needed only for `scripts/ingest_fashiongen.py`.
  - Keep `pandas` and `pyarrow` (metadata). Add `rich` explicitly; it's currently only
    a transitive dependency (decision 1.14).
- [x] **2.2 Don't change the link mode** (decision 1.2).
  - The current `.venv` is broken because the shell sets `UV_LINK_MODE=symlink`, and
    the cache entries the symlinks point to (under `$UV_CACHE_DIR/archive-v0/…`) were
    pruned.
  - Fix it with `uv sync --reinstall --all-groups`. Add a README note: with symlink
    mode, `uv cache clean`/`prune` breaks the venv, and `uv sync --reinstall` repairs
    it.
- [x] **2.3 Add `[project.scripts] frag = "frag.cli:main"`.**
- [x] **2.4 Run `uv lock` and `uv sync --all-groups`.**
- [x] **2.5 Remove the `--cov*` options from pytest `addopts`** so a plain `pytest` is
  fast. Document `pytest --cov=frag --cov-report=term-missing` in the README instead.

---

**Phase 2 done (2026-11-26).** Notes:
- Removed `awscli` and `langchain-mistralai`. `dependency-injector` stays until the end
  of Phase 4.
- Added `numpy`, `tqdm`, `rich`, `langgraph-checkpoint-sqlite`, `aiosqlite`,
  `langgraph-checkpoint-postgres`, `psycopg-pool` and `langchain-google-genai`.
- New groups:
  - `ingest = [h5py]`
  - `siglip = [torch, torchvision, open-clip-torch]`. `torchvision` was added because
    `open-clip-torch` depends on it and it must come from the same index as torch.
    Both are pinned to `https://download.pytorch.org/whl/cpu` (`explicit = true`), so
    the lock has `torch 2.15.0+cpu` on Linux and Windows and no `nvidia-*` or `triton`
    packages.
- The venv was repaired with `uv sync --reinstall --all-groups` (symlink mode kept),
  and there are now 0 broken symlinks. Chainlit was reinstalled as real files with
  `uv sync --all-groups --reinstall-package chainlit --link-mode copy`. That single
  command replaces the README's `uv remove`/`uv add` workaround, and the README is
  updated.
- `[project.scripts] frag = "frag.cli:main"` is installed, but invoking it raises
  `ModuleNotFoundError` until Phase 8 creates `frag.cli`.
- Coverage options were removed from `addopts`; the README documents the coverage
  command. The commented-out `log_file` pytest block was removed too.
- **No AWS CLI is left on the machine.** It was previously only available through the
  venv. The README now says to install v2 system-wide (on Arch:
  `sudo pacman -S aws-cli-v2`). It's needed before Phase 13.
- The existing test was **not run**, because its mock agent invokes the real
  `semantic_search` tool, which calls Bedrock Nova (§0.2C). `pytest --collect-only`
  succeeds. Phase 5 replaces the test with offline ones.
- Ruff is unchanged at 993 violations.

## Phase 3: Fix the outright bugs (small, targeted)

Do these before the big refactor so each fix is a separate, reviewable diff. Steps
marked *(subsumed)* are fixed automatically by a later phase.

- [x] **3.1 `utils/common_utils.py:76`**: `img_path` is undefined; it should be
  `img_format`. Any follow-up turn with a new uploaded image raises `NameError`.
- [x] **3.2 `evaluation/llm_as_judge.py:149,152`**: operator precedence. Change it to
  `(vis_wt*vis + suit_wt*suit + comp_wt*comp) / wt_sum`, and the same for the
  two-term branch.
- [x] **3.3 `evaluation/llm_as_judge.py:156`**: `judge_evaluation_outputs` has no
  `@inject`, so `model` is a `Provide` marker and the call raises. `track_token_use`
  then swallows the exception, so **the judge step currently does nothing**.
  *(subsumed by Phases 4 and 7.5)*
- [x] **3.4 `evaluation/llm_as_judge.py:63`**: `thread_id` should be `str(uuid4())`.
- [x] **3.5 `evaluation/llm_as_judge.py:207`**: the log message is missing its `f`
  prefix.
- [x] **3.6 `cli_main.py:29`**: `populate_vector_db()` is never imported. Remove that
  branch; building the vector DB is a script (Phase 10).
- [x] **3.7 `mcp_server/server.py:55`**: images are PNG but labelled `image/jpeg`.
- [x] **3.8 `rag_pipeline/rag_agent.py:511`**: the explanation node calls `invoke`
  without `config=self._callback_config`, so its tokens aren't counted.
- [x] **3.9 `utils/checkpointer.py:163`**: the error message omits `dynamodb`.
  *(subsumed by Phase 4.5)*
- [x] **3.10 `rag_agent.py:160`**: `{json.dumps(...)}` makes a set literal inside
  `log.debug`. Remove the braces.
- [x] **3.11 `rag_agent.py:130-142`**: `relevant_image_indexes` from the LLM aren't
  bounds-checked, so a bad index raises `IndexError`. Skip out-of-range indices with a
  warning.
- [x] **3.12 `rag_agent.py:383`**: the LLM-chosen `image_id` isn't validated, so a
  hallucinated ID raises `KeyError` or gets a 404 from S3. For now, wrap the lookup
  and skip the item with a warning. Phase 7.4 adds proper validation and a retry.
- [x] **3.13 `data_manager/dataset_read_write.py:40`**: the range check against
  `num_datapoints` is wrong for subsets. Use `fetch_index in metadata_lookup`.
  *(subsumed by Phase 6)*
- [x] **3.14 `dataset_read_write.py:43`**: a synchronous `s3_client.get_object` inside
  `async def` blocks the event loop. Use `asyncio.to_thread`. *(subsumed by Phase 6)*
- [x] **3.15 `rag_agent.py:541-543`**: the `> max_num_critiques` condition is
  unreachable. Simplify it to
  `"modifier_node" if state.critique_text.strip() else "explanation_node"`.
- [x] **3.16 `llm_schemas.py:118-122`**: `Field(..., default_factory=str)` is
  contradictory. Use `correction: str = Field("", description=...)`.
- [x] **3.17 MCP path in the recommender node: verify by reading the adapter source.**
  `recommender_node` parses db-tool output assuming the local-tools shape
  (`{"type": "image_url", "image_url": {"url": ...}}`). The raw MCP adapter tool
  (`langchain-mcp-adapters` 0.2.x) likely returns
  `{"type": "image", "base64": ..., "mime_type": ...}`, and `make_mistral_compatible`
  returns `image_url` as a plain string. With `USE_MCP_SERVER=true`, that probably
  raises "Could not find either image or description". Confirm this from the adapter
  source after Phase 2. Phase 6.6 removes the problem either way.

---

**Phase 3 done (2026-11-26).** All 17 steps were applied, including the ones marked
*(subsumed)*. They were each one or two lines and make the current code correct in the
meantime. Notes:
- **3.3:** added `@inject` under `@track_token_use`, so the judge now actually gets its
  model. `track_token_use` still swallows exceptions until Phase 7.5.
- **3.6:** the `recreate_vector_db` parameter and branch were removed from
  `cli_main.py`. The `recreate` config field stays until Phase 4.1.
- **3.11:** the bounds check uses
  `len(input_images_descriptions) + len(recommended_clothes_image_paths)`.
- **3.12:** the recommender skips items whose lookup fails or returns no image or
  description, with a warning. If every item fails, the turn ends with zero
  recommendations, and the UI already shows "No recommendations could be found". The
  broad `except` has a `noqa: BLE001` with its justification.
- **3.13:** the `num_datapoints` parameter was removed from `get_fashion_gen_data`.
- **3.14:** both `get_object` and `Body.read()` run in `asyncio.to_thread`.
- **3.15:** `@inject` and the unused `max_num_critiques` parameter were removed from
  the router.
- **3.17 confirmed** from the installed `langchain_mcp_adapters/tools.py`: for
  `ImageContent`, `_convert_mcp_content_to_lc_block` returns
  `create_image_block(base64=..., mime_type=...)`, which is
  `{"type": "image", "base64": ..., "mime_type": ...}`.
  - **The MCP path was broken:** every recommended item lacked an `image_url` block.
    Before 3.12 that raised `ValueError`; after 3.12 it would have silently skipped
    everything.
  - **Interim fix:** the recommender now also accepts `image` blocks. Phase 6.6 still
    replaces this parsing with `ToolsClient.get_item()`.
  - **Done in Phase 6.6:** `make_mistral_compatible` (used on the MCP *LLM*
    tools) emits `{"type": "image_url", "image_url": "<data-url string>"}`, a string
    rather than the `{"url": ...}` dict the local tools use. Normalise it there.
- No new ruff violations on changed lines. The total went from 993 to 984: both F821s
  are gone, plus a few incidental fixes.
- No AWS calls were made, and tests weren't run (per the user, tests run after all the
  fixes are in).

## Phase 4: Replace dependency-injector with explicit composition

At the end of this phase, nothing under `src/frag` imports `dependency_injector`.

### 4.1 Settings (`src/frag/settings.py`); follows §3.1

- [x] Merge `config/app_config.py`, `envs.py`, `data.py`, `models.py`,
  `orchestration.py`, `logs.py`, `tracking.py` and `eval.py` into one
  `Settings(BaseSettings)` with nested `BaseModel` sections. Keep the current default
  values. Suggested shape:

  ```python
  class Settings(BaseSettings):
      model_config = SettingsConfigDict(
          env_file=".env", env_nested_delimiter="__", extra="ignore", frozen=True,
      )
      app_env: Literal["development", "production"] = "development"
      aws_region: str = "us-east-1"
      llm: LLMSettings                  # provider: Literal["bedrock","ollama"]="bedrock", model_id, temperature,
                                        # rate_limit_rps: float | None, rate_limit_bucket, rate_limit_check_s
      embedding: EmbeddingSettings      # backend: Literal["siglip","nova"]="siglip",
                                        # siglip_model="hf-hub:Marqo/marqo-fashionSigLIP",
                                        # nova_model_id="amazon.nova-2-multimodal-embeddings-v1:0", nova_dim=1024,
                                        # nova_purpose="GENERIC_INDEX"
      catalogue: CatalogueSettings      # s3_bucket="frag-fashion-gen-dataset",
                                        # metadata_key="metadata/fashion_gen_metadata.parquet",
                                        # image_key_template="images/{shard:03d}/{idx}.png",
                                        # index_field="index_2", category_field="input_category",
                                        # description_field="input_description", price_field="input_msrpUSD"
      vector_db: VectorDBSettings       # url, prefer_grpc, image_vector_name="images",
                                        # collections: dict[backend, str] = {"siglip": "fashion_gen", "nova": "nova_fashion_gen"}
      tools: ToolsSettings              # mode: Literal["local","mcp"]="local", mcp_url, mcp_host="127.0.0.1", mcp_port=9000
      checkpointer: CheckpointerSettings  # see 4.5
      chat_history: ChatHistorySettings # dynamodb_table, s3_bucket (required; from .env)
      agent: AgentSettings              # default_num_recommendations, num_recommendation_attempts,
                                        # max_num_critiques, temp_dir, draw_graph_path: str | None
      evaluation: EvalSettings          # queries_path, images_dir, output_dir, weights
      logging: LogSettings              # level (derived from app_env if None), json: bool, file_dir: str | None,
                                        # library_levels: dict[str, str]
      token_usage_csv: str | None = "data/token_usage.csv"

      @property
      def collection(self) -> str: return self.vector_db.collections[self.embedding.backend]

  @lru_cache
  def get_settings() -> Settings: return Settings()
  ```
- [x] **Declare only what the app reads itself.** These read their own environment
  variables, so don't declare them:
  - LangSmith (`LANGSMITH_*`)
  - Chainlit and Cognito (`OAUTH_COGNITO_*`, `CHAINLIT_AUTH_SECRET`, `OAUTH_PROMPT`)
  - Ollama (`OLLAMA_HOST`)
  - FastMCP (`FASTMCP_*`)
  - Hugging Face (`HF_TOKEN`)
  - boto3 (`AWS_*` credentials)

  Declaring `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` made static keys *mandatory*,
  which stops boto3 falling back to an instance role. Drop `MISTRAL_API_KEY` and
  `use_mcp_server` (it becomes `TOOLS__MODE`). Use `extra="ignore"`: `.env` is shared
  with those libraries.
- [x] **Remove the hard-coded values:**
  - One `aws_region` replaces the four `"us-east-1"` literals.
  - `num_datapoints` is gone; the catalogue size is `len(metadata)`.
  - `product_categories` is gone; it's derived from the metadata at startup
    (Phase 6.5).
  - `fashion_gen_image_s3_key_lambda` is replaced by `image_key_template`. No lambdas
    in config.
- [x] **Delete the dead fields:**
  - SigLIP *ingestion* fields: `embedding_batch_size`, `data_fetch_batch_size`,
    `migration_batch_size`, `payload_update_batch_size`, `upload_batch_size`,
    `upload_parallel`. These become `scripts/build_vector_db.py` arguments if needed.
  - HDF5-era fields: `hdf5_path`, `string_attributes`, `string_codec`, `images_key`,
    `s3fs_block_size`, `dataset_object_name`. These become
    `scripts/ingest_fashiongen.py` arguments.
  - Fields that move to `build_vector_db.py`: `recreate`, `text_vectors_name`,
    `vectors_on_disk`, `hnsw_on_disk`, `payload_on_disk`, `indexing_threshold`.
  - MCP and S3 plumbing: `llm_tool_names`, `db_tool_name`, `client_transport_method`,
    `host_transport_method`, and the S3 pool and retry fields (hard-code one boto
    `Config` in `catalogue/store.py`).
  - `eval_mode` (it becomes a CLI subcommand) and `query_images_path`.
- [x] **Move prompts out of config** into `src/frag/prompts.py` as constants. Fix the
  typos while moving them: "You are a expert", "expert fashion_assistant", "Your role
  is judge", "You task is". The evaluation prompt moves to `evaluation/`.

### 4.2 LLM factory (`src/frag/llm.py`)

- [x] Replace `utils/model_factory.py` and the `ModelsConfig.llm_provider`
  `computed_field`, which makes config import utils (a layering inversion):
  ```python
  def build_llm(s: LLMSettings, region: str) -> BaseChatModel:
      limiter = (InMemoryRateLimiter(requests_per_second=s.rate_limit_rps, ...)
                 if s.rate_limit_rps else None)
      match s.provider:
          case "bedrock":
              from langchain_aws import ChatBedrockConverse
              return ChatBedrockConverse(model_id=s.model_id, temperature=s.temperature,
                                         rate_limiter=limiter, region_name=region)
          case "ollama":
              from langchain_ollama import ChatOllama
              return ChatOllama(model=s.model_id, temperature=s.temperature,
                                            rate_limiter=limiter)
  ```
- [x] ~~**Gemini caveats to verify.**~~ *Superseded: the user chose Ollama (Gemini wanted billing set up). The Ollama equivalents of these checks (images in tool results, structured output) are still open and need a local Ollama server with a vision + tool-calling model; do them before relying on Ollama for development runs.* Each check is one or two free-tier Google calls,
  which is allowed; Google isn't AWS.
  1. Pick a current vision-capable Gemini model that has free-tier access; check
     Google AI Studio for the current list. Set a conservative `rate_limit_rps` for
     it, since free tiers allow only a handful of requests per minute. Put both in
     `.env.example` as the "development" preset.
  2. **Images in tool results.** `semantic_search` returns image blocks inside a
     `ToolMessage`. Check whether `ChatGoogleGenerativeAI` accepts that, and whether the
     model actually uses the images, with one real tool round-trip.
     **If it doesn't work, switch to Ollama** (decision 1.6): replace the `google` case
     with
     `case "ollama": from langchain_ollama import ChatOllama; return ChatOllama(model=s.model_id, temperature=s.temperature)`.
     Swap `langchain-google-genai` for the optional `local-llm` group, and set the
     development preset to a vision-capable Ollama model. Report which way it went.
  3. `with_structured_output` works for every schema in `agent/schemas.py` (one call
     each is enough).

### 4.3 Resource composition (`src/frag/runtime.py`)

- [x] Replace `config/container.py` (14 `manage_*` functions, the
  `in_mcp_server_process` override and the `_conditional_checkpointer` wrapper) with:
  ```python
  @dataclass
  class Resources:
      settings: Settings
      llm: BaseChatModel | None
      checkpointer: BaseCheckpointSaver | None
      catalogue: Catalogue | None
      tools: ToolsClient | None

  @asynccontextmanager
  async def open_resources(settings: Settings, role: Literal["agent", "mcp", "eval"]):
      async with AsyncExitStack() as stack:
          needs_catalogue = role == "mcp" or settings.tools.mode == "local"
          catalogue = (await stack.enter_async_context(open_catalogue(settings))
                       if needs_catalogue else None)
          llm = build_llm(settings.llm, settings.aws_region) if role != "mcp" else None
          checkpointer = None
          if role != "mcp":
              cp_settings = settings.checkpointer if role == "agent" else CheckpointerSettings(backend="memory")
              checkpointer = await stack.enter_async_context(open_checkpointer(cp_settings, settings.aws_region))
          tools = None
          if role != "mcp":
              tools = (LocalToolsClient(catalogue) if settings.tools.mode == "local"
                       else MCPToolsClient(settings.tools.mcp_url))
          yield Resources(settings, llm, checkpointer, catalogue, tools)
  ```
- [x] Logging is **not** a resource. Entrypoints call `setup_logging` first (§3.2).

### 4.4 Thread resources through explicitly

- [x] `FashionAgent(llm, tools, checkpointer, settings: AgentSettings, callbacks=None)`.
  Remove every `@inject`/`PV[...]`; nodes read `self._settings` and `prompts.*`.
  Compile the graph in `__init__`, which is synchronous.
- [x] `mcp_server.server.build_server(catalogue) -> FastMCP` (Phase 6.7).
- [x] Evaluation functions take `(resources, eval_settings)` (Phase 9).

### 4.5 Checkpointer (`src/frag/checkpointer.py`); decision 1.7

- [x] Settings:
  ```python
  class CheckpointerSettings(BaseModel):
      backend: Literal["memory", "sqlite", "postgres", "dynamodb"] = "sqlite"
      sqlite_path: str = "data/pipeline_checkpoints.db"
      postgres_dsn: str = "postgresql://frag:frag@localhost:5432/frag"   # override via CHECKPOINTER__POSTGRES_DSN
      postgres_pool_size: int = 10
      dynamodb_table: str = "langgraph-checkpoints"                      # override via CHECKPOINTER__DYNAMODB_TABLE
      ttl_seconds: int = 86_400
      compress: bool = True
  ```
  Replace `POSTGRES_USER/PASSWORD/DB` with a single DSN, and replace
  `AWS_DYNAMODB_CHECKPOINTER_TABLE_NAME` with `CHECKPOINTER__DYNAMODB_TABLE`.
- [x] Implementation, replacing the ABC, the three provider classes and the factory
  (164 lines):
  ```python
  @asynccontextmanager
  async def open_checkpointer(c: CheckpointerSettings, region: str):
      match c.backend:
          case "memory":
              yield InMemorySaver()
          case "sqlite":
              from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
              Path(c.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
              async with AsyncSqliteSaver.from_conn_string(c.sqlite_path) as saver:
                  yield saver
          case "postgres":
              from psycopg_pool import AsyncConnectionPool
              from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
              async with AsyncConnectionPool(c.postgres_dsn, max_size=c.postgres_pool_size,
                                             kwargs={"autocommit": True, "prepare_threshold": 0}) as pool:
                  saver = AsyncPostgresSaver(pool)
                  await saver.setup()       # idempotent
                  yield saver
          case "dynamodb":
              from langgraph_checkpoint_aws import DynamoDBSaver
              yield DynamoDBSaver(table_name=c.dynamodb_table, ...)  # same args as today; check the installed signature
  ```
  Keep the explanatory comments about `autocommit`/`prepare_threshold`; they're good.
- [x] Add a `postgres` service to `compose.yaml` under `profiles: ["postgres"]`
  (`postgres:17`, bound to loopback, with a named volume and a healthcheck), so that
  `podman compose --profile postgres up -d` plus `CHECKPOINTER__BACKEND=postgres` is
  the whole toggle.
- [x] Unit-test `memory` and `sqlite` offline. Test `postgres` only if the compose
  service is running; skip otherwise with `pytest.mark.skipif` on connection failure.
  Test `dynamodb` only under `@pytest.mark.aws` (§0.2B allows it, but it stays
  deselected by default).

### 4.6 Remove the DI package

- [x] Delete `config/container.py` and the `config/` package.
- [x] `grep -rn "dependency_injector\|Provide\[\|@inject\|Container" src tests app.py`
  returns nothing.
- [x] Remove `dependency-injector` from `pyproject.toml` and run `uv lock`.

---

**Phase 4 done (2026-11-26)**, except the Gemini checks in 4.2, which need a
`GOOGLE_API_KEY` from the user. Notes:
- **New modules:** `settings.py`, `prompts.py`, `llm.py`, `checkpointer.py`,
  `runtime.py`, `tokens.py`, `evaluation/prompts.py` and `data_manager/catalogue.py`.
- **Deleted:** the `config/` package, `utils/model_factory.py`, `utils/checkpointer.py`
  and `dependency-injector`. A grep for DI references across `src`, `tests`, `scripts`
  and the entrypoints comes back empty.
- **Pulled forward from later phases**, because removing DI touched the same code:
  - 7.5: `tokens.track_tokens` replaces the exception-swallowing `track_token_use`.
  - 11.2: the DI-only `get_qdrant_points_by_id` is deleted.
  - 6.2: the metadata loader reads only the 4 columns it uses.
  - 6.5: categories are derived from the metadata (48 for the full catalogue).
  - 6.7 (partly): `FastMCP` is no longer a module-level global.
  - 11.3 (partly): `setup_logging` is idempotent and uses a per-service file name
    (`logs/*/frag-{agent,cli,eval,mcp}.log`). The Rich console is kept.
  - 8.1 (partly): `cli_main.py` takes `chat|eval` with argparse, since the eval-mode
    setting is gone. `mcp_server.py` no longer checks `USE_MCP_SERVER`.
- **Interim structure until Phase 6:** `get_fashion_gen_data` became
  `FashionGenData.get()`, and `Catalogue` is a plain frozen dataclass (data, vector DB,
  embedder, categories) opened by `open_catalogue()`. `EMBEDDING__BACKEND` defaults to
  `nova`; the default switches to `siglip` once Phase 6 restores that embedder.
- **Behaviour changes:**
  - Eval uses the in-memory checkpointer (§3.3).
  - The token CSV timestamp is now ISO UTC (`+00:00`).
  - `FashionAgent` compiles its graph in `__init__`, so `compile_graph()` is gone.
  - In `app.py`, all imports come first, then `load_dotenv()` and the monkeypatches.
    Nothing reads env or patched globals at import time.
- **`.env` renames needed** (the user does these; the agent doesn't edit `.env`):
  - `AWS_DYNAMODB_CHECKPOINTER_TABLE_NAME` becomes `CHECKPOINTER__DYNAMODB_TABLE`
  - `AWS_DYNAMODB_CHAINLIT_PERSISTENCE_TABLE_NAME` becomes
    `CHAT_HISTORY__DYNAMODB_TABLE`
  - `AWS_S3_CHAINLIT_PERSISTENCE_BUCKET_NAME` becomes `CHAT_HISTORY__S3_BUCKET`
  - `USE_MCP_SERVER=false|true` becomes `TOOLS__MODE=local|mcp`
  - Add `CHECKPOINTER__BACKEND=dynamodb` to keep the current behaviour (the default is
    now `sqlite`).
  - Optional: `POSTGRES_*` becomes `CHECKPOINTER__POSTGRES_DSN`.
  - `MISTRAL_API_KEY` can go.
  - `.env.example` is rewritten to match (part of Phase 14.1, done early).
- **`compose.yaml`:** a `postgres` service under `profiles: ["postgres"]`, bound to
  loopback, with a named volume and a healthcheck.
- **Ruff:** new and rewritten files are clean, and changed lines in partially edited
  files are clean. Total 984 → 396. New justified ignores:
  - tests: `ANN` (fixtures are injected by name)
  - `**/prompts.py`: `E501` (wrapping would change the prompt text)
  - `src/frag/{llm,checkpointer}.py`: `PLC0415` (lazy backend imports)
  - inline `BLE001` on the eval loops and the entrypoint top-level handlers
  - inline `S105` on `token_usage_csv` (false positive)
- **Smoke checks** (not the pytest suite; no Bedrock calls):
  - Empty-environment import of every module: passes. `Settings() == Settings()`.
  - Checkpoint put/get/delete on `memory`, `sqlite` (temp dir), `postgres` (compose
    profile) and `dynamodb` (the existing table; test thread deleted): all pass.
  - `open_resources(role="mcp")`: S3 metadata (260,490 items, 48 categories) and local
    Qdrant `nova_fashion_gen` (**10,000 points**). The search path, using a stored
    vector instead of an embedding, the image fetch, an unknown ID (`{}`) and the local
    lookup tool all pass.
  - `mcp_server.py` plus `MCPToolsClient`: tool discovery passes. The lookup tool
    returns `['text', 'image']` blocks, which the recommender's 3.17 parsing handles.
  - Containers were stopped afterwards.
- **AWS call log (§0.2E):**
  - (B) DynamoDB checkpointer table: about 6 requests (put, get, delete thread, get).
  - (B) S3 `frag-fashion-gen-dataset`: the metadata Parquet downloaded twice (the
    catalogue check and the MCP server startup), plus 4 image GETs.
  - No Bedrock calls, and no resource changes.
- The integration test's `container` fixture is removed, so the old test can't run
  until Phase 5 rewrites it.

## Phase 5: Offline tests (the safety net for Phases 6–9)

The current test can't run offline: `container.init_resources()` touches S3, DynamoDB,
Qdrant and Bedrock, and it needs an absolute-path image under the git-ignored `data/`.

- [x] **5.1** Add `tests/fixtures/shirt.jpg`, a small image generated with PIL (a
  64×64 solid colour is fine).
- [x] **5.2 `tests/fakes.py`:**
  - `ChatMockLLM` (from `tests/mocks/`) with `ainvoke` added.
  - `MockAgent`/`mock_create_agent`, returning IDs from the fake catalogue instead of
    `random.randint(0, 260400)`.
  - `FakeToolsClient` (`get_llm_tools`, `get_item`) and `FakeEmbedder`.
- [x] **5.3** Rewrite the integration test as `tests/integration/test_agent_graph.py`:
  - Build
    `FashionAgent(ChatMockLLM(), FakeToolsClient(), InMemorySaver(), AgentSettings(temp_dir=tmp_path))`.
  - Run one image turn, then one text follow-up.
  - **Assert on the returned state** rather than on log substrings: three image files
    exist, the explanation is non-empty, and the critique loop ran
    `max_num_critiques` times.
- [x] **5.4** Delete `tests/integration/conftest.py` and `tests/mocks/`. Keep the
  loguru `caplog` bridge. Make the session-wide `logs/pytest.log` sink opt-in.
- [x] **5.5 Unit tests:**
  - both routers
  - `get_final_score` (it would have caught bug 3.2)
  - `build_message` (it would have caught bug 3.1)
  - `Settings()` with an empty environment
  - settings idempotence (`Settings() == Settings()`), and that importing
    `frag.settings` reads nothing
  - `setup_logging` called twice adds no duplicate sinks
  - `Catalogue.get`/`search` with a fake image store, vector DB and embedder
  - the MCP block normaliser
  - `open_checkpointer` with `memory` and `sqlite`
- [x] **5.6** Register the markers `aws` (real AWS, §0.2B scope only) and `bedrock`
  (costs money, §0.2C). Use `addopts = -m "not aws and not bedrock"`.

---

**Phase 5 done (2026-11-26).** Notes:
- **Done first:**
  - **Switched Google to Ollama** (user decision). `llm.provider` is now
    `bedrock | ollama`, `langchain-google-genai` was removed, and
    `local-llm = [langchain-ollama]` was added. `ChatOllama` builds offline. Ollama
    is installed but not running, so the Ollama image-in-tool-result check is still
    open (Phase 4.2).
  - **`.env` verified.** All renamed keys are present and the legacy names are gone.
    The DynamoDB checkpointer table and the Chainlit table are ACTIVE, and the Chainlit
    bucket exists (read-only `describe-table` / `head-bucket`).
    **One problem: `TOOLS__MODE=false` is invalid.** It must be `local` (the old
    `USE_MCP_SERVER=false`), and until it's fixed `Settings()` raises at startup. A
    test now covers exactly this.
- **New tests:** `tests/fakes.py`, `tests/conftest.py`, `tests/fixtures/shirt.jpg`
  (64×64 JPEG, 693 bytes), `tests/integration/test_agent_graph.py`, and
  `tests/unit/test_{agent_nodes,catalogue,infrastructure,settings_and_logging}.py`.
  `tests/mocks/`, `tests/integration/conftest.py` and the old
  `tests/integration/rag_pipeline/` were deleted.
- **Coverage:**
  - Two real turns through the compiled graph: an image turn, then a text follow-up
    that asserts the bounds check (3.11) and description reuse.
  - The text-only path, the routers, and the critique limit.
  - The recommender with both tool-block shapes (3.17) and with an unknown ID (3.12).
  - Settings: defaults, idempotence, nested overrides, invalid or mistyped values,
    and a subprocess import with an empty environment that writes no files.
  - Idempotent logging setup.
  - Catalogue get, column pruning, category derivation, and Qdrant search via an
    in-memory `AsyncQdrantClient` (category filter, metadata-missing skip).
  - All four checkpointers. Postgres skips unless the compose profile is up;
    DynamoDB is `@pytest.mark.aws`.
  - The LLM factory, token tracking, message MIME types (3.1), the final score (3.2)
    and the evaluation inputs.
- **The MCP normaliser test (5.5) is deferred to Phase 6.6.** The normaliser's output
  is meant to change there. The recommender's handling of MCP blocks is covered
  instead.
- **Small source change:** `derive_categories()` was extracted in
  `data_manager/catalogue.py` so it can be unit-tested.
- **pytest config:**
  - `asyncio_mode = "auto"`
  - `pythonpath = ["."]`
  - markers `aws` and `bedrock`, deselected by default
  - opt-in debug log via `FRAG_TEST_LOG_FILE`
- **Results:**
  - Default run: **35 passed, 1 skipped (postgres), 1 deselected (aws)** in about 3 s.
  - It passes the same way inside `unshare -rn` (no network interfaces), so it's
    genuinely offline.
  - With Postgres up: the Postgres test passes.
  - `pytest -m aws`: the DynamoDB test passes.
- **Ruff:** clean on `tests/`. New justified test ignores: `EM`, `TRY003` and
  `ASYNC240`, plus an inline `ARG001` on a fake tool's schema-mirroring parameter.
  `tests/` was formatted with `ruff format`. Total 396 → 353.
- **AWS call log:**
  - (A) 2× `dynamodb describe-table` and 1× `s3 head-bucket` (`.env` verification).
  - (B) About 6 DynamoDB requests from `pytest -m aws`; the test thread was deleted.
  - No Bedrock calls.

## Phase 6: Catalogue, embedders and tools

Right now the same logic exists twice: `rag_pipeline/local_tools.py` and
`mcp_server/server.py` have identical method bodies. `QdrantConnector` also imports the
DI-injected S3 fetch inside a method to dodge an import cycle.

- [x] **6.1 `catalogue/models.py`**:
  `@dataclass(frozen=True) class CatalogueItem: id: int; category: str; description: str; price: float; image_b64: str; score: float | None = None`.
- [x] **6.2 `catalogue/store.py`** (S3 only, decision 1.5):
  - `S3ImageStore(bucket, key_template, client)` with
    `async def get_png_bytes(idx) -> bytes`, implemented with
    `asyncio.to_thread(client.get_object, ...)`.
  - `async def load_metadata(client, settings) -> dict[int, dict]`. Download the
    Parquet once, **read only the needed columns** (index, category, description,
    price), and `set_index(index_field)` / `to_dict("index")`. With 260k rows in
    SigLIP mode, the column pruning noticeably reduces RAM.
  - One shared boto3 `Config` (pool size 20, adaptive retries, 3 attempts) for S3.
- [x] **6.3 `catalogue/embedders.py`** (decision 1.4):
  - `class Embedder(Protocol)`: `dim: int`, `async def embed_text(self, text: str) -> list[float]`.
  - `NovaEmbedder(model_id, dim, purpose, region)` wraps `invoke_model` in
    `asyncio.to_thread`. Remove the `get_text_embedding_batch` loop; the only caller
    passes one string.
  - `SigLIPEmbedder(model_name)`: restore it from git history
    (`git show 2583639^:src/frag/data_manager/vector_db_writer.py`, class
    `FashionSigLIPEmbedding`). Keep **only the text path** (decision 1.12). Import
    `torch`/`open_clip` **inside `__init__`** so the AWS image never needs them.
    `embed_text` runs `encode_text(..., normalize=True)` in `asyncio.to_thread`. The
    model loads once per process (only in the MCP server, or the agent when
    `tools.mode=local`), and it takes a few seconds; log when it's ready.
  - `build_embedder(settings) -> Embedder`.
- [x] **6.4 `catalogue/vector_db.py`**: `QdrantConnector`.
  - `async validate(expected_dim)` checks that the collection exists **and** that its
    `image_vector_name` vector size equals the embedder's `dim` (768 for SigLIP, 1024
    for Nova). This catches backend/collection mismatches at startup.
  - `async search(vector, categories, k) -> list[tuple[int, float]]` returns IDs and
    scores only, and no longer calls the S3 fetch.
- [x] **6.5 `catalogue/service.py`**:
  ```python
  class Catalogue:
      categories: list[str]   # sorted(set(m[category] for m in metadata.values())) - honest for subsets
      async def search(self, description, categories, k) -> list[CatalogueItem]  # validate cats, embed, search,
                                                                                 # fetch images via asyncio.gather;
                                                                                 # skip ids missing from metadata
      async def get(self, item_id: int) -> CatalogueItem | None
  ```
  Plus `open_catalogue(settings)`, an async context manager that builds the S3 client,
  loads the metadata, builds the embedder, and opens and validates Qdrant. Delete
  `data_manager/`.
- [x] **6.6 `tools/`**:
  - `base.py`:
    - `ToolsClient` protocol with `get_llm_tools()` and `get_item(id)`. `get_item`
      replaces `get_db_tool()`, so the agent no longer parses tool blocks.
    - `item_to_blocks(item)`, the one LangChain content formatter.
    - `normalise_mcp_blocks()`, renamed from `make_mistral_compatible`. It emits the
      same `{"type": "image_url", "image_url": {"url": ...}}` shape as
      `item_to_blocks`, so both modes give the LLM identical tool output. It also
      has no provider-specific branches.
  - `local.py`: `LocalToolsClient(catalogue)` builds `semantic_search` and
    `get_product_categories` with `StructuredTool.from_function` over closures, and
    keeps the current tool docstrings, which the LLM reads.
  - `mcp_client.py`: `MCPToolsClient(url)` caches `client.get_tools()` after the first
    call. `get_item` calls `get_datapoint_by_index` and parses both block shapes into a
    `CatalogueItem`.
  - Delete `utils/llm_tools_setup.py`, `rag_pipeline/local_tools.py` and
    `get_tool_with_name`.
- [x] **6.7 `mcp_server/server.py`**:
  - `build_server(catalogue) -> FastMCP` creates the instance (replacing the
    module-level global) and registers three plain async functions with `@mcp.tool`,
    which return `TextContent` + `ImageContent(mimeType="image/png")`.
  - Delete the class, the stale numbered module docstring, and the unused
    `base64`/`numpy`/`asyncio` imports.
  - Add a `GET /health` custom route.
  - Run with `run_async(transport="http", host=settings.tools.mcp_host, port=...)`.
- [x] **6.8** The recommender node uses `await self._tools.get_item(id)` and
  `save_image_b64(temp_dir, item.image_b64)`.

**Phase 6 done (2026-11-26).** Notes:
- **New packages:** `frag/catalogue/` (`models`, `store`, `embedders`, `vector_db`,
  `service`) and `frag/tools/` (`base`, `local`, `mcp_client`). Deleted:
  `data_manager/`, `utils/llm_tools_setup.py`, `rag_pipeline/local_tools.py`.
  `build_tools_client()` lives in `runtime.py`, since `tools/base.py` can't import its
  own implementations.
- **Deviations from the spec:**
  - `load_metadata` renames the Parquet columns to the canonical keys `category`,
    `description` and `price`, so nothing past the store needs the field-name settings.
  - `CatalogueSettings.image_key()` was removed; `S3ImageStore.key()` replaced it.
  - `QdrantConnector` takes an open `AsyncQdrantClient` (so tests can pass an in-memory
    one), and `validate()` also checks that the named vector exists.
  - `Catalogue.search` raises `InvalidCategoryError`; both tool modes turn it into the
    same `Error: ...` text block. Hits missing from the metadata, or whose S3 image
    fetch fails, are dropped (the failures are logged).
  - The tool descriptions are shared constants in `tools/base.py`. Both modes return
    the categories as a JSON text block, so the LLM sees byte-identical tool output in
    both modes (tested over real HTTP).
  - `score` is `null` (not `0`) for items that didn't come from a search.
  - `save_image_url_to_folder` became `save_image_b64` in `utils/common_utils.py`
    (`messages.py` is a later phase).
- **SigLIP is the default backend again** (`EMBEDDING__BACKEND=siglip`, decision 1.4).
  The `siglip` group gained `transformers>=5.2.0`, because open_clip's tokenizer for
  this model needs it (the old pyproject had it). The lock also bumped
  `huggingface-hub` and `tokenizers`. The model loads in about 5 s on CPU and gives
  768-d unit vectors.
- **Tests:** 45 passed, 1 skipped (postgres), 1 deselected (aws). It still passes inside
  `unshare -rn` with only loopback up.
  - New: `tests/unit/test_tools.py` covers the formatter round trip, the MCP block
    normaliser (the 5.5 test deferred here), the local tools, and the real MCP server
    on a local port, with parity against the local tools plus `/health`.
  - `test_catalogue.py` was rewritten for the new modules: dimension validation, dropped
    hits, and the Nova request body via a stub client.
  - `FakeToolsClient` now implements `get_item`.
- **Real smoke tests (scripts kept out of the repo):**
  - Nova path: the compose Qdrant (`nova_fashion_gen`, **10,000 points**, 1024-d
    `images` and `texts`) with real S3. A stored text vector stood in for Bedrock.
    Local and MCP `semantic_search` output was identical, `get_item` matched across
    modes, and unknown IDs gave `None`.
  - SigLIP path: run against a **copy** of `data/qdrant_storage` in a throwaway
    container (the original was never opened, because Qdrant may migrate the storage
    format on open). "black leather ankle boots" in BOOTS returned black calfskin ankle
    boots.
  - **Watch out:** with `prefer_grpc=true`, the client uses gRPC port 6334 whatever port
    `VECTOR_DB__URL` gives. A Qdrant on a non-default port needs `prefer_grpc=false`
    (or a `grpc_port` setting later).
  - **For the AWS catalogue phase:** categories are derived from the metadata, and the
    full Parquet has 260,490 rows and 48 categories. With the 10k Nova collection,
    categories that have no vectors would still be offered until a matching Parquet
    is used.
- **Ruff:** the new and touched files are clean (there's one pre-existing long log line in
  `rag_agent.py`, which Phase 7 rewrites). Total 353 → 293.
- **AWS call log:**
  - (A) 1× `s3api head-object` on the metadata Parquet.
  - (B) 2× metadata Parquet GET (5.9 MB each) and about 10 image GETs from the smoke
    tests. About 6 DynamoDB requests from one `pytest -m aws` run.
  - No Bedrock calls.

---

## Phase 7: Agent graph (`agent/graph.py`, `state.py`, `schemas.py`)

- [x] **7.1 Make every node `async` and use `ainvoke`**, which resolves the README
  TODO. Update the mocks to match.
- [x] **7.2 `AgentState`**: replace `Optional[T] = Field(default_factory=T)` with
  plain `T = default`. Fix the docstring: `recommended_clothes_explanation` is `str`,
  and `cfg.rag_pipeline.*` no longer exists.
- [x] **7.3 Schemas**:
  - Delete `ValidCategories`, which is unused and a third copy of the 48 categories.
  - Move `RecommendationEvaluation` to `evaluation/schemas.py`.
  - Fix the concatenated descriptions that are missing spaces ("user'srequests",
    "nospecification").
- [x] **7.4 Recommender node**:
  - Fetch the tools once per call.
  - **Validate `image_id`**: collect the IDs that appeared in `semantic_search` tool
    results in `response["messages"]`. If the chosen ID isn't among them, retry once
    with a corrective message, then skip it. This is the plan's Phase 0.3.
  - Rewrite the docstring and remove the stale TODO.
  - **De-duplicate repeated item IDs** (added 2026-11-26 at the user's request). If
    two descriptions end up with the same catalogue item, show it once. Keep the
    first occurrence and the original order (`dict.fromkeys`), so the user isn't
    shown the same product twice and the critique and explanation nodes don't judge
    a duplicate.
  - Optional: run the per-description agent calls with `asyncio.gather`. The shared
    rate limiter throttles them anyway. Keep it sequential if a local Ollama model's
    limits make it pointless.
- [x] **7.5 `tokens.py`**: replace the `track_token_use` decorator. It injects a
  `callback_config` kwarg, duplicates its sync and async branches, and **swallows all
  exceptions and returns `None`**, which hid bug 3.3. Use:
  ```python
  @contextmanager
  def track_tokens(csv_path: str | None, service: str, thread_id: str | None = None):
      handler = UsageMetadataCallbackHandler()
      try:
          yield handler
      finally:
          if csv_path: append_usage_csv(csv_path, handler.usage_metadata, service, thread_id)
          logger.info("token usage {}", handler.usage_metadata)
  ```
  Exceptions propagate.
- [x] **7.6 One graph run per user message** (decision 1.10):
  - **Current flow:** `START → human_node` (which calls `interrupt()`) `→ … →
    explanation_node → human_node`. Every session must be primed with
    `ainvoke({"is_chat_start": True})` and then resumed with `Command(resume=...)`.
    That's duplicated in `app.py`, the CLI and the eval.
  - **New flow:** `START → intake_node → quantifier_node → (intent → vision) →
    modifier → recommender → critique → (modifier | explanation) → END`. Each message
    is `await agent.run_turn(text, image_paths, thread_id)`, which calls
    `graph.ainvoke({"new_input_text": ..., "new_input_images": [...]}, config)`, or
    `astream` for the UI.
  - `intake_node` does what `human_node` does today. If the checkpointed state has a
    previous `input_text`, it runs the "follow-up" branch (`UpdatedUserRequest`
    synthesis). Otherwise it treats the input as a first turn. It also resets the
    per-turn fields (`num_times_critiqued`, `critique_text`).
  - Remove `is_chat_start`, `interrupt`, `Command` and the priming calls.
  - Chainlit resume works unchanged, because state is keyed by `thread_id`.
  - Update `ui/node_updates.py` (`human_node` becomes `intake_node`) and the tests.
- [x] **7.7 Modifier node**: prefix a non-empty critique with "Feedback on the
  previous attempt:" when inserting it into the prompt.
- [x] **7.8** Rename `draw_langraph_topology` to `draw_graph`. Run it only if
  `agent.draw_graph_path` is set. Note in the comment that it calls the mermaid.ink
  web API.
- [x] **7.9** Rewrite the node docstrings plainly.
- [x] **7.10 Default LLM: Ollama `qwen3.5:4b`** (added 2026-11-27 at the user's request;
  it is the model known to work locally, and it's free, per §0.2). `LLM__PROVIDER`
  defaults to `ollama`. `model_id` and `rate_limit_rps` default per provider
  (`_LLM_PROVIDER_DEFAULTS` in `settings.py`): `qwen3.5:4b` and no rate limit for
  Ollama, Claude Haiku 4.5 and 0.8 rps for Bedrock, so `LLM__PROVIDER=bedrock` alone
  is a working configuration. `reasoning` defaults to `false`. `.env.example` and
  the README install line are updated.
- [x] **7.11 Review every prompt** (added 2026-11-27). See the notes below.
- [x] **7.12 Model-quality tests with catalogue ground truth** (added 2026-11-27;
  Phase 9's eval is an LLM judge with no ground truth, so this is new).
  `tests/model/`, opt-in with `uv run pytest -m ollama`. See the notes below.

**Phase 7 done (2026-11-26; 7.10–7.12 on 2026-11-27).** Notes:
- **Layout:**
  - `rag_pipeline/` was replaced by `frag/agent/` (`graph.py`, `state.py`, `schemas.py`).
  - `RecommendationEvaluation` moved to `evaluation/schemas.py`.
  - `utils/ui_node_updates.py` moved to `ui/node_updates.py` and was tidied (it's
    lint-clean now).
  - `ValidCategories` was deleted.
- **API:** `FashionAgent(llm, tools, checkpointer, settings)` no longer takes
  `callbacks`. Instead `run_turn(text, image_paths, thread_id, callbacks=None)` and
  `stream_turn(...)` (which yields node updates) put the callbacks in the run config.
  Every node takes `config: RunnableConfig` and passes it to its LLM or agent call, so
  a handler sees all of the turn's LLM calls.
  - **The graph now ends at `END`,** with no `interrupt` or `Command`. A thread whose
    checkpoint has an `input_text` is treated as a follow-up.
  - **Old saved threads still work:** I tested the 3 newest threads from a copy of
    `data/pipeline_checkpoints.db`. They continue as follow-ups, and the stale
    `is_chat_start` key is ignored.
- **Callers updated:**
  - `app.py`: no priming call. Each message runs `stream_turn` inside
    `track_tokens(csv, "agent", thread_id)`, so tokens are now recorded per message
    instead of at chat end. The thread ID comes from the session or the resumed
    thread.
  - The eval: `run_turn` on a fresh thread, with `track_tokens(csv, "eval", thread)`.
    The judge records under the service `eval-judge`.
  - CLI: the chat loop moved into `cli_main.py` (Phase 8.1 moves it into `cli.py`). It
    keeps one thread per run, so messages after the first are follow-ups, and calls
    `input()` through `asyncio.to_thread`.
- **Tokens (7.5):** `track_tokens(csv_path, service, thread_id=None)`. The CSV gained
  two trailing columns, `service` and `thread_id`, so older 5-column rows still line
  up. New files get a header. The existing `data/token_usage.csv` keeps its old
  5-column header.
- **Recommender (7.4):**
  - `offered_item_ids(messages)` reads the IDs from `semantic_search` `ToolMessage`s.
    An answer outside that set (including one given without any search) gets one
    corrective message listing the valid IDs, and is skipped if the second answer is
    invalid too.
  - The per-description agent calls run with `asyncio.gather`.
  - Repeated item IDs are de-duplicated, keeping the first occurrence and the order.
    This was added to the 7.4 spec afterwards.
- **Critique node:** at the limit it returns only `{"critique_text": ""}`, and the
  count is untouched. Rejections still increment the count, and the intake node resets
  it each turn.
- **Tests:** 62 passed, 1 skipped (postgres), 1 deselected (aws). That includes the
  later additions: the prose-answer fallback, `chosen_item_id`, and passing
  `reasoning` through. They pass the same way
  inside `unshare -rn`.
  - The fake LLM is now async-only (a sync call in a node fails) and records every
    prompt.
  - The mock agent calls the real `semantic_search` tool and returns `ToolMessage`s.
    Its answers are scripted per item description.
  - New tests:
    - intake reset
    - the quantifier default
    - the critique prefix (7.7) and the modifier retry count
    - the critique limit and rejection counting
    - `offered_item_ids`
    - the invalid-ID correction, skipping after two invalid IDs, and an answer with no
      search
    - skipping failed lookups and de-duplicating repeats
    - the critique count resetting on a follow-up turn
    - `stream_turn` running to `END` with callbacks reaching the run
    - token CSV columns and header
- **Real runs with Ollama `qwen3.5:4b` (2026-11-26/27), the default local model since
  7.10:** 4B parameters, Q4_K_M, vision, tools, thinking. The setup was local SigLIP,
  Qdrant on a *copy* of `data/qdrant_storage`, real S3 images and the memory
  checkpointer.
  - **Thinking must be off for small models.** With thinking on, the modifier's
    structured output was right 1 time in 4, at about 52 s each; with it off, 4 out of 4
    at 1.3–1.6 s. **New setting: `LLM__REASONING`** (`bool | None`, passed to
    `ChatOllama(reasoning=...)`; None = the model's default; `false` by default since
    7.10).
  - **Small models often skip the structured-response tool** of a tool-calling agent
    and answer in prose. `ToolStrategy` and the default behave the same way;
    `ProviderStrategy` forces JSON, which disables the tools, so the model answers
    without searching. Changes:
    - `chosen_item_id()` takes the first *offered* ID mentioned in the agent's final
      prose message when there is no structured response. Validation still applies.
    - The intake follow-up uses a plain `with_structured_output(UpdatedUserRequest)`
      call instead of `create_agent` (it needs no tools).
    - Pulling the ID out of the whole conversation with a separate structured call was
      tried with a smaller model and rejected: with the images in context it answered
      `0` every time.
  - **Isolated checks** (thinking off): vision correct, and the recommender agent chose
    the right item 8 times out of 8 (about 4 s each; roughly half structured, half
    through the prose fallback).
  - **Two-turn run before the prompt review:** 30 s and 45 s. Every recommender call
    returned a valid ID, with no corrections and no skips. Turn 1 gave khaki chinos and
    white sneakers (pants and shoes, as asked). The follow-up was rewritten correctly
    ("like the khaki chinos, in black") and returned three pairs of black trousers.
  - **Problems found, fixed in 7.11:** the critique judged against the reference item
    instead of the request (it rejected black trousers for not being khaki); after
    three rejections the modifier had dropped the shoes the user asked for; and the
    explanation described "white sneakers" that weren't recommended.
  - **Ollama memory: `LLAMA_ARG_CACHE_RAM=0` is required.** Ollama 0.40 runs this model
    through `llama-server`, whose prompt cache lives in system RAM and defaults to
    8 GiB (`--cache-ram 8192`). Ollama doesn't override it. Each agent call with
    images is a new prompt, so the cache grows even though the model is 100% on the
    GPU (3.7 GB of VRAM at a 16k context). On 2026-11-26 this, with up to 4 parallel
    requests, SigLIP, Qdrant and a Qdrant copy on tmpfs, filled 14 GB of RAM and the
    laptop froze. With `LLAMA_ARG_CACHE_RAM=0`, the runner's real allocations peaked at
    1.0 GB instead of 6 GB, and free RAM stayed above 7.6 GB. The runner inherits
    Ollama's environment, so set it on `ollama serve`. The tested command was:
    `LLAMA_ARG_CACHE_RAM=0 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1
    OLLAMA_CONTEXT_LENGTH=16384 systemd-run --user --scope -p MemoryMax=6G -p
    MemorySwapMax=0 ollama serve`. The memory cap is a safety net: the kernel kills only
    Ollama instead of the machine stalling. Phase 14 should document this.
- **Prompt review (7.11).** Every prompt in `prompts.py` was rewritten for small
  models: short numbered rules, one task each, and the user's request stated as what
  wins over critique feedback and reference images. Changes in behaviour:
  - **Quantifier:** also extracts the item types the user names (`["pants",
    "shoes"]`) into the new state field `requested_item_types`; the number of
    recommendations is at least one per type.
  - **Modifier:** answers `{item_type, description}` pairs (`RequiredClothes.items`);
    the node retries until every requested type is covered (a type also counts as
    covered if the description names it) and keeps the best answer. Descriptions are
    written like catalogue listings, one colour each, using the user's colour word.
    "Something like this" means the main reference item's type.
  - **Critique:** judges against the request, not the reference images; the failures
    are listed (too few items, a named type missing, an explicit constraint broken,
    a clash for "goes with" requests) and matters of taste are not failures. It is
    told the requested item types.
  - **Critique and explanation messages:** images are labelled "reference images"
    (after a follow-up they include earlier recommendations, not only uploads), and
    when there are none the message says so; before, a model took the first
    recommended image for an upload.
  - **Explanation:** speaks to the user ("you"), at most 150 words, mentions only the
    recommended items, no preamble.
  - **Intake follow-up:** keeps the original request's constraints unless the new
    message changes them, and says which recommended item "the first item" is.
  - **Intent/vision:** always describe item type and colours; for "something like
    this", the most prominent item in detail.
  - **Recommender:** steps spelled out (one category call, `num_matches` 3–6, at most
    one extra search, answer with a returned id). `Catalogue.search` clamps
    `num_matches` to 1..6 (`MAX_SEARCH_MATCHES`): each result is an image in the
    model's context and an S3 GET.
  - **Repeated products:** the catalogue stores several photos of one product under
    different ids with the same description, so the recommender now also skips an item
    whose description it has already shown.
  - Tool docstrings and schema field descriptions (which the model also reads) were
    tightened; the judge prompt now says to score visual grounding 0 without uploads,
    as its schema already did.
- **Model-quality tests (7.12):** `tests/model/` runs the real agent (configured
  Ollama model, local SigLIP, Qdrant, S3 catalogue) on 7 conversations (9 turns) whose
  products exist in the catalogue in quantity: 2 black leather belts; a navy blazer,
  white shirt and brown loafers; pants and shoes for an uploaded gingham shirt
  (catalogue item 9067); something similar to uploaded red sneakers (93191); white
  sneakers then "the same style in black"; pants for a shirt then "hats instead"; 3 red
  cocktail dresses.
  - **Grading:** a rubric (`tests/model/rubric.py`) checks each turn against catalogue
    ground truth: the category and description of every recommended product come from
    the metadata, not the model. Checks are weighted; critical ones (count, categories,
    item-type coverage) must pass, and the weighted score must reach 0.8. Colour and
    material words accept catalogue synonyms (calfskin is leather, tobacco is brown).
    The explanation is checked for item types that were neither recommended nor asked
    for, and for speaking to the user. `FRAG_MODEL_TEST_REPEATS=N` runs each case N
    times; a case passes if at least half its runs pass. The rubric has offline unit
    tests (`tests/unit/test_rubric.py`).
  - **Reports:** `data/model_test_runs/<UTC time>/report.json` holds every turn's
    request, products, explanation, check results and the agent's warnings, skips and
    critiques; a summary is printed at the end of the run.
  - **Safety:** skipped unless `LLM__PROVIDER=ollama` (a Bedrock run would be billed),
    and unless Ollama, the Qdrant collection and AWS credentials are available. The
    metadata Parquet and catalogue images are cached under
    `~/.cache/frag-model-tests/`, and a run stops after
    `FRAG_MODEL_TEST_MAX_S3_GETS` (default 300) uncached image downloads. A full run
    takes about 10–12 minutes.
  - **Results so far (2026-11-27, `qwen3.5:4b`, 1 run per case):**
    - Run 1, with the first prompt rewrite: 3 of 7 passed (the follow-up colour and
      item-type switches, and "similar to this"). The failures led to the fixes in
      7.11: a second belt dropped as the same product, a hallucinated "bow tie in your
      first image" when nothing was uploaded, "crimson"/"ruby" instead of "red", and
      pants-only answers to "pants and shoes" (which the critique accepted). Some
      failures were the rubric's (calfskin not counted as leather, loafers filed
      under FLATS), and were fixed there.
    - Run 3, after the item-type coverage: belts and pants-and-shoes passed; the
      wedding outfit failed on the rubric (the white shirt is filed under TOPS, fixed)
      and on beige rather than brown loafers; "similar to this" failed because the
      test image (a two-tone dress worn with sneakers) was ambiguous, so it was
      replaced with a product shot and the prompts now focus on the main item. The run
      stopped at its S3 budget before the last three cases.
    - A full run after these last changes is still to be done.
    - Memory stayed healthy throughout: free RAM at least 6.7 GB, no swap, the
      Ollama runner at most 1.3 GB.
    - Qdrant needs no memory cap: with a 2 GiB container limit its page cache was
      starved and filtered searches timed out (the client's 5 s limit).
- **Ruff:** the new and changed modules are clean. The total went from 293 to 204, the
  rest being in modules later phases rewrite (`app.py`, `utils/common_utils.py`, ...).
  The README's async TODO was removed.
- **AWS call log:** none for the Phase 7 code. All §0.2-B reads:
  - The early Ollama runs read the metadata Parquet (one 5.9 MB GET per process) and
    catalogue images from S3.
  - 2026-11-27: one Parquet GET into the test cache (to choose ground-truth items),
    two single-image GETs to inspect test images, and about 400 uncached image GETs
    across the model-quality runs (well under $0.001). No DynamoDB (memory
    checkpointer) and no Bedrock.

---

## Phase 8: Entrypoints

- [x] **8.1 `src/frag/cli.py`** (argparse), replacing `cli_main.py` and
  `mcp_server.py`. The README's Hydra-style `eval.eval_mode=true` overrides **don't
  work today**: nothing parses `argv`.
  - `frag chat` is the terminal REPL, moved out of `rag_agent.py`. Validate that image
    paths exist before sending them.
  - `frag eval [--skip-run] [--skip-judge] [--limit N]` uses the `memory` checkpointer
    (§3.3).
  - `frag mcp`. Remove the check that raises when `USE_MCP_SERVER` is false; the server
    shouldn't care about the client's setting.
  - Delete `cli_main.py` and `mcp_server.py`, and update the README.
- [x] **8.2 `app.py`** (Cognito and the DynamoDB+S3 history are always on, per
  decision 1.5):
  - Move the two monkeypatches into `frag/ui/chainlit_patches.py` as an idempotent
    `apply()`, and note that they target Chainlit 2.10.0. Remove the duplicate
    `S3StorageClient` import. Check whether newer Chainlit releases fix
    chainlit#2707/#972 and the async `close()` bug, and report.
  - In `@cl.on_app_startup`, run the §3.2 sequence with `role="agent"` and keep the
    `AsyncExitStack` in a module global. Close it in `@cl.on_app_shutdown`.
  - In `get_data_layer`, use `get_settings().chat_history` and
    `get_settings().aws_region`. `get_data_layer` is called by Chainlit before startup,
    which is fine because settings are pure.
  - Merge the duplicated `start_chat` and `chat_resume` bodies into
    `_init_session(thread_id)`. After Phase 7.6, neither needs a priming invoke.
  - Remove the unused imports (`chainlit.data as cl_data`, `Command`).
- [x] **8.3 `compose.yaml`**:
  - The `qdrant` storage mount becomes `${QDRANT_STORAGE_DIR:-./data/qdrant_storage}`,
    so the SigLIP storage is the default, and the snapshot mount gets the same
    treatment.
  - Fix the comment about a "Containerfile" that doesn't exist. Pin the image by
    digest, or say it's pinned by tag.
  - Add the `postgres` profile service (Phase 4.5).
- [x] **8.4 Consolidate the two local Qdrant storage directories**, done locally:
  - `data/qdrant_storage/` contains `fashion_gen` (SigLIP) **and** a
    `nova_fashion_gen`.
  - `data/nova_qdrant_storage/` contains another `nova_fashion_gen` (about 10k points,
    per the plan).
  - Start Qdrant on each directory in turn and record the point counts with
    `scripts/qdrant_admin.py info`.
  - If the copy in `nova_qdrant_storage` is the more complete one, snapshot it and
    restore it into `qdrant_storage` as `nova_fashion_gen`, replacing the stale copy.
    That needs a local snapshot and restore only, and **asking the user before
    replacing a collection**. One Qdrant instance then serves both backends.
  - Report the counts either way. Don't delete either directory.

**Phase 8 done (2026-11-27).** Notes:
- **CLI (8.1):** `src/frag/cli.py` provides `frag chat | eval | mcp` (`[project.scripts]`,
  so `uv run frag ...`). `cli_main.py` and `mcp_server.py` are deleted, and the README
  run steps are rewritten for it (Phase 14 rewrites the rest).
  - `frag chat` reads the text and then image paths one per line (empty line to
    finish); a path that doesn't exist is reported and skipped before the agent runs.
    One thread per run, so later messages are follow-ups.
  - `frag eval [--skip-run] [--skip-judge] [--limit N]`: `run_full_evaluation_pipeline`
    gained `run`, `judge` and `limit` arguments. Role `eval` (memory checkpointer).
  - `frag mcp`: the `USE_MCP_SERVER` check was already gone.
  - Each command follows §3.2; a failure is logged and exits with status 1 (before,
    `cli_main.py` logged and exited 0). Ctrl+C exits quietly.
  - Smoke-tested: `frag mcp` served `/health` (`{"status":"ok","items":260490}`) and
    shut down cleanly on Ctrl+C.
- **`app.py` (8.2):**
  - The patches are in `frag/ui/chainlit_patches.py` as an idempotent `apply()`,
    called at import. The duplicate `S3StorageClient` import and the unused imports
    were already gone; `get_data_layer` already used the settings; the resources were
    already in a module-level `AsyncExitStack`.
  - `start_chat` and `chat_resume` share `_init_session(thread_id)`; neither primes
    the graph. Handlers are typed and documented.
  - **Changed:** an agent failure shows the user a generic message instead of the
    exception text (which could leak internals); the exception is logged.
  - **Chainlit upgraded 2.10.0 -> 2.12.0** (`chainlit>=2.12.0`). 2.10.1 fixes session
    ownership on websocket restore (the app resumes threads); 2.12.0 fixes two
    critical `/mcp` CVEs (CVE-2026-45018/45019), which only apply with
    `features.mcp.enabled = true` (it's `false` here) and has breaking changes only
    for MCP. **Both bugs the patches work around are still present in 2.12.0**
    (`secret.chars` still has punctuation; `S3StorageClient.close` still awaits
    boto3's synchronous `close`), so the patches stay, re-checked against 2.12.0.
    Chainlit 2.12 now warns about the legacy `[features.mcp.*]` sections in the
    local, git-ignored `.chainlit/config.toml`; harmless while MCP is off. Remove them
    when a production config is committed (plan Phase 2). The `traceloop` pydantic
    deprecation warning Chainlit now triggers is filtered in pytest.
  - Smoke-tested: `chainlit run app.py --headless` opened the resources (catalogue,
    SigLIP, the DynamoDB checkpointer from `.env`), served `/` and `/auth/config`
    (200), and closed cleanly with the patched S3 close. The sign-in and chat flow
    was not exercised (it needs Cognito in a browser).
- **`compose.yaml` (8.3):** Qdrant mounts `${QDRANT_STORAGE_DIR:-./data/qdrant_storage}`
  and `${QDRANT_SNAPSHOTS_DIR:-./data/qdrant_snapshots}` (empty directory created).
  The container is `frag-qdrant`; the image is pinned by tag (`v1.17`). The
  `postgres` profile was already there (Phase 4.5).
- **Qdrant consolidation (8.4):** counted on copies of both directories:
  - `data/qdrant_storage`: only `fashion_gen` (SigLIP), **260,490** points, 768-d
    `images` (Dot) and `texts` (Cosine). There was no stale `nova_fashion_gen` in it,
    contrary to the step's premise.
  - `data/nova_qdrant_storage`: `nova_fashion_gen`, **10,000** points, 1024-d
    `images` and `texts` (Cosine).
  - Nothing needed replacing, so `nova_fashion_gen` was snapshotted from a copy and
    restored into `data/qdrant_storage` (additive; 1.8 GB -> 2.7 GB). Both
    collections were green with the counts above afterwards, and `podman compose up`
    serves both. Neither original directory was deleted; `data/nova_qdrant_storage`
    and `data/nova_qdrant_snapshots` are now redundant (removal candidates for the
    user). The snapshot file is in `~/.cache/frag-snapshots/` (931 MB) and a pre-change
    copy of `data/qdrant_storage` is in `~/.cache/frag-qdrant-copy`; both can be
    deleted.
- **Tests:** 85 passed, 1 skipped (postgres). New `tests/unit/test_entrypoints.py`:
  CLI argument parsing, image path validation and quitting in `frag chat`, the eval
  flags, and the Chainlit patches (URL-safe state characters, a non-awaiting S3 close,
  idempotence).
- **Ruff:** 203 -> 179; the touched files are clean.
- **AWS call log:** no direct calls. The smoke tests read the metadata Parquet (one
  GET per process, two processes) and opened the DynamoDB checkpointer (§0.2-B).

---

## Phase 9: Evaluation (`evaluation/run.py`)

- [x] **9.1** Apply fixes 3.2–3.5 if they haven't been done.
- [x] **9.2 Count failures.** 29 of the 60 stored runs have no recommendations, and the
  judge skips them, so the README's "9.37 ± 0.78" covers only 31 runs. Record a
  `status` and `error` for each run and each judgement. The summary prints
  `completed/total`, mean ± std over the completed runs, and a per-group breakdown.
- [x] **9.3 Self-contained runs.** Write to `data/evaluation_runs/<UTC timestamp>/`,
  containing:
  - `config.json`: the **agent LLM provider and model ID**, the **judge LLM provider
    and model ID**, the temperature, the embedding backend and model, the collection
    and its point count, and the git commit hash, because results from different
    models aren't comparable. Also repeat `agent_model` and `judge_model` on every
    line of `results.jsonl` and `judgement.jsonl`, so lines stay self-describing if
    files get merged. The summary prints the models next to the score.
  - `results.jsonl`, appended as it goes
  - `images/`, holding copies of the recommended images
  - `judgement.jsonl`

  Paths are relative to the run directory. This removes the read-modify-write of the
  whole JSON on every result, the absolute `/mnt/windows/...` paths into
  `temp_images/`, and the no-op `absolute_image_paths` loop. Leave the historical
  `eval_soln.json`/`eval_judgement.json` alone.
- [x] **9.4** Judge with `ainvoke`. Record judge errors instead of `continue`-ing
  silently.
- [x] **9.5** Remove `asyncio.sleep(5)`; the rate limiter throttles. Add a `--sleep`
  flag if it's needed.
- [x] **9.6** `get_evaluation_inputs(queries_path, images_dir)` builds paths with
  `Path`. Merge the two near-identical dict literals.
- [x] **9.7** A full eval against Bedrock costs money, so it needs the user's approval
  (§0.2C). The plan estimates $3–5. Test the pipeline with `--limit 2` on Ollama.

**Phase 9 done (2026-11-27).** Notes:
- **9.1:** fixes 3.2–3.5 were already done (Phases 3, 4 and 7.5).
- **Module:** `evaluation/llm_as_judge.py` is replaced by `evaluation/run.py`;
  `frag eval` calls `evaluate(resources, run=, judge=, limit=, sleep_s=, run_dir=)`.
  `EvalSettings.eval_response_file_path` and `judgement_file_path` are replaced by
  `runs_dir` (default `data/evaluation_runs`, git-ignored). The historical
  `eval_soln.json` and `eval_judgement.json` are untouched.
- **Run directory (9.3):** `data/evaluation_runs/<UTC time>/` holds `config.json`
  (agent and judge model, temperature, reasoning, embedding backend and model,
  collection and its exact point count, tools mode, limit, weights, and the git
  commit with `-dirty` when the tree has uncommitted changes), `results.jsonl` and
  `judgement.jsonl` (appended line by line; every line carries `agent_model`, and
  judgements also `judge_model`), `summary.json`, and `images/` with copies of the
  input and recommended images. All paths are relative to the run directory.
  - The judge is the same LLM as the agent (there is no separate judge setting).
    Judging a stored run later (`--skip-run`) records the judging model in
    `config.json`.
  - A new run refuses to write into a directory that already has results.
- **Failures (9.2, 9.4):** every query gets a results line with `status` `ok`,
  `no_recommendations` or `error` (with the exception), plus its time and token usage.
  Every query gets a judgement line with `status` `ok`, `skipped` (with the run
  status) or `error`. The judge uses `ainvoke`. The summary reports
  `completed/total`, the run statuses, `judged` and judge errors, mean ± std over the
  judged queries, and the same per group; it's logged with the models and written to
  `summary.json`.
- **9.5:** the `asyncio.sleep(5)` is gone; `frag eval --sleep S` waits between
  queries if a provider ever needs it.
- **9.6:** `get_evaluation_inputs(queries_path, images_dir)` uses `Path` and builds
  one dict per query.
- **9.7, real run:** `frag eval --limit 2` on Ollama `qwen3.5:4b` (local SigLIP,
  `fashion_gen`, S3 images): both queries completed (100 s and 50 s, 2 items each:
  trousers and oxfords; cargo pants and sneakers for "one pant and one pair of
  shoes"), both judged (7.8 and 9.7; 8.75 ± 0.95). The agent used about 18–38k input
  and 3–5k output tokens per query, which on Haiku 4.5 would be about $0.03–0.06 per
  query. **The full 60-query eval on Bedrock was not run**: it needs the user's
  approval (§0.2C, about $3–5 plus judging).
- **Tests:** 87 passed, 1 skipped. `tests/integration/test_evaluation.py` runs the
  pipeline offline with the fakes: a self-contained run that gets judged (relative
  image paths, scores with and without an uploaded image), agent failures counted as
  `error` and judged as `skipped`, judge errors recorded, judging the latest run with
  `--skip-run`, and refusing to overwrite a run. The fake LLM answers
  `RecommendationEvaluation`.
- **Ruff:** the touched files are clean; total 179.
- **AWS call log:** no direct calls. The real run read the metadata Parquet and the
  catalogue images its two queries returned from S3 (§0.2-B); no Bedrock.

---

## Phase 10: Scripts (the data pipeline lives outside the app package)

Every script follows §3.2: `load_dotenv`, `get_settings`, `setup_logging`, argparse,
and an interactive confirmation before anything destructive or billable. Scripts that
call Bedrock or write to S3 print a cost estimate and require `--yes` or typing `YES`
(§0.2C).

- [ ] **10.1 `scripts/ingest_fashiongen.py`**: restore the HDF5 → PNG + Parquet → S3
  pipeline from git history
  (`git show 54890dd^:src/frag/exploration/dataset_wrangling/split_hdf5.py`). It was
  deleted in commit 54890dd, but it's the only way to rebuild the catalogue
  reproducibly (decision 1.8).
  - Arguments: HDF5 path, a temporary output directory, `--s3-bucket` (default from
    settings), `--ids-file`/`--limit`, workers, and Parquet compression (zstd).
  - Uploads skip existing keys (`head_object`), so it can be re-run without
    re-uploading.
  - Needs the `ingest` group. **Don't run it**; the data is already on S3.
- [ ] **10.2 `scripts/build_vector_db.py`**, ported from
  `utils/aws_create_vector_db_local.py` as a **reproducible record of how the Nova
  collection was built** (decision 1.12). It isn't run, so keep it small:
  - Replace the module globals with argparse: `--collection`, `--start/--end` or
    `--ids-file`, `--recreate` (keep the `YES` prompt), `--qdrant-url`, and
    `--batch-size`. Defaults come from settings.
  - Use `catalogue.store.load_metadata` (S3 Parquet) instead of the local
    `/mnt/windows/...` Parquet path.
  - **Remove the hard-coded AWS account ID** (`aws_create_vector_db_local.py:61`); get
    it at runtime from `sts get-caller-identity`.
  - Keep its own small Nova image and text request bodies; don't add SigLIP or image
    support to the app's embedders for this.
  - Print an estimated cost and require confirmation before any Bedrock call (§0.2C).
  - Replace the 12 "Random docstring." placeholders with real docstrings.
  - Note in the module docstring that the SigLIP collection was built by the deleted
    `vector_db_writer.py` (commit `2583639^`), and say where to recover it from.

- [ ] **10.3 `scripts/qdrant_admin.py`** with these subcommands:
  - `info`, from `explore_nova_qdrant.py`
  - `sample`, from `check_collection_count`
  - `snapshot`, from `create_nova_qdrant_snapshot.py`
  - `restore`, taking a snapshot path argument instead of the hard-coded nonsense path
  - `copy-subset --from X --to Y --ids-file F`, which copies points with their vectors
    and payloads without re-embedding. The AWS catalogue now uses the full ~10k Nova
    collection (decision 1.4), so this is a utility rather than a required step; it's
    local only.

  The collection comes from settings or `--collection`, with no hard-coded
  `"nova_fashion_gen"`.
- [ ] **10.4** Delete `src/frag/utils/aws_create_vector_db_local.py` once its
  replacements exist.

---

## Phase 11: Small utilities and logging

- [ ] **11.1 `messages.py`**: replace the three message builders (one with bug 3.1)
  with `build_message(parts: list[str | Path]) -> list[HumanMessage]`, using
  `mimetypes.guess_type`. Delete the unused `numpy_image` branch.
- [ ] **11.2** Delete `get_qdrant_points_by_id` and `save_numpy_image_to_folder`
  (both unused). Rename `save_image_url_to_folder` to
  `save_image_b64(folder, b64) -> str`.
- [ ] **11.3 `logging_setup.py`**: cut it down from 216 lines, following §3.2.
  - **Development:** the Rich console sink, plus an optional file sink at
    `{file_dir}/frag-{service}.log`. That replaces the four `*_log_file` settings.
  - **Production:** `serialize=True` JSON on stdout only.
  - Keep `InterceptHandler` and the library level map, which moves into
    `LogSettings.library_levels` as the TODO asks.
  - **Keep the Rich console sink** (decision 1.14) and declare `rich` as a direct
    dependency. Use it in development; production uses JSON on stdout. Delete the
    commented-out `_serialize`, the unused `traceback`/`json` imports, and the
    contradictory `basicConfig(level=0)` followed by `getLogger().setLevel(WARNING)`.
  - Idempotent: store the added handler IDs and remove only those on re-entry.
- [ ] **11.4 `ui/node_updates.py`**: just the move. Replace the
  `lambda u: f(u)` wrappers with `f`.

---

## Phase 12: Verify feasibility of the local SigLIP + remote LLM mode

These checks are allowed under §0.2B, plus local Ollama calls.

- [ ] **12.1** Start Qdrant on `data/qdrant_storage`, then run
  `scripts/qdrant_admin.py info --collection fashion_gen`. Record the point count, the
  vector names and sizes (expect `images`/`texts` at 768), and confirm that the payload
  has `index_2` and `input_category`.
- [ ] **12.2** `scripts/qdrant_admin.py sample`: for 3 random points, check that
  `point.id == payload.index_2` and that `head-object` on the derived S3 image key
  exists. That's three HEAD requests.
- [ ] **12.3** Run `frag chat` with `EMBEDDING__BACKEND=siglip`,
  `TOOLS__MODE=local`, `LLM__PROVIDER=ollama` and
  `CHECKPOINTER__BACKEND=sqlite`, and send one text-only and one image query. That's
  free-tier or local LLM calls and a few dozen S3 GETs.
- [ ] **12.4** Repeat 12.3 with `TOOLS__MODE=mcp`, running `frag mcp` in another
  terminal.
- [ ] **12.5** Report the results. If 12.1–12.2 show the SigLIP collection is
  incompatible (wrong IDs or a missing payload), stop and ask the user. The user said
  it can be deleted in that case, but **don't delete it yourself**.

---

## Phase 13: AWS resources as code (decision 1.8)

The goal is that every AWS resource the project uses is owned by a committed template
or script. This phase **writes code and reads AWS state only**. Deploying or importing
anything needs the user's go-ahead (§0.2C).

- [ ] **13.1 Inventory (read-only, §0.2A).** List what exists and write
  `docs/aws_inventory.md` with one row per resource: type, name or ID, how it was
  created (CloudFormation stack, or "manual"), the code that should own it, and its
  status (keep / codify / removal candidate). Check at least:
  - S3: `frag-fashion-gen-dataset` (keep: 260k images plus Parquet, needed for local
    SigLIP mode), `frag-fashion-gen-embeddings` (removal candidate: batch-inference
    leftovers), `frag-chainlit-elements-bucket`, `frag-cloudformation-templates-bucket`,
    `my-personal-bucket-callingmicron` (exploration; removal candidate)
  - DynamoDB: the checkpointer table and the Chainlit table
  - Cognito: the user pool, app client and domain
  - IAM: `BedrockBatchInferenceRole` and its inline policy `BedrockBatchS3Access`
    (removal candidates), plus any other `frag`/project roles
  - CloudFormation stacks
  - Bedrock model access for Claude Haiku 4.5 and Nova embeddings (list-only)
- [ ] **13.2 Templates** in `infra/cloudformation/`. Keep this minimal; the plan's
  Phase 3 adds network, compute, edge and ops on top.
  - `storage.yaml`: the catalogue bucket (`DeletionPolicy: Retain`), the Chainlit
    elements bucket, the checkpointer table and the Chainlit table (merge the two
    existing templates). Keep CORS as is for now; the plan tightens it.
  - `auth.yaml`: the Cognito user pool, domain and app client. Use the
    authorization-code flow, the scopes `openid email profile`, and callback URLs as a
    parameter list. The default is `http://localhost:8000/auth/oauth/aws-cognito/callback`
    for local use; the plan adds the CloudFront URL later. **Sign-up is admin-only**
    (`AdminCreateUserConfig.AllowAdminCreateUserOnly: true`, decision 1.13). Add
    `infra/scripts/create_user.sh <email>`, which wraps `admin-create-user` and asks for
    confirmation, so creating users is scripted as well. Output the client ID and
    domain.
  - `budget.yaml`: an AWS Budget with email alerts at 50/80/100% of a user-chosen
    monthly limit. The first two budgets are free. Add this early as a guardrail.
  - `root.yaml`: nests the above, takes `EnvName` and the resource names as
    parameters, and passes them through. Child `TemplateURL`s are relative paths for
    `aws cloudformation package`, replacing the hard-coded S3 URLs in
    `root_template.yaml`.
  - Run `cfn-lint` on everything (local). `aws cloudformation validate-template` is
    allowed under §0.2A.
- [ ] **13.3 Bring existing resources under CloudFormation.** For resources that exist
  but are manual (very likely the dataset bucket and the Cognito pool), write
  `infra/scripts/import_existing.sh`. It creates an `IMPORT` change set with the
  identifiers from the inventory and `DeletionPolicy: Retain` on every imported
  resource, then **stops and prints the change set for review**. Executing it needs
  the user (§0.2C). Resources already in a stack stay where they are, or move with the
  same import technique; describe the options in `docs/aws_inventory.md` and let the
  user choose.
- [ ] **13.4 `infra/scripts/deploy.sh`** runs
  `aws cloudformation package` (to the templates bucket) and then
  `aws cloudformation deploy --no-execute-changeset`, prints the change set, and only
  executes it after a typed `YES`. Add **`infra/scripts/teardown.sh`**, which
  refuses to touch `Retain` resources and asks for confirmation. Both scripts are
  idempotent.
- [ ] **13.5 `infra/scripts/write_env_from_stack.sh`** reads the stack outputs (table
  names, bucket names, Cognito client ID and domain) and prints the matching `.env`
  lines. The Cognito client secret is retrieved with `describe-user-pool-client`; this
  script is the only place allowed to read it, and it writes only to `.env`, never to
  stdout. This replaces hand-copying values.
- [ ] **13.6** Data uploads are only ever done by `scripts/ingest_fashiongen.py`
  (catalogue) and the Qdrant snapshot upload that the plan adds. Document that in the
  README.

---

## Phase 14: Documentation

- [ ] **14.1 `.env.example`**: rewrite it to match `Settings` plus the third-party
  variables, grouped as **local development** and **AWS**.
  - Development preset: `LLM__PROVIDER=ollama` with a pulled vision model,
    `EMBEDDING__BACKEND=siglip` and `CHECKPOINTER__BACKEND=sqlite`.
  - AWS preset: `LLM__PROVIDER=bedrock`, `EMBEDDING__BACKEND=nova` and
    `CHECKPOINTER__BACKEND=dynamodb`.
  - Remove `HYDRA_FULL_ERROR`, `MISTRAL_API_KEY`, `POSTGRES_*`,
    `DYNAMODB_ENDPOINT_URL`, `AWS_ACCOUNT_ID`, `AWS_BEDROCK_BATCH_INFERENCE_ROLE_ARN`,
    and the misleading `AWS_ACCESS_KEY_ID=local`/`AWS_SECRET_ACCESS_KEY=local`.
  - Recommend `aws configure sso` or a named profile instead of static keys in `.env`.
- [ ] **14.2 `README.md`**:
  - **Features.** Describe both embedding backends (Marqo-FashionSigLIP locally, Nova
    on AWS), all four checkpointers, Bedrock and Ollama, and per-turn graph runs.
    Replace the eval score with "pending re-run", and report the completion rate once
    Phase 9 is re-run.
  - **Install and run.**
    - `uv sync --group siglip`, keeping the Chainlit `--link-mode copy` workaround
      (decision 1.2) and adding the symlink-mode note from Phase 2.2.
    - `podman compose up -d` (plus `--profile postgres`).
    - `infra/scripts/write_env_from_stack.sh`.
    - `frag mcp`, `chainlit run app.py`, `frag chat`, `frag eval`, `pytest`.
  - Fix the `mkdir -p data/qdrant_storate/` typo.
  - Remove the "AWS Deployment Considerations" and "Changes to MCP Server" sections and
    link to `docs/` instead. Remove both TODOs.
  - **Contribution guidelines.** Keep them, but change "create branches from
    develop" to `main`, since PRs merge into `main` (decision 1.15). Remove the line
    about the `exploration` folder getting lighter review, since that folder is gone.
- [ ] **14.3** Update `docs/frag_aws_completion_plan.md` wherever it conflicts with
  §1:
  - Keep the full catalogue bucket (local SigLIP mode).
  - Delete the batch-inference leftovers instead of the catalogue.
  - Add Cognito to `auth.yaml`, which already matches.
  - Embedding is per-item `InvokeModel` only, and no new embeddings are needed
    (decision 1.12).
  - Make the image build use `--no-group siglip`.
- [ ] **14.4** List the removal candidates for the user. Don't act on them.
  - Local: `data/jsonl_files/`, the root `__pycache__/`, and `data/temp_images/`
    (which `eval_soln.json` references).
  - AWS: from the inventory in 13.1.
- [ ] **14.5** Delete `cleanup.md` when done.

---

## Phase 15: Final checks

- [ ] `uv run ruff check` is clean, and every ignore has a justification comment.
- [ ] The empty-environment import check from §0.5 passes.
- [ ] `uv run pytest` passes offline.
- [ ] `grep -rn "/mnt/windows\|650539477368\|260_490\|dependency_injector" --include=*.py .`
  returns nothing outside `.venv`.
- [ ] Final report to the user covering:
  - what changed
  - skipped steps and why
  - findings that aren't in this document
  - the AWS call log (§0.2E)
  - the ruff baseline compared with the final state, and every added ignore
  - the Ollama image-in-tool-result result (Phase 4.2)
  - the Qdrant consolidation result (Phase 8.4)
  - the remaining manual tests that need Bedrock (§0.2C), for the user to approve:
    1. Bedrock LLM + SigLIP: one chat turn.
    2. Bedrock LLM + Nova + `nova_fashion_gen`: one chat turn.
    3. Chainlit with Cognito login: text query, image query, a follow-up with a new
       image (bug 3.1), and resuming a thread from history.
    4. `frag eval --limit 5` on Bedrock, then the full eval.

---

## Phase 16: GitHub Actions CI (decision 1.15)

- [ ] Add `.github/workflows/04-ci.yml`, triggered on `pull_request` to `main` and on
  `push` to `main`.
  - Steps: `astral-sh/setup-uv` (pinned version), then `uv sync --locked` without the
    `siglip` and `ingest` groups, then `uv run ruff check`, then `uv run pytest`
    (offline; the `aws` and `bedrock` markers are deselected by default). Add a
    `cfn-lint infra/cloudformation/**/*.yaml` job.
  - **No AWS credentials, no OIDC role, no secrets.** The plan's Phase 4 adds a deploy
    job later, with the user's approval.
  - Public repositories get standard GitHub-hosted runners for free; private ones get
    a free monthly minute allowance. Mention which applies in the report.
- [ ] Don't modify `01-branch-naming.yml`, `02-commit-message-check.yml`,
  `03-pr-title-enforcer.yml` or `.github/hooks/`.

---

## Appendix: file-by-file disposition

| Current file | Action |
|---|---|
| `app.py` | Keep; slim down (8.2) |
| `cli_main.py`, `mcp_server.py` | Replace with `src/frag/cli.py` (8.1) |
| `compose.yaml` | Configurable storage, postgres profile (8.3) |
| `cloudformation/*` | Move to `infra/cloudformation/`, then restructure (1.6, 13) |
| `src/frag/config/*` (except prompts and exploration) | Merge into `settings.py` (4.1) |
| `src/frag/config/prompts.py` | Becomes `src/frag/prompts.py` (4.1) |
| `src/frag/config/exploration.py` | Delete (1.2) |
| `src/frag/config/container.py` | Delete; replaced by `runtime.py` (4.3) |
| `src/frag/data_manager/*` | Becomes `catalogue/` (6) |
| `src/frag/evaluation/llm_as_judge.py` | Becomes `evaluation/run.py` (9) |
| `src/frag/exploration/**` | Delete after salvaging (1.1) |
| `src/frag/mcp_server/server.py` | `build_server()` (6.7) |
| `src/frag/rag_pipeline/rag_agent.py` | Becomes `agent/graph.py`; the REPL moves to `cli.py` (7, 8.1) |
| `src/frag/rag_pipeline/llm_schemas.py` | Splits into `agent/state.py`, `agent/schemas.py`, `evaluation/schemas.py` (7.2, 7.3) |
| `src/frag/rag_pipeline/local_tools.py` | Becomes `tools/local.py` (6.6) |
| `src/frag/utils/llm_tools_setup.py` | Becomes `tools/base.py` + `tools/mcp_client.py` (6.6) |
| `src/frag/utils/model_factory.py` | Becomes `llm.py`: Bedrock + Ollama (4.2) |
| `src/frag/utils/checkpointer.py` | Becomes `checkpointer.py`: memory/sqlite/postgres/dynamodb (4.5) |
| `src/frag/utils/common_utils.py` | Becomes `messages.py` + `tokens.py`; dead code deleted (7.5, 11) |
| `src/frag/utils/logger_setup.py` | Becomes `logging_setup.py` (11.3) |
| `src/frag/utils/ui_node_updates.py` | Becomes `ui/node_updates.py` (11.4) |
| `src/frag/utils/aws_utils.py` | Delete (1.3) |
| `src/frag/utils/aws_generate_embeddings.py` | Delete (1.5) |
| `src/frag/utils/aws_create_vector_db_local.py` | Becomes `scripts/build_vector_db.py` (record only) + `scripts/qdrant_admin.py` (10) |
| `src/frag/utils/userdata.sh` | Delete (1.4) |
| git history: `split_hdf5.py` | Restore as `scripts/ingest_fashiongen.py` (10.1) |
| git history: `FashionSigLIPEmbedding` | Restore into `catalogue/embedders.py` (6.3) |
| `tests/**` | Rewrite offline (5) |
| `.env.example`, `README.md`, `.gitignore`, `pyproject.toml`, `ruff.toml` | Update (1.7–1.9, 2, 14) |
