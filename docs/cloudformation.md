# AWS resources as code

Every AWS resource FRAG uses belongs to a CloudFormation stack built from
`infra/cloudformation/`, and is created or changed only by the scripts in
`infra/scripts/`. Don't use the console or one-off CLI commands. Account and resource
IDs are left out on purpose (the repo is public); the scripts read them from the stack
outputs.

## What exists (us-east-1)

| Stack | Resource | Type | Purpose |
|---|---|---|---|
| `frag-dev-storage` | `frag-fashion-gen-dataset` | S3 bucket | Catalogue: 260,490 PNGs (`images/{idx//1000:03d}/{idx}.png`) and `metadata/fashion_gen_metadata.parquet`, ~7.6 GB. Filled by `scripts/ingest_fashiongen.py`; read in both embedding modes. |
| `frag-dev-storage` | `frag-chainlit-elements-bucket` | S3 bucket | Chainlit chat-history elements (uploaded and recommended images). |
| `frag-dev-storage` | `langgraph-checkpoints` | DynamoDB table (on-demand, TTL `ttl`) | LangGraph checkpointer when `CHECKPOINTER__BACKEND=dynamodb`. |
| `frag-dev-storage` | `chainlit_users` | DynamoDB table (on-demand, GSI `UserThread`) | Chainlit users, threads and steps. |
| `frag-dev-auth` | FRAG user pool | Cognito user pool (Essentials, deletion protection on) | Login for the Chainlit UI. Sign-up is admin-only; `create_user.sh` creates users, with their email as the username. |
| `frag-dev-auth` | hosted-UI domain | Cognito domain (managed login v2) | `<prefix>.auth.us-east-1.amazoncognito.com` |
| `frag-dev-auth` | `fashion-mas` | Cognito app client | Authorization-code flow with a client secret, scopes `openid email profile`, callback `http://localhost:8000/auth/oauth/aws-cognito/callback`. |
| `frag-dev-auth` | branding | Cognito managed-login branding | The default login page (required by managed login v2). |

Every resource is `DeletionPolicy: Retain`, so even deleting a stack leaves the data in
place. **Bedrock** has nothing to manage: model access (Claude Haiku 4.5 or Nova Lite
through the `us.` inference profiles, and Nova 2 Multimodal Embeddings) is enabled once
per account in the console.

**Cost:** about $0.18/month, nearly all of it S3 storage for the catalogue. DynamoDB on
demand costs about $0.000001 per request, and Cognito is free up to 10,000 monthly
active users. Bedrock is billed per call: about $0.035 per chat message with Claude Haiku
4.5, and roughly 15× less with Nova Lite.

## Layout

| Path | Contents |
|---|---|
| `infra/cloudformation/storage.yaml` | catalogue and Chainlit elements buckets; checkpointer and Chainlit tables |
| `infra/cloudformation/auth.yaml` | Cognito user pool, hosted-UI domain, app client, login branding |
| `infra/scripts/` | the only supported way to touch AWS resources (below) |

Each layer is its own top-level stack, `frag-$FRAG_ENV-<layer>` (`FRAG_ENV` defaults to
`dev`), so teardown is per layer: removing a future compute layer never touches the
catalogue or the user pool. Later layers follow the same pattern and share values
through `Export`/`!ImportValue`. The region is `AWS_REGION` (default `us-east-1`).
Credentials come from the normal AWS CLI chain (a profile, SSO or environment
variables); no script reads or prints them.

## Scripts

Every script that changes something prints what it will do (a change set, or the list
of resources) and continues only after a typed confirmation. All are safe to re-run.

| Script | What it does |
|---|---|
| `deploy.sh [layer] [Key=Value ...] [--dry-run]` | Sets up or updates the stacks through change sets: all layers in order, or one layer. Parameters you don't pass keep their current values. `--dry-run` prints the change set and discards it. |
| `write_env_from_stack.sh [--write]` | Prints the `.env` lines from the stack outputs. `--write` merges them, plus the Cognito client secret (never printed), into `.env`, keeping a `.env.bak`. |
| `create_user.sh <email>` | Admin-creates a Cognito user. Cognito emails them a temporary password. |
| `teardown.sh <layer>` | Deletes a layer stack and its resources. Refuses any stack that contains a Retain resource, which today means both layers; it's meant for future stateless layers such as compute. |

## Setting it up in a fresh account

```bash
infra/scripts/deploy.sh storage          # pass CatalogueBucketName=/ElementsBucketName= if the names are taken
infra/scripts/deploy.sh auth DomainPrefix=<globally-unique-prefix>
uv run python scripts/ingest_fashiongen.py ...   # fill the catalogue bucket
infra/scripts/write_env_from_stack.sh --write    # .env from the stack outputs
infra/scripts/create_user.sh <email>
```

After that, `infra/scripts/deploy.sh` with no arguments updates every layer, and does
nothing if the templates haven't changed.

## Checks

- `uv run cfn-lint infra/cloudformation/*.yaml` (local; also run in CI).
- `infra/scripts/deploy.sh --dry-run` shows what a deploy would change without changing it.
- Drift: `aws cloudformation detect-stack-drift --stack-name frag-dev-auth` (free), then
  `describe-stack-resource-drifts`.

## Notes

1. Chainlit's Cognito OAuth needs a monkeypatch (`src/frag/ui/chainlit_patches.py`):
   Chainlit's `state` parameter contains characters Cognito rejects.
2. Chainlit's DynamoDB data layer doesn't create its table; `storage.yaml` defines it
   with the schema Chainlit expects (PK/SK + the `UserThread` GSI).
3. Run `uv run chainlit create-secret` once and put the value in `.env` as
   `CHAINLIT_AUTH_SECRET`.
4. Changing the Cognito pool's sign-in attributes, the client's `GenerateSecret`, or the
   domain prefix replaces that resource. `deploy.sh` shows `Replacement: True` in the
   change set; don't execute that casually.
