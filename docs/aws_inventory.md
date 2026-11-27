# AWS inventory (us-east-1)

Everything FRAG uses in AWS, and the code that owns it. Each resource belongs to a
CloudFormation stack built from `infra/cloudformation/` by `infra/scripts/deploy.sh`.
There is nothing else in the account: no other buckets, tables, roles, parameters,
functions or instances. Account and resource IDs are left out on purpose (the repo
is public); the scripts read them from the stack outputs.

## Resources

| Stack | Resource | Type | Purpose | Deletion |
|---|---|---|---|---|
| `frag-dev-storage` | `frag-fashion-gen-dataset` | S3 bucket | Catalogue: 260,490 PNGs (`images/{idx//1000:03d}/{idx}.png`) and `metadata/fashion_gen_metadata.parquet`, ~7.6 GB. Filled by `scripts/ingest_fashiongen.py`. Read by both embedding modes (decision 1.5). | Retain |
| `frag-dev-storage` | `frag-chainlit-elements-bucket` | S3 bucket | Chainlit chat-history elements (uploaded and recommended images). | Retain |
| `frag-dev-storage` | `langgraph-checkpoints` | DynamoDB table (on-demand, TTL `ttl`) | LangGraph checkpointer when `CHECKPOINTER__BACKEND=dynamodb`. | Retain |
| `frag-dev-storage` | `chainlit_users` | DynamoDB table (on-demand, GSI `UserThread`) | Chainlit users, threads and steps. | Retain |
| `frag-dev-auth` | FRAG user pool | Cognito user pool (Essentials tier, deletion protection on) | Login for the Chainlit UI. Sign-up is admin-only (decision 1.13); users are created with `create_user.sh`. Users sign in with a username, which `create_user.sh` sets to their email. | Retain |
| `frag-dev-auth` | hosted-UI domain | Cognito domain (managed login v2) | `<prefix>.auth.us-east-1.amazoncognito.com` | Retain |
| `frag-dev-auth` | `fashion-mas` | Cognito app client | Authorization-code flow with a client secret, scopes `openid email profile`, callback `http://localhost:8000/auth/oauth/aws-cognito/callback`. | Retain |
| `frag-dev-auth` | branding | Cognito managed-login branding | The default look of the login page (required by managed login v2). | Retain |

**Bedrock** has no resources to manage. Model access for Claude Haiku 4.5 (through the
`us.` inference profile) and Nova 2 Multimodal Embeddings is enabled once in the
console, per account.

**AWS-managed, not FRAG's:** the service-linked roles `AWSServiceRoleForSupport`,
`AWSServiceRoleForTrustedAdvisor`, `AWSServiceRoleForResourceExplorer` and
`AWSServiceRoleForRDS`, plus the Resource Explorer indexes. They're free account
features that AWS creates on demand.

## Cost

About **$0.18/month**, nearly all of it S3 storage for the catalogue (7.8 GB at
$0.023/GB). DynamoDB on demand costs about $0.000001 per request, and Cognito is free
up to 10,000 monthly active users. Bedrock is billed per call; see the deployment plan.

## Stack layout

Each layer is its own top-level stack: `frag-<env>-storage` and `frag-<env>-auth`.
The deployment plan's later layers (network, compute, edge, wake, ops) follow the same
pattern and read values through `Export`/`!ImportValue`. Separate stacks keep teardown
per layer: removing compute never touches the catalogue or the user pool. All
stateful resources are `Retain`, so even deleting a stack leaves the data in place.

## Setting it up

In a fresh account:

```bash
infra/scripts/deploy.sh storage          # pass CatalogueBucketName=/ElementsBucketName= if the names are taken
infra/scripts/deploy.sh auth DomainPrefix=<globally-unique-prefix>
uv run python scripts/ingest_fashiongen.py ...   # fill the catalogue bucket
infra/scripts/write_env_from_stack.sh --write    # .env from the stack outputs
infra/scripts/create_user.sh <email>
```

After that, `infra/scripts/deploy.sh` with no arguments updates every layer, and does
nothing if the templates haven't changed.
