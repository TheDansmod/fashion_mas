# Infrastructure as code: how-to

All AWS resources are created and changed by committed templates and scripts
(decision 1.8 in `cleanup.md`). Don't use the console or one-off CLI commands. What
exists and why: `docs/aws_inventory.md`.

## Layout

| Path | Contents |
|---|---|
| `infra/cloudformation/storage.yaml` | catalogue and Chainlit elements buckets; checkpointer and Chainlit tables (all Retain) |
| `infra/cloudformation/auth.yaml` | Cognito user pool (admin-only sign-up), hosted-UI domain, app client, login branding (all Retain) |
| `infra/scripts/` | the only supported way to touch AWS resources (below) |

Stacks are named `frag-$FRAG_ENV-<layer>` (`FRAG_ENV` defaults to `dev`). Region is
`AWS_REGION`, default `us-east-1`. Credentials come from the normal AWS CLI chain:
a profile, SSO or environment variables. No script reads or prints them.

## Scripts

Every script that changes something prints what it will do (a change set, or the
list of resources) and continues only after a typed confirmation. All are safe to re-run.

| Script | What it does |
|---|---|
| `deploy.sh [layer] [Key=Value ...] [--dry-run]` | Sets up or updates the stacks through change sets: all layers in order, or one layer. Parameters you don't pass keep their current values. `--dry-run` prints the change set and discards it. |
| `write_env_from_stack.sh [--write]` | Prints the `.env` lines from the stack outputs. `--write` merges them, plus the Cognito client secret (never printed), into `.env`, keeping a `.env.bak`. |
| `create_user.sh <email>` | Admin-creates a Cognito user. Cognito emails them a temporary password. |
| `teardown.sh <layer>` | Deletes a layer stack and its resources. Refuses any stack that contains a Retain resource, which today means both layers; it's meant for future stateless layers such as compute. |

First-time setup order: `docs/aws_inventory.md`, "Setting it up".

## Checks

- `uv run cfn-lint infra/cloudformation/*.yaml` (local; `cfn-lint` is in the dev group).
- `infra/scripts/deploy.sh --dry-run` shows what a deploy would change without changing it.
- Drift: `aws cloudformation detect-stack-drift --stack-name frag-dev-auth` (free), then
  `describe-stack-resource-drifts`.

## Notes

1. Chainlit's Cognito OAuth needs a monkeypatch (in `app.py`): Chainlit's `state`
   parameter isn't valid base64 by Cognito's rules.
2. Chainlit's DynamoDB data layer doesn't create its table; `storage.yaml` defines it
   with the schema Chainlit expects (PK/SK + the `UserThread` GSI).
3. Run `chainlit create-secret` once and put the value in `.env` as `CHAINLIT_AUTH_SECRET`.
4. Changing the Cognito pool's sign-in attributes, the client's `GenerateSecret`, or
   the domain prefix replaces that resource. `deploy.sh` shows `Replacement: True`
   in the change set; don't execute that casually.
