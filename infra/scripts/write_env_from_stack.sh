#!/usr/bin/env bash
# Turn the storage and auth stack outputs into .env lines (replaces hand-copying).
#
#   infra/scripts/write_env_from_stack.sh            # print the non-secret lines
#   infra/scripts/write_env_from_stack.sh --write    # also update .env in place
#
# --write updates only the keys listed below (other lines, comments and order are kept;
# missing keys are appended), saves the previous file as .env.bak, and is the ONLY
# place that reads the Cognito client secret: it goes from describe-user-pool-client
# straight into .env and is never printed. Re-running it is idempotent.
#
# Env: FRAG_ENV (default dev), FRAG_ENV_FILE (default <repo>/.env).

source "$(dirname "$0")/_lib.sh"

write=false
case "${1:-}" in
  "") ;;
  --write) write=true ;;
  *) die "usage: $0 [--write]" ;;
esac
env_file="${FRAG_ENV_FILE:-${REPO_DIR}/.env}"

storage="$(stack_name storage)"
auth="$(stack_name auth)"
for s in "$storage" "$auth"; do
  [[ "$(stack_status "$s")" == *_COMPLETE ]] || die "stack $s not found; import/deploy it first"
done
get() {
  local v
  v="$(stack_output "$1" "$2")"
  [[ -n "$v" && "$v" != None ]] || die "$1 has no output $2 (run deploy.sh once after an import)"
  echo "$v"
}

# key=value pairs, in the order they're printed/appended
pairs=(
  "CATALOGUE__S3_BUCKET=$(get "$storage" CatalogueBucketName)"
  "CHECKPOINTER__DYNAMODB_TABLE=$(get "$storage" CheckpointTableName)"
  "CHAT_HISTORY__DYNAMODB_TABLE=$(get "$storage" ChatHistoryTableName)"
  "CHAT_HISTORY__S3_BUCKET=$(get "$storage" ElementsBucketName)"
  "OAUTH_COGNITO_CLIENT_ID=$(get "$auth" ClientId)"
  "OAUTH_COGNITO_DOMAIN=$(get "$auth" CognitoDomain)"
  'OAUTH_COGNITO_SCOPES="openid email profile"'
)
printf '%s\n' "${pairs[@]}"

if ! $write; then
  echo "# OAUTH_COGNITO_CLIENT_SECRET is not printed; run with --write to put it in $env_file" >&2
  exit 0
fi

pool="$(get "$auth" UserPoolId)"
client="$(get "$auth" ClientId)"
# Secret: captured into an env var for the python step only; never echoed.
FRAG_SECRET="$(aws cognito-idp describe-user-pool-client --user-pool-id "$pool" --client-id "$client" \
  --query 'UserPoolClient.ClientSecret' --output text)"
[[ -n "$FRAG_SECRET" && "$FRAG_SECRET" != None ]] || die "client has no secret"
export FRAG_SECRET

python3 - "$env_file" "${pairs[@]}" <<'PY'
import os, re, shutil, sys
from pathlib import Path

path = Path(sys.argv[1])
updates = dict(p.split("=", 1) for p in sys.argv[2:])
updates["OAUTH_COGNITO_CLIENT_SECRET"] = os.environ["FRAG_SECRET"]

lines = path.read_text().splitlines() if path.exists() else []
if path.exists():
    shutil.copy2(path, path.with_name(path.name + ".bak"))
seen = set()
out = []
for line in lines:
    m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
    if m and m.group(1) in updates:
        key = m.group(1)
        if key in seen:
            continue  # drop duplicates of a managed key
        out.append(f"{key}={updates[key]}")
        seen.add(key)
    else:
        out.append(line)
missing = [k for k in updates if k not in seen]
if missing:
    out += ["", "# written by infra/scripts/write_env_from_stack.sh"]
    out += [f"{k}={updates[k]}" for k in missing]
path.write_text("\n".join(out) + "\n")
os.chmod(path, 0o600)
print(f"updated {len(updates)} keys in {path} (previous version: {path.name}.bak)", file=sys.stderr)
PY
