#!/usr/bin/env bash
# Set up (or update) FRAG's AWS resources through reviewed change sets.
#
#   infra/scripts/deploy.sh [layer] [ParamKey=Value ...] [--dry-run]
#
#   layer           storage | auth; omit to deploy all layers in order (storage, auth)
#   ParamKey=Value  parameter overrides for a single layer; on updates, parameters you
#                   don't pass keep their current values
#   --dry-run       create and print the change set(s), then discard them
#
# For each layer: `aws cloudformation deploy --no-execute-changeset` -> print the
# change set -> execute only after a typed YES. Re-running with no template or
# parameter changes is a no-op, so this is also the update command.
#
# First-time setup in a new account:
#   infra/scripts/deploy.sh storage                # buckets + tables
#   infra/scripts/deploy.sh auth DomainPrefix=<globally-unique-prefix>
#   infra/scripts/write_env_from_stack.sh --write  # .env from the stack outputs
#   infra/scripts/create_user.sh <email>
# Bucket names are global too: pass CatalogueBucketName=... ElementsBucketName=... to
# storage if the defaults are taken. The catalogue is then filled with
# scripts/ingest_fashiongen.py.
#
# Later example: add the CloudFront callback once the edge stack exists:
#   infra/scripts/deploy.sh auth "CallbackUrls=http://localhost:8000/auth/oauth/aws-cognito/callback,https://dxxxx.cloudfront.net/auth/oauth/aws-cognito/callback"

source "$(dirname "$0")/_lib.sh"

dry_run=false
layers=()
params=()
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry_run=true ;;
    *=*) params+=("$arg") ;;
    -*) die "unknown option '$arg'" ;;
    *) template_for "$arg" >/dev/null; layers+=("$arg") ;;
  esac
done
[[ ${#layers[@]} -le 1 ]] || die "give at most one layer"
if [[ ${#layers[@]} -eq 0 ]]; then
  [[ ${#params[@]} -eq 0 ]] || die "parameter overrides need a layer"
  layers=("${LAYERS[@]}")
fi

deploy_layer() {
  local layer="$1" template stack status out arn
  template="$(template_for "$layer")"
  stack="$(stack_name "$layer")"
  status="$(stack_status "$stack")"
  case "$status" in
    NONE | *_COMPLETE) ;;
    *) die "$stack is in $status; resolve that first" ;;
  esac

  info "creating change set for $stack (current status: $status)"
  out="$(aws cloudformation deploy --template-file "$template" --stack-name "$stack" \
    --no-execute-changeset --no-fail-on-empty-changeset \
    --tags project=frag "env=$FRAG_ENV" \
    --parameter-overrides "EnvName=$FRAG_ENV" ${params[@]+"${params[@]}"} 2>&1)" \
    || { echo "$out" >&2; die "change set creation failed"; }
  arn="$(grep -oE 'arn:aws[a-z-]*:cloudformation:[^ ]*:changeSet/[^ ]*' <<<"$out" | head -1 || true)"
  if [[ -z "$arn" ]]; then
    info "no changes for $stack"
    return
  fi

  show_change_set "$stack" "$arn"
  if $dry_run; then
    discard_change_set "$stack" "$arn"
    info "dry run: change set discarded"
    return
  fi
  if ! ask_word YES "Execute this change set on $stack?"; then
    discard_change_set "$stack" "$arn"
    die "declined; the change set was deleted and nothing was changed"
  fi
  execute_change_set "$stack" "$arn"
  aws cloudformation describe-stacks --stack-name "$stack" \
    --query 'Stacks[0].Outputs[].[OutputKey,OutputValue]' --output table
}

for layer in "${layers[@]}"; do deploy_layer "$layer"; done
