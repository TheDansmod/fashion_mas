#!/usr/bin/env bash
# Delete one FRAG layer stack and the resources it created.
#
#   infra/scripts/teardown.sh <layer>
#
# Refuses any stack that contains a resource with DeletionPolicy Retain (storage and
# auth: the catalogue, chat history and user pool are never torn down by script; to
# remove them, delete the stack, then the retained resources by hand). For other
# stacks it lists what will be deleted and asks you to type the stack name.
# Idempotent: a stack that's already gone is reported and skipped.

source "$(dirname "$0")/_lib.sh"

[[ $# -eq 1 ]] || die "usage: $0 <layer>"
layer="$1"
template_for "$layer" >/dev/null
stack="$(stack_name "$layer")"
status="$(stack_status "$stack")"
if [[ "$status" == NONE ]]; then
  info "$stack doesn't exist; nothing to do"
  exit 0
fi

# Read the DeletionPolicy of every resource from the stack's *live* template.
body="$(mktemp)"
trap 'rm -f "$body"' EXIT
aws cloudformation get-template --stack-name "$stack" --query TemplateBody --output json >"$body"
retained="$(cd "$REPO_DIR" && uv run --quiet python - "$body" <<'PY'
import json, sys
from cfnlint.decode import decode_str
raw = json.load(open(sys.argv[1]))          # a JSON template arrives as an object,
text = raw if isinstance(raw, str) else json.dumps(raw)   # a YAML one as a string
template, errors = decode_str(text)
if errors or template is None:
    sys.exit(f"could not parse the live template: {errors}")
for logical, res in template.get("Resources", {}).items():
    if res.get("DeletionPolicy", "Delete") == "Retain":
        print(logical)
PY
)"
if [[ -n "$retained" ]]; then
  die "$stack has resources with DeletionPolicy Retain, which this script never touches:
$(sed 's/^/  /' <<<"$retained")"
fi

info "$stack ($status) will be deleted together with:"
aws cloudformation describe-stack-resources --stack-name "$stack" \
  --query 'StackResources[].[LogicalResourceId,ResourceType,PhysicalResourceId]' --output table
confirm_word "$stack" "Delete these resources?"
aws cloudformation delete-stack --stack-name "$stack"
info "waiting for $stack to be deleted ..."
aws cloudformation wait stack-delete-complete --stack-name "$stack"
info "deleted $stack"
