# Shared helpers for infra/scripts/*.sh. Source it; don't run it.
#
# Stack layout: one standalone stack per layer, named frag-<env>-<layer>, e.g.
# frag-dev-storage. Templates live in infra/cloudformation/<layer>.yaml.
#
# Environment:
#   FRAG_ENV          environment name (default: dev)
#   AWS_REGION        default us-east-1; credentials come from the usual AWS CLI chain
#                     (AWS_PROFILE, SSO, ...). Nothing here reads or prints credentials.
#   FRAG_ASSUME_YES   set to 1 to answer confirmations automatically (automation only;
#                     change sets and resource lists are still printed)

set -euo pipefail

export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_PAGER=""
FRAG_ENV="${FRAG_ENV:-dev}"

INFRA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${INFRA_DIR}/.." && pwd)"
CFN_DIR="${INFRA_DIR}/cloudformation"

# Layers in deployment order.
LAYERS=(storage auth)

die() { echo "error: $*" >&2; exit 1; }
info() { echo "==> $*" >&2; }

stack_name() { echo "frag-${FRAG_ENV}-$1"; }

template_for() {
  local f="${CFN_DIR}/$1.yaml"
  [[ -f "$f" ]] || die "unknown layer '$1' (expected one of: ${LAYERS[*]})"
  echo "$f"
}

# Prints the stack status, or NONE if the stack doesn't exist.
stack_status() {
  aws cloudformation describe-stacks --stack-name "$1" \
    --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo NONE
}

stack_output() {  # stack_output <stack> <OutputKey>
  aws cloudformation describe-stacks --stack-name "$1" \
    --query "Stacks[0].Outputs[?OutputKey=='$2'].OutputValue" --output text
}

# ask_word <word> <prompt>: succeeds only if the user types <word> exactly.
ask_word() {
  local reply
  if [[ "${FRAG_ASSUME_YES:-}" == 1 ]]; then
    echo "$2 [FRAG_ASSUME_YES=1: answering '$1']" >&2
    return 0
  fi
  [[ -t 0 ]] || die "needs an interactive terminal to confirm (or FRAG_ASSUME_YES=1)"
  read -r -p "$2 Type '$1' to continue: " reply
  [[ "$reply" == "$1" ]]
}

# confirm_word <word> <prompt>: like ask_word, but exits if not confirmed.
confirm_word() {
  ask_word "$@" || die "not confirmed; nothing was changed by this step"
}

# Pretty-print a change set and its resource changes.
show_change_set() {  # show_change_set <stack> <change-set-name-or-arn>
  aws cloudformation describe-change-set --stack-name "$1" --change-set-name "$2" \
    --query '{Status:Status,Reason:StatusReason,Parameters:Parameters[].[ParameterKey,ParameterValue]}' \
    --output yaml
  aws cloudformation describe-change-set --stack-name "$1" --change-set-name "$2" \
    --query 'Changes[].ResourceChange.[Action,LogicalResourceId,ResourceType,Replacement,join(`,`,Details[].Target.Name || `[]`)]' \
    --output table
}

# Execute an already-reviewed change set and wait for the stack to settle.
execute_change_set() {  # execute_change_set <stack> <change-set-arn>
  aws cloudformation execute-change-set --stack-name "$1" --change-set-name "$2"
  info "waiting for $1 ..."
  local status
  while :; do
    status="$(stack_status "$1")"
    case "$status" in
      *_IN_PROGRESS) sleep 10 ;;
      *) break ;;
    esac
  done
  info "$1: $status"
  case "$status" in
    CREATE_COMPLETE | UPDATE_COMPLETE) ;;
    *) die "stack ended in $status; check its events in the console or with describe-stack-events" ;;
  esac
}

# Delete a change set that won't be executed (and the empty REVIEW_IN_PROGRESS
# placeholder a CREATE change set leaves behind for a new stack; it holds no resources).
discard_change_set() {  # discard_change_set <stack> <change-set-arn>
  aws cloudformation delete-change-set --stack-name "$1" --change-set-name "$2"
  if [[ "$(stack_status "$1")" == REVIEW_IN_PROGRESS ]]; then
    aws cloudformation delete-stack --stack-name "$1"
  fi
}
