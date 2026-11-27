#!/usr/bin/env bash
# Create a Chainlit user in the FRAG Cognito pool (sign-up is admin-only, decision 1.13).
#
#   infra/scripts/create_user.sh <email>
#
# The email is the username and is marked verified. Cognito emails the user a temporary
# password (valid 7 days); they choose a new one at first sign-in. Free: the first
# 10,000 monthly active users are free on the Essentials tier. Asks for confirmation.
# Re-running for an existing user fails with UsernameExistsException and changes nothing.
#
# Env: FRAG_ENV (default dev).

source "$(dirname "$0")/_lib.sh"

[[ $# -eq 1 ]] || die "usage: $0 <email>"
email="$1"
[[ "$email" =~ ^[^@[:space:]]+@[^@[:space:]]+\.[^@[:space:]]+$ ]] || die "'$email' isn't an email address"

auth="$(stack_name auth)"
pool="$(stack_output "$auth" UserPoolId)"
[[ -n "$pool" && "$pool" != None ]] || die "no UserPoolId output on $auth (import/deploy auth first)"

echo "Create user $email in pool $pool? Cognito will email them a temporary password."
confirm_word YES ""
aws cognito-idp admin-create-user --user-pool-id "$pool" --username "$email" \
  --user-attributes Name=email,Value="$email" Name=email_verified,Value=true \
  --desired-delivery-mediums EMAIL \
  --query 'User.[Username,UserStatus]' --output text
info "created; they sign in through the app's Cognito login"
