#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DEPLOY_REMOTE="${CHESSENGINES_DEPLOY_REMOTE:-origin}"
DEPLOY_BRANCH="${CHESSENGINES_DEPLOY_BRANCH:-main}"

cd "$REPO_ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Refusing to update a dirty working tree:" >&2
    git status --short >&2
    echo "Commit and push these changes, or remove them, before deploying." >&2
    exit 1
fi

CURRENT_BRANCH="$(git symbolic-ref --quiet --short HEAD || true)"
if [[ "$CURRENT_BRANCH" != "$DEPLOY_BRANCH" ]]; then
    echo "Refusing to deploy branch '${CURRENT_BRANCH:-detached HEAD}'." >&2
    echo "Switch to $DEPLOY_BRANCH first: git switch $DEPLOY_BRANCH" >&2
    exit 1
fi

echo "Fetching $DEPLOY_REMOTE/$DEPLOY_BRANCH..."
git fetch --prune "$DEPLOY_REMOTE" "$DEPLOY_BRANCH"

REMOTE_REF="$DEPLOY_REMOTE/$DEPLOY_BRANCH"
if ! git merge-base --is-ancestor HEAD "$REMOTE_REF"; then
    echo "Cannot fast-forward $DEPLOY_BRANCH to $REMOTE_REF." >&2
    echo "The production checkout has local-only or divergent commits." >&2
    echo "Resolve them without rewriting production history, then retry." >&2
    exit 1
fi

git merge --ff-only "$REMOTE_REF"

LOCAL_COMMIT="$(git rev-parse HEAD)"
REMOTE_COMMIT="$(git rev-parse "$REMOTE_REF")"
if [[ "$LOCAL_COMMIT" != "$REMOTE_COMMIT" ]]; then
    echo "Refusing to deploy: HEAD does not match $REMOTE_REF." >&2
    exit 1
fi

echo "Deploying $DEPLOY_BRANCH at $LOCAL_COMMIT..."
exec "$SCRIPT_DIR/deploy.sh"
