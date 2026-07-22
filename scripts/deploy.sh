#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
STATIC_ROOT="${CHESSENGINES_STATIC_ROOT:-/srv/chessengines}"
PUBLIC_URL="${CHESSENGINES_PUBLIC_URL:-/projects/chessengines/}"
SERVICE_NAME="${CHESSENGINES_SERVICE:-chessengines-random.service}"
LIVE_URL="${CHESSENGINES_LIVE_URL:-https://matteob.dev/projects/chessengines/}"
CARGO_COMMAND="${CARGO_COMMAND:-$HOME/.cargo/bin/cargo}"
TRUNK_COMMAND="${TRUNK_COMMAND:-$HOME/.cargo/bin/trunk}"

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Required command not found: $1" >&2
        exit 1
    fi
}

require_command "$CARGO_COMMAND"
require_command "$TRUNK_COMMAND"
require_command rsync
require_command systemctl
require_command curl

CARGO_BIN_DIR="$(dirname -- "$(command -v "$CARGO_COMMAND")")"

cd "$REPO_ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Refusing to deploy a dirty working tree. Commit or stash changes first." >&2
    exit 1
fi

echo "Testing workspace..."
"$CARGO_COMMAND" test --workspace

echo "Building random bot..."
"$CARGO_COMMAND" build --release -p random

echo "Building frontend..."
(
    cd frontend
    PATH="$CARGO_BIN_DIR:$PATH" NO_COLOR=true \
        "$TRUNK_COMMAND" build --release --public-url "$PUBLIC_URL"
)

echo "Publishing frontend to $STATIC_ROOT..."
rsync -a frontend/dist/ "$STATIC_ROOT/"

echo "Restarting $SERVICE_NAME..."
systemctl --user restart "$SERVICE_NAME"
systemctl --user is-active --quiet "$SERVICE_NAME"

echo "Verifying live page and API..."
curl --fail --silent --show-error "$LIVE_URL" >/dev/null
curl --fail --silent --show-error \
    -H "content-type: application/json" \
    --data '{"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","san":"e4"}' \
    "${LIVE_URL%/}/api/move" >/dev/null

echo "Deployment complete: $LIVE_URL"
