#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
STATIC_ROOT="${CHESSENGINES_STATIC_ROOT:-/srv/chessengines}"
PUBLIC_URL="${CHESSENGINES_PUBLIC_URL:-/projects/chessengines/}"
RANDOM_SERVICE_NAME="${CHESSENGINES_RANDOM_SERVICE:-${CHESSENGINES_SERVICE:-chessengines-random.service}}"
MINIMAX_SERVICE_NAME="${CHESSENGINES_MINIMAX_SERVICE:-chessengines-minimax.service}"
ALPHAMINI_SERVICE_NAME="${CHESSENGINES_ALPHAMINI_SERVICE:-chessengines-alphamini.service}"
ALPHAMINI_MODEL_PATH="${ALPHAMINI_MODEL_PATH:-$REPO_ROOT/artifacts/alphamini/current/model.onnx}"
ALPHAMINI_MANIFEST_PATH="${ALPHAMINI_MANIFEST_PATH:-$REPO_ROOT/artifacts/alphamini/current/manifest.json}"
LIVE_URL="${CHESSENGINES_LIVE_URL:-https://apps.matteob.dev/projects/chessengines/}"
CADDY_CONFIG="${CHESSENGINES_CADDY_CONFIG:-/etc/caddy/Caddyfile}"
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

check_proxy_config() {
    local route
    local missing_routes=()
    local required_routes=(
        "handle_path /projects/chessengines/api/random/*"
        "handle_path /projects/chessengines/api/minimax/*"
        "handle_path /projects/chessengines/api/alphamini/*"
    )

    if [[ ! -r "$CADDY_CONFIG" ]]; then
        echo "Cannot read Caddy config: $CADDY_CONFIG" >&2
        echo "Set CHESSENGINES_CADDY_CONFIG to the active config path." >&2
        exit 1
    fi

    for route in "${required_routes[@]}"; do
        if ! grep -Fq "$route" "$CADDY_CONFIG"; then
            missing_routes+=("$route")
        fi
    done

    if (( ${#missing_routes[@]} > 0 )); then
        echo "Caddy is missing the namespaced Chess Engines API routes:" >&2
        printf '  %s\n' "${missing_routes[@]}" >&2
        echo >&2
        echo "Merge deploy/caddy/chessengines.caddy before the static-file handler in:" >&2
        echo "  $CADDY_CONFIG" >&2
        echo "Then validate and reload Caddy before deploying:" >&2
        echo "  sudo caddy validate --config $CADDY_CONFIG" >&2
        echo "  sudo systemctl reload caddy" >&2
        exit 1
    fi
}

check_proxy_config

if [[ ! -r "$ALPHAMINI_MODEL_PATH" || ! -r "$ALPHAMINI_MANIFEST_PATH" ]]; then
    echo "AlphaMini's immutable model artifact is not provisioned." >&2
    echo "Expected readable files:" >&2
    echo "  $ALPHAMINI_MODEL_PATH" >&2
    echo "  $ALPHAMINI_MANIFEST_PATH" >&2
    echo "Provision a manifest-validated checkpoint, then atomically switch artifacts/alphamini/current." >&2
    exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Refusing to deploy a dirty working tree. Commit or stash changes first." >&2
    exit 1
fi

echo "Testing workspace..."
"$CARGO_COMMAND" test --workspace --locked

echo "Building bot APIs..."
"$CARGO_COMMAND" build --locked --release -p random -p minimax -p alphamini --features alphamini/onnx

echo "Verifying AlphaMini model and manifest..."
ALPHAMINI_MODEL_PATH="$ALPHAMINI_MODEL_PATH" \
ALPHAMINI_MANIFEST_PATH="$ALPHAMINI_MANIFEST_PATH" \
    target/release/alphamini --verify-only

echo "Building frontend..."
(
    cd frontend
    PATH="$CARGO_BIN_DIR:$PATH" NO_COLOR=true \
        "$TRUNK_COMMAND" build --release --public-url "$PUBLIC_URL"
)

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Build changed the working tree; refusing to publish an unrecorded state." >&2
    exit 1
fi

echo "Publishing frontend to $STATIC_ROOT..."
rsync -a --delete frontend/dist/ "$STATIC_ROOT/"

echo "Restarting $RANDOM_SERVICE_NAME, $MINIMAX_SERVICE_NAME, and $ALPHAMINI_SERVICE_NAME..."
systemctl --user restart "$RANDOM_SERVICE_NAME" "$MINIMAX_SERVICE_NAME" "$ALPHAMINI_SERVICE_NAME"
systemctl --user is-active --quiet "$RANDOM_SERVICE_NAME"
systemctl --user is-active --quiet "$MINIMAX_SERVICE_NAME"
systemctl --user is-active --quiet "$ALPHAMINI_SERVICE_NAME"

echo "Verifying live page and API..."
curl --fail --silent --show-error "$LIVE_URL" >/dev/null
curl --fail --silent --show-error \
    -H "content-type: application/json" \
    --data '{"san":"1. e4 e5 2. Nf3"}' \
    "${LIVE_URL%/}/api/random/move" >/dev/null
curl --fail --silent --show-error --max-time 60 \
    -H "content-type: application/json" \
    --data '{"san":"1. e4 e5 2. Nf3"}' \
    "${LIVE_URL%/}/api/minimax/move" >/dev/null
curl --fail --silent --show-error --max-time 60 \
    -H "content-type: application/json" \
    --data '{"san":"1. e4 e5 2. Nf3"}' \
    "${LIVE_URL%/}/api/minimax/depth-3/move" >/dev/null
curl --fail --silent --show-error --max-time 60 \
    -H "content-type: application/json" \
    --data '{"san":"1. e4 e5 2. Nf3"}' \
    "${LIVE_URL%/}/api/alphamini/move" >/dev/null

echo "Deployment complete: $LIVE_URL"
