#!/usr/bin/env bash
# Recreate Anurag's OpenJarvis setup on a Mac (Apple Silicon, Homebrew).
#
#   git clone -b AnuragMain https://github.com/Anuragsingh691/OpenJarvis.git
#   cd OpenJarvis && ./scripts/setup-anurag.sh
#
# Safe to re-run: installed tools, pulled models and an identical config are
# left alone. An existing, different ~/.openjarvis/config.toml is backed up.
set -euo pipefail

MODELS=(qwen3.5:9b qwen3:14b gemma3:12b qwen3.5:4b qwen3:0.6b)

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

info() { printf '\033[0;34m[info]\033[0m  %s\n' "$*"; }
ok()   { printf '\033[0;32m[ok]\033[0m    %s\n' "$*"; }
fail() { printf '\033[0;31m[fail]\033[0m  %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = Darwin ] || fail "This script targets macOS."
command -v brew >/dev/null || fail "Homebrew is required: https://brew.sh"
command -v node >/dev/null || fail "Node.js 22+ is required: https://nodejs.org"

# 1. System tools: uv (Python), Ollama (local models), rustup (native
#    extension), espeak-ng (Kokoro text-to-speech pronunciation).
for pkg in uv ollama rustup espeak-ng; do
  if brew list --formula "$pkg" >/dev/null 2>&1; then
    ok "$pkg already installed"
  else
    info "Installing $pkg..."
    brew install "$pkg"
  fi
done
export PATH="$(brew --prefix rustup)/bin:$HOME/.cargo/bin:$PATH"

# 2. Rust toolchain pinned by the repo.
RUST_TOOLCHAIN="$(sed -n 's/^channel *= *"\(.*\)"/\1/p' rust/rust-toolchain.toml)"
info "Ensuring Rust $RUST_TOOLCHAIN..."
rustup toolchain install "$RUST_TOOLCHAIN" --profile minimal -c rustfmt -c clippy >/dev/null
ok "Rust $(cd rust && rustc --version | cut -d' ' -f2)"

# 3. Ollama as a login service.
brew services start ollama >/dev/null 2>&1 || true
for _ in $(seq 30); do
  curl -sf http://localhost:11434/api/tags >/dev/null && break
  sleep 1
done
curl -sf http://localhost:11434/api/tags >/dev/null || fail "Ollama did not start"
ok "Ollama running"

# 4. Python environment (Python 3.12; the project does not support 3.14) with
#    the GUI, web search and voice extras.
info "Installing Python dependencies..."
uv sync --python 3.12 --extra desktop --extra tools-search --extra voice --quiet
ok "Python dependencies installed"

# 5. Native extension. Built after `uv sync`, which removes packages it did
#    not install itself.
info "Building the Rust extension..."
uv pip install --quiet maturin
(cd rust && uv run --no-sync --project .. maturin develop --release --quiet \
  -m crates/openjarvis-python/Cargo.toml)
ok "Rust extension built"

# 6. Frontend, with the npm version the frontend pins (without changing the
#    global npm).
NPM_PIN="$(sed -n 's/.*"packageManager": *"npm@\([^"]*\)".*/\1/p' frontend/package.json)"
info "Installing frontend dependencies (npm $NPM_PIN)..."
(cd frontend && npx -y "npm@$NPM_PIN" install --silent)
ok "Frontend dependencies installed"

# 7. Local models (~30 GB total).
for model in "${MODELS[@]}"; do
  if ollama list | awk 'NR>1 {print $1}' | grep -qx "$model"; then
    ok "Model $model already pulled"
  else
    info "Pulling $model..."
    ollama pull "$model"
  fi
done

# 8. Personal config.
CONFIG_DIR="$HOME/.openjarvis"
TARGET="$CONFIG_DIR/config.toml"
mkdir -p "$CONFIG_DIR"
if [ -f "$TARGET" ] && cmp -s "$TARGET" configs/anurag/config.toml; then
  ok "Config already up to date"
else
  if [ -f "$TARGET" ]; then
    BACKUP="$TARGET.bak.$(date +%Y%m%d-%H%M%S)"
    cp "$TARGET" "$BACKUP"
    info "Backed up existing config to $BACKUP"
  fi
  cp configs/anurag/config.toml "$TARGET"
  chmod 600 "$TARGET"
  ok "Installed config to $TARGET"
fi

info "Health check:"
uv run --no-sync jarvis doctor 2>/dev/null | tail -4 || true

cat <<'EOF'

Setup complete. Start Jarvis with:
  uv run jarvis gui        # browser chat UI
  uv run jarvis chat --wake-word   # hands-free in the terminal

In the GUI: press Cmd+K to pick a model, then Settings -> Speech and turn on
Speech-to-Text, "Hey Jarvis" wake word, Text-to-Speech and
"Speak replies automatically".
EOF
