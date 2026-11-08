# voqx justfile (Rust workspace)

# Default recipe - show help
default:
    @just --list

# === Development ===

# Install all binaries
install:
    cargo install --path .

# Install all binaries from workspace
install-all:
    @for crate in $(cargo metadata --no-deps --format-version 1 | jq -r '.packages[] | select(.targets[] | .kind[] == "bin") | .manifest_path | split("/") | .[-2]'); do \
        echo "Installing $crate..."; \
        cargo install --path crates/$crate; \
    done

# Install a specific crate
install-crate CRATE:
    cargo install --path crates/{{CRATE}}

# Uninstall all binaries
uninstall:
    @for crate in $(cargo metadata --no-deps --format-version 1 | jq -r '.packages[] | select(.targets[] | .kind[] == "bin") | .name'); do \
        echo "Uninstalling $crate..."; \
        cargo uninstall $crate 2>/dev/null || true; \
    done

# === Building ===

# Debug build (all crates)
build:
    cargo build --workspace

# Release build (all crates)
build-release:
    cargo build --workspace --release

# Build a specific crate
build-crate CRATE:
    cargo build -p {{CRATE}}

# Build with all features
build-all:
    cargo build --workspace --all-features

# Fast compile check
check:
    cargo check --workspace

# Check a specific crate
check-crate CRATE:
    cargo check -p {{CRATE}}

# Clean build artifacts
clean:
    cargo clean

# === Testing ===

# Run all tests
test:
    cargo test --workspace

# Run tests for a specific crate
test-crate CRATE:
    cargo test -p {{CRATE}}

# Run tests with all features
test-all:
    cargo test --workspace --all-features

# Run tests verbosely
test-v:
    cargo test --workspace -- --nocapture

# Run a specific test
test-one TEST:
    cargo test --workspace {{TEST}}

# Portable and FASTDIAR-only tests; no standalone recognition tool is installed.
test-speaker:
    cargo test --locked -p voqx-recognition --no-default-features --features fastdiar,registry

# Compile/lint library targets and the portable WASM target.
check-speaker:
    cargo check --locked -p voqx-recognition --no-default-features --features fastdiar,registry
    cargo clippy --locked -p voqx-recognition --all-features --all-targets -- -D warnings
    cargo clippy --locked -p voqx-recognition --no-default-features --features wasm --target wasm32-unknown-unknown -- -D warnings

# Explicit generated-signal fixture; no model download or private recordings.
test-speaker-model MODEL FIXTURES:
    VOQX_FASTDIAR_MODEL="{{MODEL}}" VOQX_FASTDIAR_FIXTURE="{{FIXTURES}}" cargo test --locked -p voqx-recognition --no-default-features --features fastdiar,registry --test fastdiar_cpu -- --ignored --nocapture

# wasm-bindgen CLI must match the pinned workspace dependency.
speaker-wasm-build OUTPUT:
    cargo build --locked -p voqx-recognition --no-default-features --features wasm --target wasm32-unknown-unknown --release
    wasm-bindgen --target web --out-dir "{{OUTPUT}}" target/wasm32-unknown-unknown/release/voqx_recognition.wasm

# Synthetic native evidence; stdout can be redirected to a private fixture.
speaker-contract-fixture:
    cargo run --locked -p voqx-recognition --no-default-features --example speaker_contract

# Requires WASM_PKG, NATIVE_FIXTURE, CHROMIUM, PLAYWRIGHT and NEW EVIDENCE_DIR.
speaker-wasm-check:
    bun crates/voqx-recognition/tests/wasm-contract.mjs

# === Code Quality ===

# Format all code
fmt:
    cargo fmt --all

# Check formatting
fmt-check:
    cargo fmt --all -- --check

# Run clippy on all crates
clippy:
    cargo clippy --workspace -- -D warnings

# Alias for clippy
lint: clippy

# Run ast-grep guardrails for Rust sources
lint-rust-ai-guardrails:
    sg scan --config .ast-grep/sgconfig.yml

# Clippy on a specific crate
clippy-crate CRATE:
    cargo clippy -p {{CRATE}} -- -D warnings

# Auto-fix clippy warnings
fix:
    cargo clippy --workspace --fix --allow-dirty

# Run all checks
check-all: fmt-check clippy lint-rust-ai-guardrails test

# === Config Generation ===

# Generate config.toml and schema from Rust structs
generate-config:
    cargo run -p voqx-core --example generate_config

# Validate that examples/ config files are up to date
validate-config:
    cargo test -p voqx-core validate_examples_are_up_to_date

# === Documentation ===

# Generate docs for all crates
docs:
    cargo doc --workspace --no-deps

# Generate and open docs
docs-open:
    cargo doc --workspace --no-deps --open

# Docs for a specific crate
docs-crate CRATE:
    cargo doc -p {{CRATE}} --no-deps --open

# === Dependencies ===

# Update all dependencies
update:
    cargo update

# Check for outdated dependencies
outdated:
    cargo outdated --workspace

# === Workspace Info ===

# List all crates in workspace
list:
    @cargo metadata --no-deps --format-version 1 | jq -r '.packages[] | "\(.name) (\(.version))"'

# List binary crates
list-bins:
    @cargo metadata --no-deps --format-version 1 | jq -r '.packages[] | select(.targets[] | .kind[] == "bin") | .name'

# List library crates
list-libs:
    @cargo metadata --no-deps --format-version 1 | jq -r '.packages[] | select(.targets[] | .kind[] == "lib") | .name'

# === Release ===

# Release build and show binary sizes
release: build-release
    @echo "Binary sizes:"
    @find target/release -maxdepth 1 -type f -perm +111 ! -name "*.d" -exec ls -lh {} \; 2>/dev/null || true

# Tag and push a release
release-tag VERSION:
    git tag v{{VERSION}}
    git push --tags

# Set up GitHub secrets for automated releases (requires byt)
setup-secrets:
    byt secrets setup voqx

# === Registry UI ===

# Serve the registry API (defaults to the shared XDG registry)
ui-api REGISTRY="":
    cargo run -p voqx-api -- --port 3111 {{ if REGISTRY != "" { "--registry " + REGISTRY } else { "" } }}

# Run the registry cleanup UI (needs `just ui-api` in another shell)
ui:
    cd ui && bun install && bun run dev

# Typecheck the UI
ui-check:
    cd ui && bun run typecheck
