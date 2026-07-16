# context-server

A lightweight [MCP](https://modelcontextprotocol.io/) server for **semantic search over markdown knowledge bases**.

One Rust binary. ONNX Runtime is **statically linked** (via [`ort`](https://github.com/pykeio/ort) / [`fastembed`](https://github.com/Anush008/fastembed-rs)) — no separate `libonnxruntime.so` to ship. SQLite is bundled. Built for AI coding agents (Claude Code, Cursor, etc.).

## Features

- Index markdown into a local SQLite vector database
- Chunk by `##` / `###` headings (hierarchy kept in each chunk); oversized sections are split with overlap
- Hybrid search: dense embeddings (All-MiniLM-L6-v2) + BM25, fused with reciprocal rank fusion
- MCP tools: `semantic_search`, `list_documents`, `answer_question`
- CLI for index / search / embed smoke tests

**Input contract:** feed searchable prose (markdown). Structured data (YAML, etc.) should be converted to markdown *before* indexing — raw YAML in code fences searches poorly.

## Requirements

- Rust 1.75+ (edition 2021)
- Linux x86_64 (primary target today)
- At build/link time: a C++ standard library (`libstdc++`) and OpenSSL development headers if your platform needs them for `native-tls`

On Fedora/RHEL, if the linker cannot find `-lstdc++` (only `libstdc++.so.6` is installed):

```bash
mkdir -p .linker && ln -sfn /usr/lib64/libstdc++.so.6 .linker/libstdc++.so
export RUSTFLAGS="-L native=$(pwd)/.linker"
```

## Install

```bash
pip install context-server
```

Platform wheels: Linux x86_64/aarch64 (`manylinux_2_39`, glibc 2.39+ / Ubuntu 24.04+) and macOS Apple Silicon.

## Build

```bash
cargo build --release
```

The first embedding run downloads the MiniLM model into the local Hugging Face / fastembed cache (~tens of MB, once).

### Linux wheels (Podman)

Same Containerfile CI uses (Ubuntu 24.04 / glibc 2.39 — required by current ORT prebuilts):

```bash
./scripts/build-wheel.sh   # writes dist/*.whl
VERSION=2026.716.1 ./scripts/build-wheel.sh  # optional CalVer override
```

## Releasing (CalVer)

Versions are **CalVer** `YYYY.MMDD.N` (e.g. `2026.716.1`) so they are valid for
both Cargo SemVer and PyPI. `pyproject.toml` takes the version from
`Cargo.toml`; release CI rewrites that from the git tag before building.

```bash
tag="$(./scripts/next-calver.sh)"
git tag -a "$tag" -m "$tag"
git push origin "$tag"    # triggers Release workflow → PyPI
```

`setuptools-scm` is not used (maturin cannot consume it).
## Usage

```bash
# Preview how documents will be chunked
./target/release/context-server index --input ./docs --dry-run

# Embed and write the database
./target/release/context-server index --input ./docs --db context.db

# Optional: store MCP server instructions in the DB (when to call this corpus)
./target/release/context-server index --input ./docs --db context.db \
  --instructions-file ./mcp-instructions.txt
# or: --instructions 'Call semantic_search for questions about …'

# CLI search (hybrid by default; also --mode dense|lexical)
./target/release/context-server search --db context.db "how do we handle backports"

# MCP stdio server
./target/release/context-server serve --db context.db
```

### MCP instructions in the DB

`index --instructions` / `--instructions-file` writes `meta.instructions`.
`serve` exposes that text as MCP `ServerInfo.instructions` so clients know when
to call this corpus (falls back to a generic blurb if unset).

`serve` and `search` accept a remote `--db` and download it into the local cache
(`$XDG_CACHE_HOME/context-server/dbs/...`, or `~/.cache/...`) before opening:

```bash
# Short form (globally unique bucket)
context-server serve --db 'gs://vme-cnv-context/latest/cnv.db'

# Project-qualified (still requires gs://; stripped for the Storage API)
context-server serve --db \
  'gs://projects/itpc-gcp-hcm-pe-eng-claude/buckets/vme-cnv-context/objects/latest/cnv.db'
```

Uses [Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials)
(`gcloud auth application-default login`, or `GOOGLE_APPLICATION_CREDENTIALS`).
When a sibling `{object}.sha256` exists (sha256sum format), the download is
skipped if the local cache already matches; otherwise the DB is re-fetched and
verified. `index` still writes a local path only.

### Claude Code

```bash
claude mcp add --transport stdio --scope user context-server \
  -- /absolute/path/to/context-server serve --db /absolute/path/to/context.db
```

Re-index when content changes, then restart the MCP session so `serve` reloads the DB into memory.
For a GCS-backed DB, point `--db` at a `gs://...` URI.

If Claude rarely calls the tools (tool search defers MCP tools), add `"alwaysLoad": true` to the server entry in your Claude MCP config so these tools stay visible every turn.

## MCP tools

| Tool | Description |
|------|-------------|
| `semantic_search` | Ranked passages with similarity scores |
| `list_documents` | Indexed chunk listing |
| `answer_question` | Top passage for a question (retrieval only, no generative QA) |

## Architecture

| Piece | Choice |
|-------|--------|
| Embeddings | fastembed → All-MiniLM-L6-v2, L2-normalized (model id stored in DB) |
| Inference | ort (static ONNX Runtime) |
| Storage | rusqlite (bundled SQLite), float32 blobs |
| Search | Hybrid: cosine dense + BM25 → reciprocal rank fusion |
| Chunking | `##` / `###` + split when text exceeds ~900 chars |
| MCP | [rmcp](https://github.com/modelcontextprotocol/rust-sdk) stdio |

See [PLAN.md](PLAN.md) for design notes and roadmap.

## Development

```bash
cargo test
cargo build --release
```

## License

MIT — see [LICENSE](LICENSE).
