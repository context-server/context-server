# context-server

A lightweight [MCP](https://modelcontextprotocol.io/) server for **semantic search over markdown knowledge bases**.

One Rust binary. ONNX Runtime is **statically linked** (via [`ort`](https://github.com/pykeio/ort) / [`fastembed`](https://github.com/Anush008/fastembed-rs)) — no separate `libonnxruntime.so` to ship. SQLite is bundled. Built for AI coding agents (Claude Code, Cursor, etc.).

## Features

- Index markdown into a local SQLite vector database
- Chunk by `##` / `###` headings (hierarchy kept in each chunk)
- Semantic search with All-MiniLM-L6-v2 (384-dim)
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

## Build

```bash
cargo build --release
```

The first embedding run downloads the MiniLM model into the local Hugging Face / fastembed cache (~tens of MB, once).

## Usage

```bash
# Preview how documents will be chunked
./target/release/context-server index --input ./docs --dry-run

# Embed and write the database
./target/release/context-server index --input ./docs --db context.db

# CLI search
./target/release/context-server search --db context.db "how do we handle backports"

# MCP stdio server
./target/release/context-server serve --db context.db
```

### Claude Code

```bash
claude mcp add --transport stdio --scope user context-server \
  -- /absolute/path/to/context-server serve --db /absolute/path/to/context.db
```

Re-index when content changes, then restart the MCP session so `serve` reloads the DB into memory.

## MCP tools

| Tool | Description |
|------|-------------|
| `semantic_search` | Ranked passages with similarity scores |
| `list_documents` | Indexed chunk listing |
| `answer_question` | Top passage for a question (retrieval only, no generative QA) |

## Architecture

| Piece | Choice |
|-------|--------|
| Embeddings | fastembed → All-MiniLM-L6-v2, L2-normalized |
| Inference | ort (static ONNX Runtime) |
| Storage | rusqlite (bundled SQLite), float32 blobs |
| Search | Brute-force cosine (fine for &lt;100K chunks) |
| MCP | [rmcp](https://github.com/modelcontextprotocol/rust-sdk) stdio |

See [PLAN.md](PLAN.md) for design notes and roadmap.

## Development

```bash
cargo test
cargo build --release
```

## License

MIT — see [LICENSE](LICENSE).
