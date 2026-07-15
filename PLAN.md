# Design notes

## Goal

A single-binary MCP server that indexes organizational markdown into a local vector database and serves semantic search over stdio. Replace heavyweight Python/torch knowledge-base stacks with something you can distribute as one file.

## Why Rust

- [`ort`](https://github.com/pykeio/ort) can statically link ONNX Runtime (no separate `.so` for end users)
- [`fastembed`](https://github.com/Anush008/fastembed-rs) provides All-MiniLM-L6-v2 + tokenizers
- Official [`rmcp`](https://github.com/modelcontextprotocol/rust-sdk) MCP SDK

## CLI

```text
context-server index --input <dir> --db context.db
context-server serve --db context.db
context-server search --db context.db "<query>"
context-server embed "<text>"
```

## Modules (`src/`)

| Module | Role |
|--------|------|
| `embed` | fastembed AllMiniLML6V2 (384-dim, L2-normalized) |
| `index` | Markdown heading chunker (`##` / `###`), directory walk |
| `store` | SQLite: documents + embeddings |
| `search` | Brute-force cosine (dot product on normalized vectors) |
| `mcp` | rmcp stdio tools |
| `main` | clap CLI |

## Schema

- `documents`: id, source_path, chunk_index, text, headings (JSON), metadata (JSON)
- `embeddings`: id, dim, vector (little-endian float32 blob)

`index` replaces the full DB contents each run (`ReplaceAll`).

## Input contract

Only `.md` / `.markdown` files are indexed. Structured sources (team YAML, etc.) must be converted to prose markdown **upstream**. Putting YAML in fenced code blocks produces poor retrieval; a summary paragraph that keeps entity, roles, and parent together in one chunk works much better.

## MCP tools

- `semantic_search(query, limit)`
- `list_documents(limit)`
- `answer_question(question, limit)` — retrieval only

## Status

- [x] Index / search / embed / serve
- [x] Markdown chunking + unit tests
- [x] Static ORT (no `libonnxruntime.so` in `ldd`)
- [x] Claude Code stdio MCP verified against a local knowledge base

## Roadmap

1. Multi-arch release binaries (linux/amd64 first)
2. Optional musl / static OpenSSL builds for fewer system deps
3. macOS / Windows support as needed
4. Incremental re-index (per-file) instead of full ReplaceAll
