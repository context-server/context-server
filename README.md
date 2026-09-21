<p align="center">
  <img src="assets/logo.png" alt="context-server" width="144">
</p>

<h1 align="center">context-server</h1>

<p align="center">
  <a href="https://github.com/context-server/context-server/actions/workflows/ci.yml"><img src="https://github.com/context-server/context-server/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/context-server/"><img src="https://img.shields.io/pypi/v/context-server" alt="PyPI"></a>
</p>

<p align="center">
  Semantic search over a folder of markdown, served as an
  <a href="https://modelcontextprotocol.io/">MCP</a> server for coding agents.
</p>

Index once into a SQLite DB (embeddings + BM25). Point Claude Code, Cursor, or any MCP client at `serve`, and the agent can search that corpus instead of guessing from memory.

One Rust binary. ONNX Runtime is linked in via [`ort`](https://github.com/pykeio/ort) / [`fastembed`](https://github.com/Anush008/fastembed-rs) — no separate `libonnxruntime` to ship. SQLite is bundled.

## Quick start

```bash
pip install context-server
# or: uvx context-server@latest …

context-server index --input ./docs --db context.db
context-server search --db context.db "how do we handle backports"
context-server serve --db context.db
```

Wheels: Linux x86_64/aarch64 (`manylinux_2_39` / glibc 2.39+, e.g. Ubuntu 24.04+) and macOS Apple Silicon.

The first embedding run downloads the embedding model into
`$XDG_CACHE_HOME/context-server/fastembed/` (or `~/.cache/...`; once, tens of MB).
Override with `FASTEMBED_CACHE_DIR` or `HF_HOME`.

### Non-English corpora: pick a multilingual model

The default model is English-only. On a non-English corpus the dense half of
hybrid search does not merely get weaker — it returns the same few chunks for
unrelated queries, and hybrid then scores *below* plain BM25. Select a
multilingual model with `CONTEXT_SERVER_MODEL`:

```bash
export CONTEXT_SERVER_MODEL=e5-small     # multilingual-e5-small, 384-d
context-server index --input ./docs --db context.db
context-server serve --db context.db     # same variable at search time
```

| key | model | dim |
|---|---|---|
| `bge-small-en` (default) | BGE-small-en-v1.5 | 384 |
| `e5-small` | multilingual-e5-small | 384 |
| `e5-base` | multilingual-e5-base | 768 |
| `paraphrase-ml` | paraphrase-multilingual-MiniLM-L12-v2 | 384 |
| `bge-m3` | BGE-M3 | 1024 |

Each model carries its own retrieval prefixes (BGE instructs the query only;
E5 needs `query: ` / `passage: ` on both sides) and its own fingerprint, so a
database built with one model refuses to be searched with another — switch the
variable and re-run `index`.

Measured on a 4932-chunk Russian corpus, hit@5 over 20 queries (same
chunking, only the model changed):

| mode | `bge-small-en` | `e5-small` | `e5-base` |
|---|---|---|---|
| dense | 3/20 | 16/20 | 18/20 |
| hybrid | 14/20 | 16/20 | **19/20** |
| lexical | 15/20 | 15/20 | 15/20 |

`lexical` is the control: BM25 is untouched, so the whole difference comes
from the dense half. Note that with the English model hybrid scored *below*
lexical — the dense half was subtracting.

Dense results also stopped collapsing onto attractor chunks. Distinct chunks
per 100 dense results: 59 with `bge-small-en`, 97 with `e5-small`, 96 with
`e5-base`. One chunk came back for 9 of 20 unrelated queries with the English
model; the worst repeat with either multilingual model is 2.

### Optional: tell the agent when to use this corpus

```bash
context-server index --input ./docs --db context.db \
  --instructions-file ./mcp-instructions.txt
# or: --instructions 'Use semantic_search for questions about …'
```

That text is stored in the DB and exposed as MCP `ServerInfo.instructions` when you `serve`.

### Claude Code

```bash
claude mcp add --transport stdio --scope user context-server \
  -- uvx --refresh context-server@latest \
  serve --db /absolute/path/to/context.db
```

`--refresh` + `@latest` rechecks PyPI on each start. If Claude rarely surfaces the tools, set `"alwaysLoad": true` on the server entry in your Claude MCP config.

### Cursor

`~/.cursor/mcp.json` (or project `.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "context-server": {
      "command": "uvx",
      "args": [
        "--refresh",
        "context-server@latest",
        "serve",
        "--db",
        "/absolute/path/to/context.db"
      ]
    }
  }
}
```

Reload MCP after editing. Re-index when content changes, then restart the MCP session so `serve` reloads the DB.

## What it indexes

Only `.md` / `.markdown`. Chunks on `#` / `##` / `###`, keeps the heading path on each chunk, and splits long sections with overlap.

Convert structured sources (YAML, etc.) to prose **before** indexing. Fenced YAML searches poorly; a short paragraph that keeps names, roles, and relationships together works much better.

Try the sample set:

```bash
cargo build --release
./target/release/context-server index --input examples/sample-docs --dry-run
./target/release/context-server index --input examples/sample-docs --db /tmp/sample.db
./target/release/context-server search --db /tmp/sample.db "password reset"
```

## Search

Default mode is **hybrid**: dense cosine (the selected embedding model) plus BM25, fused with reciprocal rank fusion. Dense catches paraphrase; BM25 catches exact tokens (usernames, acronyms, IDs).

```bash
context-server search --db context.db --mode hybrid "query"   # default
context-server search --db context.db --mode dense "query"
context-server search --db context.db --mode lexical "query"

# Scope to a subtree / heading / metadata tag
context-server search --db context.db --path-prefix teams/ "who owns storage"
context-server search --db context.db --heading Backport "z-stream"
context-server get --db context.db --path teams/storage.md --chunk 0
```

## MCP tools

| Tool | Role |
|------|------|
| `semantic_search` | Ranked passages + scores; optional `path_prefix` / `heading` / `tag` filters |
| `list_documents` | Indexed chunks; optional `path_prefix` |
| `get_document` | Full chunk by citation (`source_path` + `chunk_index`), or all chunks for a path |

Search hits cite chunks as `source_path#chunk_index`. Call `get_document` to pull the full text for quoting.

## Remote database (GCS)

`serve` and `search` accept a `gs://` URI. The object is cached under `$XDG_CACHE_HOME/context-server/dbs/` (or `~/.cache/...`). `index` still writes a local path only.

```bash
context-server serve --db 'gs://my-bucket/latest/context.db'

# Project-qualified form also works (gs:// required; stripped for the Storage API)
context-server serve --db \
  'gs://projects/my-gcp-project/buckets/my-bucket/objects/latest/context.db'
```

Uses [Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials). If a sibling `{object}.sha256` exists (sha256sum format), a matching local cache is reused; otherwise the DB is re-fetched and verified.

## CLI

```text
context-server index  --input <path> [--db FILE] [--dry-run] [--batch N]
                      [--full] [--sync]
                      [--instructions TEXT | --instructions-file FILE]
context-server serve  --db <local path | gs://…>
context-server search --db <local path | gs://…> [--limit N] [--mode hybrid|dense|lexical]
                      [--path-prefix P] [--heading H] [--tag T] <query>
context-server get    --db <local path | gs://…> --path FILE [--chunk N]
context-server embed  <query>         # smoke-test query embedding (model's query instruction)
```

`index` is upsert-only by default. Use `--sync` only when the database should
exactly mirror the current input: it deletes indexed paths missing from that
input, and an empty input removes every indexed document. The former `--update`
behavior is now the default; replace previous prune-by-default commands with an
explicit `--sync`.

## Build from source

```bash
cargo build --release
cargo test
```

Rust 1.90+, Linux x86_64 is the primary target. You need a C++ stdlib for the linker (`libstdc++`) and whatever OpenSSL/`native-tls` needs on your platform.

On Fedora/RHEL, if the linker wants `-lstdc++` but only `libstdc++.so.6` exists:

```bash
mkdir -p .linker && ln -sfn /usr/lib64/libstdc++.so.6 .linker/libstdc++.so
export RUSTFLAGS="-L native=$(pwd)/.linker"
```

Linux wheels (same image CI uses — Ubuntu 24.04 / glibc 2.39):

```bash
./scripts/build-wheel.sh
VERSION=2026.716.1 ./scripts/build-wheel.sh   # optional override
```

## Releasing

CalVer `YYYY.MMDD.N` (e.g. `2026.716.1`) so versions work for both Cargo and PyPI. Run the **Release** workflow on `main` (Actions UI or CLI); it picks the next version, builds wheels, publishes to PyPI, then creates the matching git tag and GitHub Release (with wheels attached).

```bash
gh workflow run release.yml --repo context-server/context-server
```

## Design notes

Under the hood: fastembed (L2-normalized vectors; per-model query/passage instructions applied at index and search time; BGE-small-en-v1.5 by default, see `CONTEXT_SERVER_MODEL`), rusqlite with float32 blobs, [`rmcp`](https://github.com/modelcontextprotocol/rust-sdk) over stdio. `index` is incremental by file: unchanged files (same post-chunk content hash) are skipped, so the embedding model is not loaded. Indexing safely upserts by default. Pass `--sync` to also remove database paths missing from `--input`, or `--full` to re-embed everything collected. A model or chunker migration requires a complete-corpus `--sync` run.

More detail and roadmap: [PLAN.md](PLAN.md).

### Supported scale

The primary target is up to 10,000 chunks; 50,000 chunks is the regularly
benchmarked upper range for the exact in-memory implementation. Re-evaluate
storage/index architecture around 100,000 chunks or 500 MiB resident memory.
Run `scripts/benchmark-scale.py --source-db context.db` to reproduce structural
latency and database-size measurements.

## License

MIT — see [LICENSE](LICENSE).
