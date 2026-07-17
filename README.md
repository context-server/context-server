# context-server

Semantic search over a folder of markdown, served as an [MCP](https://modelcontextprotocol.io/) server for coding agents.

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

The first embedding run downloads All-MiniLM-L6-v2 into the local Hugging Face / fastembed cache (once, tens of MB).

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

Default mode is **hybrid**: dense cosine (MiniLM) plus BM25, fused with reciprocal rank fusion. Dense catches paraphrase; BM25 catches exact tokens (usernames, acronyms, IDs).

```bash
context-server search --db context.db --mode hybrid "query"   # default
context-server search --db context.db --mode dense "query"
context-server search --db context.db --mode lexical "query"
```

## MCP tools

| Tool | Role |
|------|------|
| `semantic_search` | Ranked passages + scores |
| `list_documents` | Indexed chunks |
| `answer_question` | Best matching passage(s) — retrieval only, not generative QA |

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
                      [--instructions TEXT | --instructions-file FILE]
context-server serve  --db <local path | gs://…>
context-server search --db <local path | gs://…> [--limit N] [--mode hybrid|dense|lexical] <query>
context-server embed  <text>          # smoke-test embeddings
```

## Build from source

```bash
cargo build --release
cargo test
```

Rust 1.75+, Linux x86_64 is the primary target. You need a C++ stdlib for the linker (`libstdc++`) and whatever OpenSSL/`native-tls` needs on your platform.

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

CalVer `YYYY.MMDD.N` (e.g. `2026.716.1`) so versions work for both Cargo and PyPI. `pyproject.toml` reads the version from `Cargo.toml`; release CI rewrites that from the git tag.

```bash
tag="$(./scripts/next-calver.sh)"
git tag -a "$tag" -m "$tag"
git push origin "$tag"    # Release workflow → PyPI
```

## Design notes

Under the hood: fastembed All-MiniLM-L6-v2 (384-d, L2-normalized), rusqlite with float32 blobs, [`rmcp`](https://github.com/modelcontextprotocol/rust-sdk) over stdio. Each `index` run replaces the DB contents.

More detail and roadmap: [PLAN.md](PLAN.md).

## License

MIT — see [LICENSE](LICENSE).
