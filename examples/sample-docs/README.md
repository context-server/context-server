# Sample docs

Tiny markdown set for a dry-run:

```bash
cargo build --release
./target/release/context-server index --input examples/sample-docs --dry-run
./target/release/context-server index --input examples/sample-docs --db /tmp/sample.db
./target/release/context-server search --db /tmp/sample.db "password reset"
```
