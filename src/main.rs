mod bm25;
mod embed;
mod index;
mod mcp;
mod remote;
mod search;
mod store;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use search::SearchMode;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "context-server", about = "Semantic search MCP server for markdown knowledge bases")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Index markdown into the SQLite database
    Index {
        /// Markdown file or directory
        #[arg(long)]
        input: PathBuf,
        /// Local SQLite database path
        #[arg(long, default_value = "context.db")]
        db: PathBuf,
        /// Chunk and print without embedding
        #[arg(long)]
        dry_run: bool,
        /// Embedding batch size
        #[arg(long, default_value_t = 16)]
        batch: usize,
        /// MCP server instructions stored in DB meta (when to call this corpus)
        #[arg(long)]
        instructions: Option<String>,
        /// Read MCP instructions from a UTF-8 text file
        #[arg(long)]
        instructions_file: Option<PathBuf>,
    },
    /// Start the MCP server (stdio)
    Serve {
        /// Local path, or `gs://bucket/object` /
        /// `gs://projects/PROJECT/buckets/BUCKET/objects/OBJECT`
        #[arg(long, default_value = "context.db")]
        db: String,
    },
    /// Search the database (CLI)
    Search {
        /// Local path, or `gs://bucket/object` /
        /// `gs://projects/PROJECT/buckets/BUCKET/objects/OBJECT`
        #[arg(long, default_value = "context.db")]
        db: String,
        #[arg(long, default_value_t = 5)]
        limit: usize,
        /// Search mode: hybrid (default), dense, or lexical
        #[arg(long, default_value = "hybrid")]
        mode: String,
        /// Query text
        query: Vec<String>,
    },
    /// Embed a string (smoke test)
    Embed {
        text: Vec<String>,
    },
}

fn main() -> Result<()> {
    // google-cloud-storage / reqwest enable both rustls aws-lc-rs and ring features;
    // pick an explicit process default before any TLS.
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();

    let cli = Cli::parse();
    match cli.command {
        Commands::Index {
            input,
            db,
            dry_run,
            batch,
            instructions,
            instructions_file,
        } => run_index(input, db, dry_run, batch, instructions, instructions_file),
        Commands::Serve { db } => {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(run_serve(db))
        }
        Commands::Search {
            db,
            limit,
            mode,
            query,
        } => run_search(db, limit, mode, query),
        Commands::Embed { text } => run_embed(text),
    }
}

fn run_index(
    input: PathBuf,
    db_path: PathBuf,
    dry_run: bool,
    batch: usize,
    instructions: Option<String>,
    instructions_file: Option<PathBuf>,
) -> Result<()> {
    let chunks = index::collect(&input)?;
    if chunks.is_empty() {
        bail!("no markdown chunks found under {}", input.display());
    }
    if dry_run {
        println!("chunked {} pieces from {}", chunks.len(), input.display());
        for c in &chunks {
            println!("  {}: {}", c.source_path, index::format_chunk_debug(c));
        }
        return Ok(());
    }

    let instructions = match (instructions, instructions_file) {
        (Some(_), Some(_)) => {
            bail!("pass only one of --instructions or --instructions-file");
        }
        (Some(text), None) => Some(text),
        (None, Some(path)) => {
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("read instructions file {}", path.display()))?;
            Some(text.trim().to_string())
        }
        (None, None) => None,
    };
    if let Some(ref text) = instructions {
        if text.is_empty() {
            bail!("instructions text is empty");
        }
    }

    println!(
        "indexing {} chunks from {} -> {} ({})",
        chunks.len(),
        input.display(),
        db_path.display(),
        embed::MODEL_ID
    );
    let mut emb = embed::Embedder::new()?;
    let batch = batch.max(1);
    let mut vectors = Vec::with_capacity(chunks.len());
    for i in (0..chunks.len()).step_by(batch) {
        let end = (i + batch).min(chunks.len());
        eprintln!("  embedding {}-{}/{}", i + 1, end, chunks.len());
        let texts: Vec<String> = chunks[i..end].iter().map(|c| c.text.clone()).collect();
        let batch_vecs = emb.embed_batch(&texts)?;
        vectors.extend(batch_vecs);
    }

    let mut db = store::Db::open(&db_path)?;
    db.replace_all(&chunks, &vectors, instructions.as_deref())?;
    if let Some(text) = db.get_meta("instructions")? {
        eprintln!(
            "MCP instructions ({} chars): {}",
            text.len(),
            text.chars().take(80).collect::<String>()
        );
    }
    println!("wrote {}", db.summary()?);
    Ok(())
}

async fn run_serve(db_spec: String) -> Result<()> {
    use rmcp::ServiceExt;

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();

    let db_path = remote::resolve_db(&db_spec).await?;
    let db = store::Db::open(&db_path)?;
    let n = db.count()?;
    if n == 0 {
        bail!(
            "database {} has no documents; run index first",
            db_path.display()
        );
    }
    let index = search::Index::load(&db)?;
    let embedder = embed::Embedder::new()?;
    let service = mcp::ContextService::new(db, index, embedder);

    eprintln!(
        "context-server: serving MCP stdio ({} chunks from {}, hybrid search, {})",
        n,
        db_path.display(),
        embed::MODEL_ID
    );
    let server = service.serve(rmcp::transport::stdio()).await?;
    server.waiting().await?;
    Ok(())
}

fn run_search(db_spec: String, limit: usize, mode: String, query: Vec<String>) -> Result<()> {
    let q = query.join(" ").trim().to_string();
    if q.is_empty() {
        bail!("usage: context-server search --db context.db <query>");
    }
    let mode = SearchMode::parse(&mode).ok_or_else(|| {
        anyhow::anyhow!("unknown --mode {mode:?} (expected hybrid, dense, or lexical)")
    })?;
    let db_path = remote::resolve_db_blocking(&db_spec)?;
    let db = store::Db::open(&db_path)?;
    let idx = search::Index::load(&db)?;
    if idx.is_empty() {
        bail!("database {} has no documents", db_path.display());
    }
    let mut emb = embed::Embedder::new()?;
    let hits = idx.query(&mut emb, &q, limit, mode)?;
    println!(
        "query={q:?} mode={mode:?} ({} indexed chunks)",
        idx.len()
    );
    for (i, h) in hits.iter().enumerate() {
        let mut preview = h.text.clone();
        if preview.len() > 240 {
            preview = format!("{}...", &preview[..237]);
        }
        preview = preview.replace('\n', " ");
        println!(
            "\n{}. score={:.4} (dense={:.4} lexical={:.4})  {}#{}\n   {}",
            i + 1,
            h.score,
            h.dense_score,
            h.lexical_score,
            h.source_path,
            h.chunk_index,
            preview
        );
    }
    Ok(())
}

fn run_embed(text: Vec<String>) -> Result<()> {
    let t = text.join(" ").trim().to_string();
    if t.is_empty() {
        bail!("usage: context-server embed <text>");
    }
    let mut emb = embed::Embedder::new().context("init embedder")?;
    let vec = emb.embed(&t)?;
    print!("dim={} text={t:?}\nfirst8=[", vec.len());
    for (i, v) in vec.iter().take(8).enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{v:.6}");
    }
    println!("]");

    let base = "The dog is running in the park";
    let similar = "A canine is running through the park";
    let other = "I love eating pizza for dinner";
    let vecs = emb.embed_batch(&[base.into(), similar.into(), other.into()])?;
    println!(
        "cosine({base:?}, {similar:?}) = {:.4}",
        embed::cosine(&vecs[0], &vecs[1])
    );
    println!(
        "cosine({base:?}, {other:?}) = {:.4}",
        embed::cosine(&vecs[0], &vecs[2])
    );
    Ok(())
}
