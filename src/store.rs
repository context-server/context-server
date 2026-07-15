//! SQLite storage for chunks and embeddings.

use crate::index::Chunk;
use anyhow::{bail, Context, Result};
use rusqlite::{params, Connection};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Document {
    pub id: i64,
    pub source_path: String,
    pub chunk_index: usize,
    pub text: String,
    pub headings: Vec<String>,
    pub metadata: serde_json::Map<String, serde_json::Value>,
    pub vector: Vec<f32>,
}

pub struct Db {
    conn: Connection,
}

impl Db {
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path).with_context(|| format!("open {}", path.display()))?;
        conn.execute_batch(
            r#"
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY,
  source_path TEXT NOT NULL,
  chunk_index INTEGER NOT NULL,
  text TEXT NOT NULL,
  headings TEXT NOT NULL DEFAULT '[]',
  metadata TEXT NOT NULL DEFAULT '{}',
  UNIQUE(source_path, chunk_index)
);

CREATE TABLE IF NOT EXISTS embeddings (
  id INTEGER PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
  dim INTEGER NOT NULL,
  vector BLOB NOT NULL
);
"#,
        )?;
        Ok(Self { conn })
    }

    pub fn clear(&self) -> Result<()> {
        self.conn.execute_batch("DELETE FROM embeddings; DELETE FROM documents;")?;
        Ok(())
    }

    pub fn replace_all(&mut self, chunks: &[Chunk], vectors: &[Vec<f32>]) -> Result<()> {
        if chunks.len() != vectors.len() {
            bail!(
                "chunks ({}) and vectors ({}) length mismatch",
                chunks.len(),
                vectors.len()
            );
        }
        let tx = self.conn.transaction()?;
        tx.execute_batch("DELETE FROM embeddings; DELETE FROM documents;")?;
        {
            let mut doc_stmt = tx.prepare(
                "INSERT INTO documents (source_path, chunk_index, text, headings, metadata) VALUES (?1, ?2, ?3, ?4, ?5)",
            )?;
            let mut emb_stmt =
                tx.prepare("INSERT INTO embeddings (id, dim, vector) VALUES (?1, ?2, ?3)")?;

            for (c, vec) in chunks.iter().zip(vectors.iter()) {
                if vec.is_empty() {
                    bail!("empty vector for {}[{}]", c.source_path, c.chunk_index);
                }
                let headings = serde_json::to_string(&c.headings)?;
                let metadata = serde_json::to_string(&c.metadata)?;
                doc_stmt.execute(params![
                    c.source_path,
                    c.chunk_index as i64,
                    c.text,
                    headings,
                    metadata
                ])?;
                let id = tx.last_insert_rowid();
                emb_stmt.execute(params![id, vec.len() as i64, float32_to_bytes(vec)])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn count(&self) -> Result<usize> {
        let n: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM documents", [], |r| r.get(0))?;
        Ok(n as usize)
    }

    pub fn list(&self, limit: usize) -> Result<Vec<Document>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, source_path, chunk_index, text, headings, metadata FROM documents ORDER BY source_path, chunk_index LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (id, source_path, chunk_index, text, headings_json, meta_json) = row?;
            out.push(Document {
                id,
                source_path,
                chunk_index: chunk_index as usize,
                text,
                headings: serde_json::from_str(&headings_json).unwrap_or_default(),
                metadata: serde_json::from_str(&meta_json).unwrap_or_default(),
                vector: vec![],
            });
        }
        Ok(out)
    }

    pub fn load_all(&self) -> Result<Vec<Document>> {
        let mut stmt = self.conn.prepare(
            r#"
SELECT d.id, d.source_path, d.chunk_index, d.text, d.headings, d.metadata, e.dim, e.vector
FROM documents d
JOIN embeddings e ON e.id = d.id
ORDER BY d.id
"#,
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, i64>(6)?,
                row.get::<_, Vec<u8>>(7)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (id, source_path, chunk_index, text, headings_json, meta_json, dim, blob) = row?;
            let vector = bytes_to_float32(&blob, dim as usize)?;
            out.push(Document {
                id,
                source_path,
                chunk_index: chunk_index as usize,
                text,
                headings: serde_json::from_str(&headings_json).unwrap_or_default(),
                metadata: serde_json::from_str(&meta_json).unwrap_or_default(),
                vector,
            });
        }
        Ok(out)
    }

    pub fn summary(&self) -> Result<String> {
        let n = self.count()?;
        let mut stmt = self
            .conn
            .prepare("SELECT DISTINCT source_path FROM documents ORDER BY source_path")?;
        let sources: Vec<String> = stmt
            .query_map([], |r| r.get(0))?
            .filter_map(|r| r.ok())
            .collect();
        let shown: Vec<&str> = sources.iter().take(5).map(|s| s.as_str()).collect();
        let extra = if sources.len() > 5 { ", …" } else { "" };
        Ok(format!(
            "{} chunks across {} files ({}{})",
            n,
            sources.len(),
            shown.join(", "),
            extra
        ))
    }
}

fn float32_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for f in v {
        b.extend_from_slice(&f.to_le_bytes());
    }
    b
}

fn bytes_to_float32(b: &[u8], dim: usize) -> Result<Vec<f32>> {
    if b.len() != dim * 4 {
        bail!("blob length {} != dim*4 ({})", b.len(), dim * 4);
    }
    let mut out = Vec::with_capacity(dim);
    for chunk in b.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::Chunk;
    use tempfile::tempdir;

    #[test]
    fn replace_and_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("t.db");
        let mut db = Db::open(&path).unwrap();
        let chunks = vec![Chunk {
            source_path: "a.md".into(),
            chunk_index: 0,
            text: "hello".into(),
            headings: vec!["H".into()],
            metadata: serde_json::Map::new(),
        }];
        let vectors = vec![vec![1.0f32, 0.0, 0.0]];
        db.replace_all(&chunks, &vectors).unwrap();
        assert_eq!(db.count().unwrap(), 1);
        let docs = db.load_all().unwrap();
        assert_eq!(docs[0].text, "hello");
        assert_eq!(docs[0].vector, vec![1.0, 0.0, 0.0]);
    }
}
