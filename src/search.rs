//! Brute-force cosine search over loaded documents.

use crate::embed::{self, Embedder};
use crate::store::{Db, Document};
use anyhow::{Context, Result};

#[derive(Debug, Clone)]
pub struct ResultHit {
    pub score: f32,
    pub source_path: String,
    pub chunk_index: usize,
    pub headings: Vec<String>,
    pub text: String,
}

pub struct Index {
    docs: Vec<Document>,
}

impl Index {
    pub fn load(db: &Db) -> Result<Self> {
        let docs = db.load_all().context("load documents")?;
        Ok(Self { docs })
    }

    pub fn len(&self) -> usize {
        self.docs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    pub fn query(&self, emb: &mut Embedder, query: &str, limit: usize) -> Result<Vec<ResultHit>> {
        let limit = if limit == 0 { 5 } else { limit };
        if self.docs.is_empty() {
            return Ok(vec![]);
        }
        let qv = emb.embed(query)?;
        Ok(self.query_vector(&qv, limit))
    }

    pub fn query_vector(&self, qv: &[f32], limit: usize) -> Vec<ResultHit> {
        let limit = if limit == 0 { 5 } else { limit };
        let mut scored: Vec<(usize, f32)> = self
            .docs
            .iter()
            .enumerate()
            .map(|(i, d)| (i, embed::cosine(qv, &d.vector)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit.min(scored.len()));
        scored
            .into_iter()
            .map(|(i, score)| {
                let d = &self.docs[i];
                ResultHit {
                    score,
                    source_path: d.source_path.clone(),
                    chunk_index: d.chunk_index,
                    headings: d.headings.clone(),
                    text: d.text.clone(),
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Document;

    #[test]
    fn ranks_by_cosine() {
        let idx = Index {
            docs: vec![
                Document {
                    id: 1,
                    source_path: "a.md".into(),
                    chunk_index: 0,
                    text: "dogs".into(),
                    headings: vec![],
                    metadata: Default::default(),
                    vector: vec![1.0, 0.0, 0.0],
                },
                Document {
                    id: 2,
                    source_path: "b.md".into(),
                    chunk_index: 0,
                    text: "cats".into(),
                    headings: vec![],
                    metadata: Default::default(),
                    vector: vec![0.0, 1.0, 0.0],
                },
            ],
        };
        let hits = idx.query_vector(&[0.9, 0.1, 0.0], 2);
        assert_eq!(hits[0].text, "dogs");
    }
}
