//! Hybrid dense + BM25 search over loaded documents.

use crate::bm25::{reciprocal_rank_fusion, Bm25};
use crate::embed::{self, Embedder};
use crate::store::{Db, Document};
use anyhow::{Context, Result};

const RRF_K: f32 = 60.0;
/// How many candidates to take from each ranker before fusion.
const CANDIDATE_POOL: usize = 50;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    Hybrid,
    Dense,
    Lexical,
}

impl SearchMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "hybrid" | "both" => Some(Self::Hybrid),
            "dense" | "semantic" | "vector" => Some(Self::Dense),
            "lexical" | "bm25" | "text" => Some(Self::Lexical),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ResultHit {
    pub score: f32,
    pub dense_score: f32,
    pub lexical_score: f32,
    pub source_path: String,
    pub chunk_index: usize,
    pub headings: Vec<String>,
    pub text: String,
}

pub struct Index {
    docs: Vec<Document>,
    bm25: Bm25,
}

impl Index {
    pub fn load(db: &Db) -> Result<Self> {
        let docs = db.load_all().context("load documents")?;
        let texts: Vec<String> = docs.iter().map(|d| d.text.clone()).collect();
        let bm25 = Bm25::build(&texts);
        Ok(Self { docs, bm25 })
    }

    pub fn len(&self) -> usize {
        self.docs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    pub fn query(
        &self,
        emb: &mut Embedder,
        query: &str,
        limit: usize,
        mode: SearchMode,
    ) -> Result<Vec<ResultHit>> {
        let limit = if limit == 0 { 5 } else { limit };
        if self.docs.is_empty() {
            return Ok(vec![]);
        }

        let dense_scores: Vec<f32> = match mode {
            SearchMode::Lexical => vec![0.0; self.docs.len()],
            SearchMode::Dense | SearchMode::Hybrid => {
                let qv = emb.embed(query)?;
                self.docs
                    .iter()
                    .map(|d| embed::cosine(&qv, &d.vector))
                    .collect()
            }
        };
        let lexical_scores: Vec<f32> = match mode {
            SearchMode::Dense => vec![0.0; self.docs.len()],
            SearchMode::Lexical | SearchMode::Hybrid => self.bm25.scores(query),
        };

        let hits = match mode {
            SearchMode::Dense => rank_by_scores(&self.docs, &dense_scores, &lexical_scores, limit),
            SearchMode::Lexical => {
                rank_by_scores(&self.docs, &lexical_scores, &dense_scores, limit)
            }
            SearchMode::Hybrid => hybrid_rrf(&self.docs, &dense_scores, &lexical_scores, limit),
        };
        Ok(hits)
    }

    #[allow(dead_code)] // used by unit tests; kept for dense-only callers
    pub fn query_vector(&self, qv: &[f32], limit: usize) -> Vec<ResultHit> {
        let dense: Vec<f32> = self
            .docs
            .iter()
            .map(|d| embed::cosine(qv, &d.vector))
            .collect();
        let lexical = vec![0.0; self.docs.len()];
        rank_by_scores(&self.docs, &dense, &lexical, limit)
    }
}

fn top_ids(scores: &[f32], pool: usize) -> Vec<usize> {
    let mut idxs: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    idxs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    idxs.into_iter()
        .filter(|(_, s)| *s > 0.0)
        .take(pool)
        .map(|(i, _)| i)
        .collect()
}

fn hybrid_rrf(docs: &[Document], dense: &[f32], lexical: &[f32], limit: usize) -> Vec<ResultHit> {
    let pool = CANDIDATE_POOL.max(limit);
    let dense_rank = top_ids(dense, pool);
    let lex_rank = top_ids(lexical, pool);
    // If one side is all zeros, fall back to the other.
    let fused = if dense_rank.is_empty() {
        lex_rank.into_iter().map(|i| (i, lexical[i])).collect()
    } else if lex_rank.is_empty() {
        dense_rank.into_iter().map(|i| (i, dense[i])).collect()
    } else {
        reciprocal_rank_fusion(&[dense_rank, lex_rank], RRF_K)
    };
    fused
        .into_iter()
        .take(limit)
        .map(|(i, score)| hit_from(docs, i, score, dense[i], lexical[i]))
        .collect()
}

fn rank_by_scores(
    docs: &[Document],
    primary: &[f32],
    secondary: &[f32],
    limit: usize,
) -> Vec<ResultHit> {
    let mut idxs: Vec<usize> = (0..docs.len()).collect();
    idxs.sort_by(|a, b| {
        primary[*b]
            .partial_cmp(&primary[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idxs.truncate(limit.min(docs.len()));
    idxs.into_iter()
        .map(|i| hit_from(docs, i, primary[i], primary[i], secondary[i]))
        .collect()
}

fn hit_from(
    docs: &[Document],
    i: usize,
    score: f32,
    dense_score: f32,
    lexical_score: f32,
) -> ResultHit {
    let d = &docs[i];
    ResultHit {
        score,
        dense_score,
        lexical_score,
        source_path: d.source_path.clone(),
        chunk_index: d.chunk_index,
        headings: d.headings.clone(),
        text: d.text.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Document;

    fn doc(id: i64, text: &str, vector: Vec<f32>) -> Document {
        Document {
            id,
            source_path: format!("{id}.md"),
            chunk_index: 0,
            text: text.into(),
            headings: vec![],
            metadata: Default::default(),
            vector,
        }
    }

    #[test]
    fn ranks_by_cosine() {
        let docs = vec![
            doc(1, "dogs", vec![1.0, 0.0, 0.0]),
            doc(2, "cats", vec![0.0, 1.0, 0.0]),
        ];
        let texts: Vec<String> = docs.iter().map(|d| d.text.clone()).collect();
        let idx = Index {
            bm25: Bm25::build(&texts),
            docs,
        };
        let hits = idx.query_vector(&[0.9, 0.1, 0.0], 2);
        assert_eq!(hits[0].text, "dogs");
    }

    #[test]
    fn hybrid_surfaces_exact_id() {
        // Dense would prefer the storage vector; lexical should pull the username doc up.
        let docs = vec![
            doc(
                1,
                "storage team owns persistent volumes and ceph",
                vec![1.0, 0.0, 0.0],
            ),
            doc(
                2,
                "jsmith manages platform virtualization",
                vec![0.0, 1.0, 0.0],
            ),
        ];
        let texts: Vec<String> = docs.iter().map(|d| d.text.clone()).collect();
        let idx = Index {
            bm25: Bm25::build(&texts),
            docs,
        };
        let dense = vec![0.95f32, 0.10]; // favors doc 0
        let lexical = idx.bm25.scores("jsmith");
        let hits = hybrid_rrf(&idx.docs, &dense, &lexical, 2);
        assert!(hits[0].text.contains("jsmith"), "{hits:?}");
    }
}
