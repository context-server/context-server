//! Sentence embeddings via fastembed (ONNX Runtime, All-MiniLM-L6-v2).

use anyhow::{Context, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

/// Stored in the DB so we refuse to search against an incompatible index.
pub const MODEL_ID: &str = "AllMiniLML6V2";
pub const DIM: usize = 384;

pub struct Embedder {
    model: TextEmbedding,
}

impl Embedder {
    pub fn new() -> Result<Self> {
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true),
        )
        .context("load embedding model (AllMiniLML6V2)")?;
        Ok(Self { model })
    }

    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        let mut out = self
            .model
            .embed(vec![text.to_string()], None)
            .context("embed")?;
        let mut v = out.pop().context("empty embedding")?;
        l2_normalize(&mut v);
        Ok(v)
    }

    pub fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        let mut out = self
            .model
            .embed(texts.to_vec(), None)
            .context("embed batch")?;
        for v in &mut out {
            l2_normalize(v);
        }
        Ok(out)
    }
}

fn l2_normalize(v: &mut [f32]) {
    let mut sum = 0.0f32;
    for x in v.iter() {
        sum += x * x;
    }
    let norm = sum.sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Cosine similarity for L2-normalized vectors (dot product).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut sum = 0.0f32;
    for i in 0..n {
        sum += a[i] * b[i];
    }
    sum
}
