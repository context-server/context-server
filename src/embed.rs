//! Sentence embeddings via fastembed (ONNX Runtime).
//!
//! The model is selected at startup from `CONTEXT_SERVER_MODEL` (default:
//! `bge-small-en`). Every model carries its own retrieval prefixes and a
//! fingerprint, so an index built with one model refuses to be searched with
//! another.

use anyhow::{bail, Context, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::path::PathBuf;
use std::sync::OnceLock;

/// Everything about a model that affects vector compatibility.
#[derive(Debug)]
pub struct ModelSpec {
    /// Short name accepted in `CONTEXT_SERVER_MODEL`.
    pub key: &'static str,
    /// Stored in the DB so we refuse to search against an incompatible index.
    pub id: &'static str,
    pub dim: usize,
    pub model: EmbeddingModel,
    /// Retrieval instruction prefixed to queries only.
    pub query_prefix: &'static str,
    /// Retrieval instruction prefixed to passages only.
    pub passage_prefix: &'static str,
    /// Persisted provenance for every behavior that affects vector compatibility.
    pub fingerprint: &'static str,
}

/// BGE prefixes queries only; E5 prefixes both sides. Getting this wrong costs
/// more retrieval quality than the model choice itself.
/// See https://huggingface.co/BAAI/bge-small-en-v1.5 and
/// https://huggingface.co/intfloat/multilingual-e5-small
pub const MODELS: &[ModelSpec] = &[
    ModelSpec {
        key: "bge-small-en",
        id: "BGESmallENV15",
        dim: 384,
        model: EmbeddingModel::BGESmallENV15,
        query_prefix: "Represent this sentence for searching relevant passages: ",
        passage_prefix: "",
        fingerprint: "v1|fastembed:BGESmallENV15|dim:384|pool:model-default|l2:true|query:bge-v1.5",
    },
    ModelSpec {
        key: "e5-small",
        id: "MultilingualE5Small",
        dim: 384,
        model: EmbeddingModel::MultilingualE5Small,
        query_prefix: "query: ",
        passage_prefix: "passage: ",
        fingerprint: "v1|fastembed:MultilingualE5Small|dim:384|pool:model-default|l2:true|query:e5",
    },
    ModelSpec {
        key: "e5-base",
        id: "MultilingualE5Base",
        dim: 768,
        model: EmbeddingModel::MultilingualE5Base,
        query_prefix: "query: ",
        passage_prefix: "passage: ",
        fingerprint: "v1|fastembed:MultilingualE5Base|dim:768|pool:model-default|l2:true|query:e5",
    },
    ModelSpec {
        key: "paraphrase-ml",
        id: "ParaphraseMLMiniLML12V2",
        dim: 384,
        model: EmbeddingModel::ParaphraseMLMiniLML12V2,
        query_prefix: "",
        passage_prefix: "",
        fingerprint:
            "v1|fastembed:ParaphraseMLMiniLML12V2|dim:384|pool:model-default|l2:true|query:none",
    },
    ModelSpec {
        key: "bge-m3",
        id: "BGEM3",
        dim: 1024,
        model: EmbeddingModel::BGEM3,
        query_prefix: "",
        passage_prefix: "",
        fingerprint: "v1|fastembed:BGEM3|dim:1024|pool:model-default|l2:true|query:none",
    },
];

pub const DEFAULT_MODEL_KEY: &str = "bge-small-en";
const MODEL_ENV: &str = "CONTEXT_SERVER_MODEL";

static SPEC: OnceLock<&'static ModelSpec> = OnceLock::new();

/// Resolve once per process; a bad name is a hard error, never a silent default.
pub fn spec() -> &'static ModelSpec {
    SPEC.get_or_init(|| match resolve(std::env::var(MODEL_ENV).ok().as_deref()) {
        Ok(spec) => spec,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(2);
        }
    })
}

pub fn resolve(key: Option<&str>) -> Result<&'static ModelSpec> {
    let key = key.map(str::trim).filter(|k| !k.is_empty());
    let want = key.unwrap_or(DEFAULT_MODEL_KEY);
    MODELS
        .iter()
        .find(|m| m.key.eq_ignore_ascii_case(want) || m.id.eq_ignore_ascii_case(want))
        .map_or_else(
            || {
                let known: Vec<&str> = MODELS.iter().map(|m| m.key).collect();
                bail!(
                    "unknown {MODEL_ENV}={want:?}; known models: {}",
                    known.join(", ")
                )
            },
            Ok,
        )
}

pub fn model_id() -> &'static str {
    spec().id
}

pub fn dim() -> usize {
    spec().dim
}

pub fn embedding_fingerprint() -> &'static str {
    spec().fingerprint
}

pub struct Embedder {
    model: TextEmbedding,
}

impl Embedder {
    pub fn new() -> Result<Self> {
        let spec = spec();
        let cache_dir = model_cache_dir()?;
        std::fs::create_dir_all(&cache_dir)
            .with_context(|| format!("create model cache dir {}", cache_dir.display()))?;
        let model = TextEmbedding::try_new(
            InitOptions::new(spec.model.clone())
                .with_cache_dir(cache_dir)
                .with_show_download_progress(true),
        )
        .with_context(|| format!("load embedding model ({})", spec.id))?;
        Ok(Self { model })
    }

    /// Embed a search query (applies the model's query instruction).
    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        let instructed = format!("{}{text}", spec().query_prefix);
        let mut out = self.model.embed(vec![instructed], None).context("embed")?;
        let mut v = out.pop().context("empty embedding")?;
        l2_normalize(&mut v);
        Ok(v)
    }

    /// Embed document passages (applies the model's passage instruction).
    pub fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        let prefix = spec().passage_prefix;
        let mut out = if prefix.is_empty() {
            self.model.embed(texts, None).context("embed batch")?
        } else {
            let prefixed: Vec<String> = texts.iter().map(|t| format!("{prefix}{t}")).collect();
            self.model.embed(prefixed, None).context("embed batch")?
        };
        for v in &mut out {
            l2_normalize(v);
        }
        Ok(out)
    }
}

/// Prefer explicit env overrides, otherwise use the XDG cache (not the process cwd).
/// fastembed's default is `.fastembed_cache` in PWD, which pollutes project trees.
pub fn model_cache_dir() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("FASTEMBED_CACHE_DIR") {
        return Ok(PathBuf::from(p));
    }
    if let Ok(p) = std::env::var("HF_HOME") {
        return Ok(PathBuf::from(p));
    }
    Ok(dirs::cache_dir()
        .context("no cache directory (set XDG_CACHE_HOME or HOME)")?
        .join("context-server")
        .join("fastembed"))
}

pub fn model_is_cached() -> Result<bool> {
    let dir = model_cache_dir()?;
    if !dir.is_dir() {
        return Ok(false);
    }
    Ok(walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(std::result::Result::ok)
        .any(|entry| entry.file_type().is_file() && entry.file_name() == "model.onnx"))
}

pub fn estimated_tokens(text: &str) -> usize {
    text.split_whitespace()
        .map(|word| word.chars().count().max(1).div_ceil(4))
        .sum::<usize>()
        .max(1)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_defaults_to_bge_small_en() {
        let spec = resolve(None).expect("default model");
        assert_eq!(spec.key, DEFAULT_MODEL_KEY);
        assert_eq!(spec.id, "BGESmallENV15");
        assert_eq!(spec.dim, 384);
    }

    #[test]
    fn resolve_accepts_key_or_model_id_case_insensitively() {
        assert_eq!(
            resolve(Some("e5-small")).expect("by key").id,
            "MultilingualE5Small"
        );
        assert_eq!(
            resolve(Some("E5-Small")).expect("mixed case").id,
            "MultilingualE5Small"
        );
        assert_eq!(
            resolve(Some("MultilingualE5Small")).expect("by id").key,
            "e5-small"
        );
    }

    #[test]
    fn resolve_treats_blank_as_unset_and_rejects_unknown() {
        assert_eq!(resolve(Some("  ")).expect("blank").key, DEFAULT_MODEL_KEY);
        let err = resolve(Some("gpt-embeddings")).expect_err("unknown model");
        let msg = err.to_string();
        assert!(msg.contains("gpt-embeddings"), "{msg}");
        assert!(msg.contains("e5-small"), "should list known models: {msg}");
    }

    /// A shared fingerprint across models would let one model's vectors be
    /// searched with another's query embeddings.
    #[test]
    fn every_model_has_a_distinct_id_key_and_fingerprint() {
        for (i, a) in MODELS.iter().enumerate() {
            assert!(
                a.fingerprint.contains(a.id),
                "{} fingerprint must name the model",
                a.key
            );
            assert!(
                a.fingerprint.contains(&format!("dim:{}", a.dim)),
                "{} fingerprint must name the dim",
                a.key
            );
            for b in MODELS.iter().skip(i + 1) {
                assert_ne!(a.key, b.key);
                assert_ne!(a.id, b.id);
                assert_ne!(a.fingerprint, b.fingerprint);
            }
        }
    }

    #[test]
    fn model_cache_dir_defaults_under_xdg_cache() {
        if std::env::var_os("FASTEMBED_CACHE_DIR").is_some()
            || std::env::var_os("HF_HOME").is_some()
        {
            return;
        }
        let dir = model_cache_dir().expect("cache dir");
        assert!(dir.is_absolute(), "cache dir should be absolute: {dir:?}");
        assert_eq!(dir.file_name().and_then(|s| s.to_str()), Some("fastembed"));
        assert!(
            dir.components().any(|c| c.as_os_str() == "context-server"),
            "unexpected cache dir: {dir:?}"
        );
    }
}
