//! Exact lookup of explicitly configured JSON artifacts, separate from search.

use anyhow::{ensure, Context, Result};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::{fs::File, io::Read, path::Path};

#[derive(Default)]
pub struct Artifacts(Map<String, Value>);

impl Artifacts {
    pub fn load(path: &Path) -> Result<Self> {
        const MAX_BYTES: u64 = 1024 * 1024;
        let mut bytes = Vec::new();
        File::open(path)
            .with_context(|| format!("open artifact manifest {}", path.display()))?
            .take(MAX_BYTES + 1)
            .read_to_end(&mut bytes)?;
        ensure!(
            bytes.len() as u64 <= MAX_BYTES,
            "artifact manifest exceeds 1 MiB"
        );
        let artifacts: Map<String, Value> = serde_json::from_slice(&bytes)
            .with_context(|| format!("parse artifact manifest {}", path.display()))?;
        for (name, value) in &artifacts {
            ensure!(
                !name.trim().is_empty() && value.is_object(),
                "expected a named JSON object: {name:?}"
            );
        }
        Ok(Self(artifacts))
    }

    pub fn get(&self, name: &str) -> Option<Value> {
        let artifact = self.0.get(name)?;
        let digest = Sha256::digest(artifact.to_string().as_bytes());
        Some(
            json!({"name": name, "digest": format!("sha256:{}", hex::encode(digest)), "artifact": artifact}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_lookup_with_per_artifact_digest() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("artifacts.json");
        std::fs::write(
            &path,
            r#"{"rubrics/example":{"rules":["one","two"]},"unrelated":{}}"#,
        )
        .unwrap();
        let first = Artifacts::load(&path)
            .unwrap()
            .get("rubrics/example")
            .unwrap();
        assert_eq!(first["artifact"]["rules"], json!(["one", "two"]));
        std::fs::write(
            &path,
            r#"{ "rubrics/example": {"rules":["one","two"]}, "unrelated":{"changed":true}}"#,
        )
        .unwrap();
        let artifacts = Artifacts::load(&path).unwrap();
        assert_eq!(
            artifacts.get("rubrics/example").unwrap()["digest"],
            first["digest"]
        );
        assert!(artifacts.get("example").is_none());
        assert!(artifacts.get("../artifacts.json").is_none());
        std::fs::write(&path, r#"{"rubrics/example":{"rules":["two","one"]}}"#).unwrap();
        assert_ne!(
            Artifacts::load(&path)
                .unwrap()
                .get("rubrics/example")
                .unwrap()["digest"],
            first["digest"]
        );
    }

    #[test]
    fn rejects_invalid_manifests() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("artifacts.json");
        for invalid in ["not JSON", "[]", r#"{"missing":null}"#, r#"{"":{}}"#] {
            std::fs::write(&path, invalid).unwrap();
            assert!(Artifacts::load(&path).is_err(), "accepted {invalid}");
        }
    }
}
