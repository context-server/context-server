//! Markdown chunking and directory collection.

use anyhow::{bail, Context, Result};
use regex::Regex;
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

/// Soft cap on embedded text length. MiniLM truncates around ~256 tokens;
/// ~4 chars/token ⇒ keep chunks under this so the tail is not dropped.
pub const MAX_CHUNK_CHARS: usize = 900;
/// Overlap between consecutive splits of an oversized section.
pub const CHUNK_OVERLAP_CHARS: usize = 150;

#[derive(Debug, Clone)]
pub struct Chunk {
    pub source_path: String,
    pub chunk_index: usize,
    pub text: String,
    pub headings: Vec<String>,
    pub metadata: serde_json::Map<String, serde_json::Value>,
}
/// Split markdown into chunks on ## and ### boundaries.
pub fn split_markdown(source_path: &str, content: &str) -> Vec<Chunk> {
    let content = strip_front_matter(content);
    let heading_re = Regex::new(r"^(#{1,6})\s+(.+?)\s*$").unwrap();

    let mut doc_title = String::new();
    let mut h2 = String::new();
    let mut h3 = String::new();
    let mut body: Vec<String> = Vec::new();
    let mut chunks: Vec<Chunk> = Vec::new();

    let emit = |body: &mut Vec<String>,
                    doc_title: &str,
                    h2: &str,
                    h3: &str,
                    chunks: &mut Vec<Chunk>,
                    source_path: &str| {
        let text = body.join("\n").trim().to_string();
        body.clear();
        if text.is_empty() {
            return;
        }
        let mut headings = Vec::new();
        if !doc_title.is_empty() {
            headings.push(doc_title.to_string());
        }
        if !h2.is_empty() {
            headings.push(h2.to_string());
        }
        if !h3.is_empty() {
            headings.push(h3.to_string());
        }
        let prefixed = if headings.is_empty() {
            text.clone()
        } else {
            format!("{}\n\n{}", headings.join(" > "), text)
        };
        let idx = chunks.len();
        chunks.push(Chunk {
            source_path: source_path.to_string(),
            chunk_index: idx,
            text: prefixed,
            headings,
            metadata: serde_json::Map::new(),
        });
    };

    for line in content.lines() {
        if let Some(caps) = heading_re.captures(line) {
            let level = caps[1].len();
            let title = caps[2].trim().to_string();
            match level {
                1 => {
                    emit(
                        &mut body,
                        &doc_title,
                        &h2,
                        &h3,
                        &mut chunks,
                        source_path,
                    );
                    doc_title = title;
                    h2.clear();
                    h3.clear();
                }
                2 => {
                    emit(
                        &mut body,
                        &doc_title,
                        &h2,
                        &h3,
                        &mut chunks,
                        source_path,
                    );
                    h2 = title;
                    h3.clear();
                }
                3 => {
                    emit(
                        &mut body,
                        &doc_title,
                        &h2,
                        &h3,
                        &mut chunks,
                        source_path,
                    );
                    h3 = title;
                }
                _ => body.push(line.to_string()),
            }
            continue;
        }
        body.push(line.to_string());
    }
    emit(
        &mut body,
        &doc_title,
        &h2,
        &h3,
        &mut chunks,
        source_path,
    );
    split_oversized(chunks)
}

/// Split any chunk whose embedded text exceeds [`MAX_CHUNK_CHARS`], keeping the
/// heading prefix on each piece and overlapping body windows.
fn split_oversized(chunks: Vec<Chunk>) -> Vec<Chunk> {
    let mut out = Vec::new();
    for chunk in chunks {
        if chunk.text.chars().count() <= MAX_CHUNK_CHARS {
            out.push(chunk);
            continue;
        }
        let prefix = if chunk.headings.is_empty() {
            String::new()
        } else {
            format!("{}\n\n", chunk.headings.join(" > "))
        };
        let body = chunk
            .text
            .strip_prefix(&prefix)
            .unwrap_or(chunk.text.as_str());
        let prefix_len = prefix.chars().count();
        let body_budget = MAX_CHUNK_CHARS.saturating_sub(prefix_len).max(200);
        let overlap = CHUNK_OVERLAP_CHARS.min(body_budget / 3);

        let body_chars: Vec<char> = body.chars().collect();
        if body_chars.is_empty() {
            out.push(chunk);
            continue;
        }

        let mut start = 0usize;
        while start < body_chars.len() {
            let mut end = (start + body_budget).min(body_chars.len());
            // Prefer breaking on whitespace when not at the end.
            if end < body_chars.len() {
                if let Some(rel) = body_chars[start..end]
                    .iter()
                    .rposition(|c| c.is_whitespace())
                {
                    if rel > body_budget / 4 {
                        end = start + rel;
                    }
                }
            }
            let piece: String = body_chars[start..end].iter().collect();
            let piece = piece.trim();
            if !piece.is_empty() {
                let text = if prefix.is_empty() {
                    piece.to_string()
                } else {
                    format!("{prefix}{piece}")
                };
                out.push(Chunk {
                    source_path: chunk.source_path.clone(),
                    chunk_index: 0, // renumbered below
                    text,
                    headings: chunk.headings.clone(),
                    metadata: chunk.metadata.clone(),
                });
            }
            if end >= body_chars.len() {
                break;
            }
            let next = end.saturating_sub(overlap);
            start = if next <= start { end } else { next };
        }
    }
    for (i, c) in out.iter_mut().enumerate() {
        c.chunk_index = i;
    }
    out
}
fn strip_front_matter(content: &str) -> String {
    if !content.starts_with("---") {
        return content.to_string();
    }
    let re = Regex::new(r"(?s)^---\r?\n.*?\r?\n---\r?\n?").unwrap();
    re.replace(content, "").into_owned()
}

pub fn heading_path(c: &Chunk) -> String {
    if c.headings.is_empty() {
        "(root)".into()
    } else {
        c.headings.join(" > ")
    }
}

pub fn format_chunk_debug(c: &Chunk) -> String {
    let mut preview = c.text.clone();
    if preview.len() > 120 {
        preview = format!("{}...", &preview[..117]);
    }
    preview = preview.replace('\n', " ");
    format!("[{}] {} | {}", c.chunk_index, heading_path(c), preview)
}

/// Walk root and return chunks for every .md file.
pub fn collect(root: &Path) -> Result<Vec<Chunk>> {
    let meta = fs::metadata(root).with_context(|| format!("stat {}", root.display()))?;
    let mut chunks = Vec::new();

    let mut add_file = |path: &Path, rel: &str| -> Result<()> {
        let data = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        chunks.extend(split_markdown(rel, &data));
        Ok(())
    };

    if meta.is_file() {
        if !is_markdown(root) {
            bail!("{}: only .md files are supported", root.display());
        }
        let name = root
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| root.display().to_string());
        add_file(root, &name)?;
        return Ok(chunks);
    }

    for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
        if entry.file_type().is_dir() {
            let name = entry.file_name().to_string_lossy();
            if name == ".git" || name == "node_modules" || name == "vendor" || name == "target" {
                // WalkDir doesn't skip easily mid-walk without filter_entry; fine to skip files only
            }
            continue;
        }
        let path = entry.path();
        if !is_markdown(path) {
            continue;
        }
        // Skip under ignored dirs
        if path.components().any(|c| {
            matches!(
                c.as_os_str().to_str(),
                Some(".git" | "node_modules" | "vendor" | "target")
            )
        }) {
            continue;
        }
        let rel = path
            .strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .into_owned();
        add_file(path, &rel)?;
    }
    Ok(chunks)
}

fn is_markdown(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()),
        Some(ref e) if e == "md" || e == "markdown"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headings_and_hierarchy() {
        let md = r#"---
name: example
---

# Backport Process

Intro paragraph about backports.

## Overview

When a bug fix targets the current release.

## Requirements by Bug Status

### NEW

No PR requirements.

### ASSIGNED

Required:
- Fix version set

## Branch Naming

Upstream repos use stable branches.
"#;
        let chunks = split_markdown("backport-process.md", md);
        assert_eq!(chunks.len(), 5, "{:?}", chunks.iter().map(format_chunk_debug).collect::<Vec<_>>());
        assert_eq!(chunks[0].headings, ["Backport Process"]);
        assert!(chunks[0].text.contains("Intro paragraph"));
        assert_eq!(
            chunks[2].headings,
            ["Backport Process", "Requirements by Bug Status", "NEW"]
        );
    }

    #[test]
    fn empty_sections_skipped() {
        let md = "# Title\n\n## Empty\n\n## Has Content\n\nHello.\n";
        let chunks = split_markdown("x.md", md);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].headings, ["Title", "Has Content"]);
    }

    #[test]
    fn oversized_section_is_split() {
        let long = "word ".repeat(400); // well over MAX_CHUNK_CHARS
        let md = format!("# Doc\n\n## Big\n\n{long}");
        let chunks = split_markdown("big.md", &md);
        assert!(chunks.len() > 1, "expected split, got {}", chunks.len());
        for c in &chunks {
            assert!(
                c.text.chars().count() <= MAX_CHUNK_CHARS + 50,
                "chunk too long: {}",
                c.text.chars().count()
            );
            assert!(c.text.contains("Doc > Big") || c.headings.contains(&"Big".into()));
        }
    }
}
