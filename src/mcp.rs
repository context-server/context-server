//! MCP stdio server tools.

use crate::embed::Embedder;
use crate::search::Index;
use crate::store::Db;
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router, ServerHandler,
};
use std::sync::Mutex;

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SearchRequest {
    #[schemars(description = "the search query")]
    pub query: String,
    #[schemars(description = "max results to return (default 5)")]
    pub limit: Option<usize>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ListRequest {
    #[schemars(description = "max documents to list (default 50)")]
    pub limit: Option<usize>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct QuestionRequest {
    #[schemars(description = "the question to answer from indexed content")]
    pub question: String,
    #[schemars(description = "candidate passages to consider (default 3)")]
    pub limit: Option<usize>,
}

pub struct ContextService {
    pub db: Mutex<Db>,
    pub index: Index,
    pub embedder: Mutex<Embedder>,
    tool_router: ToolRouter<Self>,
}

impl ContextService {
    pub fn new(db: Db, index: Index, embedder: Embedder) -> Self {
        Self {
            db: Mutex::new(db),
            index,
            embedder: Mutex::new(embedder),
            tool_router: Self::tool_router(),
        }
    }
}

#[tool_router]
impl ContextService {
    #[tool(description = "Semantic search over the indexed knowledge base. Returns ranked passages with similarity scores.")]
    fn semantic_search(
        &self,
        Parameters(SearchRequest { query, limit }): Parameters<SearchRequest>,
    ) -> String {
        let limit = limit.unwrap_or(5);
        if query.trim().is_empty() {
            return "error: query is required".into();
        }
        let mut emb = self.embedder.lock().unwrap();
        match self.index.query(&mut emb, &query, limit) {
            Ok(hits) => format_hits(&query, &hits),
            Err(e) => format!("error: {e:#}"),
        }
    }

    #[tool(description = "List indexed document chunks (source path, headings, text preview).")]
    fn list_documents(
        &self,
        Parameters(ListRequest { limit }): Parameters<ListRequest>,
    ) -> String {
        let limit = limit.unwrap_or(50);
        let db = self.db.lock().unwrap();
        match db.list(limit) {
            Ok(docs) => {
                let mut out = format!("Showing {} document chunks:\n", docs.len());
                for d in docs {
                    let mut preview = d.text.clone();
                    if preview.len() > 160 {
                        preview = format!("{}...", &preview[..157]);
                    }
                    preview = preview.replace('\n', " ");
                    let heading = if d.headings.is_empty() {
                        "(root)".into()
                    } else {
                        d.headings.join(" > ")
                    };
                    out.push_str(&format!(
                        "- {}#{} [{}] {}\n",
                        d.source_path, d.chunk_index, heading, preview
                    ));
                }
                out
            }
            Err(e) => format!("error: {e:#}"),
        }
    }

    #[tool(description = "Answer a question by returning the most relevant passage from the knowledge base (search only; no generative QA).")]
    fn answer_question(
        &self,
        Parameters(QuestionRequest { question, limit }): Parameters<QuestionRequest>,
    ) -> String {
        let limit = limit.unwrap_or(3);
        if question.trim().is_empty() {
            return "error: question is required".into();
        }
        let mut emb = self.embedder.lock().unwrap();
        match self.index.query(&mut emb, &question, limit) {
            Ok(hits) if hits.is_empty() => "No relevant passages found.".into(),
            Ok(hits) => {
                let top = &hits[0];
                let mut out = format!(
                    "Best match (score={:.4}) from {}#{}\n\n{}\n",
                    top.score, top.source_path, top.chunk_index, top.text
                );
                if hits.len() > 1 {
                    out.push_str("\n---\nOther candidates:\n");
                    for h in &hits[1..] {
                        out.push_str(&format!(
                            "- score={:.4} {}#{}\n",
                            h.score, h.source_path, h.chunk_index
                        ));
                    }
                }
                out
            }
            Err(e) => format!("error: {e:#}"),
        }
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for ContextService {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

fn format_hits(query: &str, hits: &[crate::search::ResultHit]) -> String {
    let mut out = format!("Results for {query:?}:\n");
    if hits.is_empty() {
        out.push_str("(no hits)\n");
        return out;
    }
    for (i, h) in hits.iter().enumerate() {
        out.push_str(&format!(
            "\n{}. score={:.4}  {}#{}\n{}\n",
            i + 1,
            h.score,
            h.source_path,
            h.chunk_index,
            h.text
        ));
    }
    out
}
