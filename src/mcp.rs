//! MCP stdio server tools.

use crate::embed::Embedder;
use crate::search::{Index, SearchMode};
use crate::store::Db;
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router, ServerHandler,
};
use std::sync::Mutex;

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SearchRequest {
    #[schemars(
        description = "Natural-language search query. Include names, teams, acronyms, or topic keywords (e.g. 'who manages the storage team', 'backport process')."
    )]
    pub query: String,
    #[schemars(description = "Max passages to return (default 5)")]
    pub limit: Option<usize>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ListRequest {
    #[schemars(description = "Max chunks to list (default 50)")]
    pub limit: Option<usize>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct QuestionRequest {
    #[schemars(
        description = "Question about people, teams, ownership, processes, or org docs in the knowledge base."
    )]
    pub question: String,
    #[schemars(description = "Candidate passages to consider (default 3)")]
    pub limit: Option<usize>,
}

pub struct ContextService {
    pub db: Mutex<Db>,
    pub index: Index,
    pub embedder: Mutex<Embedder>,
    instructions: String,
    tool_router: ToolRouter<Self>,
}

const DEFAULT_INSTRUCTIONS: &str =
    "Organizational markdown knowledge base (teams, people, ownership, processes, guides). \
ALWAYS call semantic_search (or answer_question) before answering questions about \
who owns what, team structure, managers, acronyms, backports, or internal process — \
do not guess from general knowledge. Use list_documents to see what is indexed.";

impl ContextService {
    pub fn new(db: Db, index: Index, embedder: Embedder) -> Self {
        let instructions = db
            .get_meta("instructions")
            .ok()
            .flatten()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_INSTRUCTIONS.to_string());
        Self {
            db: Mutex::new(db),
            index,
            embedder: Mutex::new(embedder),
            instructions,
            tool_router: Self::tool_router(),
        }
    }
}

#[tool_router]
impl ContextService {
    #[tool(
        description = "REQUIRED for org/knowledge questions: search the indexed markdown knowledge base (people, teams, ownership, processes, guides). Call this instead of guessing whenever the user asks who owns something, how a process works, or anything that may be in team/org docs. Returns ranked passages with scores."
    )]
    fn semantic_search(
        &self,
        Parameters(SearchRequest { query, limit }): Parameters<SearchRequest>,
    ) -> String {
        let limit = limit.unwrap_or(5);
        if query.trim().is_empty() {
            return "error: query is required".into();
        }
        let mut emb = self.embedder.lock().unwrap();
        match self
            .index
            .query(&mut emb, &query, limit, SearchMode::Hybrid)
        {
            Ok(hits) => format_hits(&query, &hits),
            Err(e) => format!("error: {e:#}"),
        }
    }

    #[tool(
        description = "List what is indexed in the knowledge base (paths, headings, previews). Use when the user asks what docs are available or to browse the corpus."
    )]
    fn list_documents(&self, Parameters(ListRequest { limit }): Parameters<ListRequest>) -> String {
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

    #[tool(
        description = "Ask a question against the knowledge base and get the best matching passage. Prefer semantic_search for exploration; use this for a direct 'who/what/how' answer from indexed docs (retrieval only, not generative)."
    )]
    fn answer_question(
        &self,
        Parameters(QuestionRequest { question, limit }): Parameters<QuestionRequest>,
    ) -> String {
        let limit = limit.unwrap_or(3);
        if question.trim().is_empty() {
            return "error: question is required".into();
        }
        let mut emb = self.embedder.lock().unwrap();
        match self
            .index
            .query(&mut emb, &question, limit, SearchMode::Hybrid)
        {
            Ok(hits) if hits.is_empty() => "No relevant passages found.".into(),
            Ok(hits) => {
                let top = &hits[0];
                let mut out = format!(
                    "Best match (score={:.4}, dense={:.4}, lexical={:.4}) from {}#{}\n\n{}\n",
                    top.score,
                    top.dense_score,
                    top.lexical_score,
                    top.source_path,
                    top.chunk_index,
                    top.text
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
            instructions: Some(self.instructions.clone()),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

fn format_hits(query: &str, hits: &[crate::search::ResultHit]) -> String {
    let mut out = format!("Results for {query:?} (hybrid):\n");
    if hits.is_empty() {
        out.push_str("(no hits)\n");
        return out;
    }
    for (i, h) in hits.iter().enumerate() {
        let heading = if h.headings.is_empty() {
            String::new()
        } else {
            format!(" [{}]", h.headings.join(" > "))
        };
        out.push_str(&format!(
            "\n{}. score={:.4} (dense={:.4} lexical={:.4})  {}#{}{}\n{}\n",
            i + 1,
            h.score,
            h.dense_score,
            h.lexical_score,
            h.source_path,
            h.chunk_index,
            heading,
            h.text
        ));
    }
    out
}
