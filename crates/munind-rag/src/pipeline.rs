use munind_api::engine::MunindEngine;
use munind_core::domain::{MemoryId, SearchHit, SearchRequest};
use munind_core::engine::VectorEngine;
use munind_core::error::{MunindError, Result};
use reqwest::blocking::Client;
use serde::Deserialize;
use serde_json::{Value, json};
use std::cmp::Ordering;
use std::collections::HashSet;
use std::time::Duration;

/// A simple markdown-aware text splitter.
pub struct TextSplitter {
    chunk_size: usize,
    chunk_overlap: usize,
}

impl TextSplitter {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size,
            chunk_overlap,
        }
    }

    /// Splits text into overlapping chunks, attempting to respect paragraph boundaries where possible.
    pub fn split_text(&self, text: &str) -> Vec<String> {
        let mut chunks = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let total_chars = chars.len();

        let mut i = 0;
        while i < total_chars {
            let end_idx = std::cmp::min(i + self.chunk_size, total_chars);
            let mut chunk: String = chars[i..end_idx].iter().collect();

            if end_idx < total_chars {
                if let Some(break_pos) = chunk.rfind("\n\n") {
                    chunk = chars[i..i + break_pos].iter().collect();
                } else if let Some(break_pos) = chunk.rfind(". ") {
                    chunk = chars[i..=i + break_pos].iter().collect();
                }
            }

            let chunk_len = chunk.chars().count();
            chunks.push(chunk);

            let step = std::cmp::max(1, chunk_len.saturating_sub(self.chunk_overlap));
            i += step;
        }

        chunks
    }
}

/// Embedding provider abstraction for production integrations.
pub trait EmbeddingProvider {
    fn model_id(&self) -> &str;
    fn embed_batch(&self, chunks: &[String], dim: usize) -> Result<Vec<Vec<f32>>>;
}

/// Deterministic local embedder for offline development/testing.
#[derive(Debug, Clone)]
pub struct DeterministicEmbedder {
    model_id: String,
}

impl Default for DeterministicEmbedder {
    fn default() -> Self {
        Self::new("deterministic-v1")
    }
}

impl DeterministicEmbedder {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
        }
    }
}

impl EmbeddingProvider for DeterministicEmbedder {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn embed_batch(&self, chunks: &[String], dim: usize) -> Result<Vec<Vec<f32>>> {
        let mut vectors = Vec::with_capacity(chunks.len());

        for chunk in chunks {
            let mut vec = vec![0.0_f32; dim];
            if dim == 0 {
                vectors.push(vec);
                continue;
            }

            // Stable byte-based projection with L2 normalization.
            for (i, b) in chunk.bytes().enumerate() {
                let idx = i % dim;
                let centered = (b as f32 - 127.5) / 127.5;
                vec[idx] += centered;
            }

            let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut vec {
                    *x /= norm;
                }
            }

            vectors.push(vec);
        }

        Ok(vectors)
    }
}

/// OpenAI-compatible embedding provider (`POST /v1/embeddings`).
#[derive(Debug, Clone)]
pub struct OpenAICompatibleEmbedder {
    endpoint: String,
    model_id: String,
    api_key: Option<String>,
    timeout: Duration,
}

impl OpenAICompatibleEmbedder {
    pub fn new(endpoint: impl Into<String>, model_id: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            model_id: model_id.into(),
            api_key: None,
            timeout: Duration::from_secs(30),
        }
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

#[derive(Debug, Deserialize)]
struct EmbeddingItem {
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingItem>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RerankResult {
    pub index: usize,
    pub score: f32,
}

/// Reranker abstraction for second-stage ranking over top-N candidates.
pub trait Reranker {
    fn model_id(&self) -> &str;
    fn rerank(&self, query: &str, documents: &[String], top_k: usize) -> Result<Vec<RerankResult>>;
}

/// Deterministic token-overlap reranker for offline testing.
#[derive(Debug, Clone)]
pub struct DeterministicReranker {
    model_id: String,
}

impl Default for DeterministicReranker {
    fn default() -> Self {
        Self::new("token-overlap-reranker-v1")
    }
}

impl DeterministicReranker {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
        }
    }
}

impl Reranker for DeterministicReranker {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn rerank(&self, query: &str, documents: &[String], top_k: usize) -> Result<Vec<RerankResult>> {
        if top_k == 0 || documents.is_empty() {
            return Ok(Vec::new());
        }

        let query_terms = unique_terms(query);
        if query_terms.is_empty() {
            return Ok(Vec::new());
        }

        let query_lc = query.to_lowercase();
        let mut scored = Vec::with_capacity(documents.len());

        for (index, doc) in documents.iter().enumerate() {
            let doc_terms = unique_terms(doc);
            if doc_terms.is_empty() {
                scored.push(RerankResult { index, score: 0.0 });
                continue;
            }

            let overlap = query_terms.intersection(&doc_terms).count() as f32;
            let coverage = overlap / query_terms.len() as f32;
            let specificity = overlap / doc_terms.len() as f32;
            let phrase_boost = if doc.to_lowercase().contains(&query_lc) {
                0.2
            } else {
                0.0
            };

            let score = (0.75 * coverage) + (0.25 * specificity) + phrase_boost;
            scored.push(RerankResult { index, score });
        }

        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.index.cmp(&b.index))
        });
        scored.truncate(top_k.min(scored.len()));
        Ok(scored)
    }
}

/// OpenAI-compatible reranker (`POST /v1/rerank` style payload).
#[derive(Debug, Clone)]
pub struct OpenAICompatibleReranker {
    endpoint: String,
    model_id: String,
    api_key: Option<String>,
    timeout: Duration,
}

impl OpenAICompatibleReranker {
    pub fn new(endpoint: impl Into<String>, model_id: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            model_id: model_id.into(),
            api_key: None,
            timeout: Duration::from_secs(30),
        }
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

#[derive(Debug, Deserialize)]
struct RerankItem {
    index: usize,
    #[serde(default)]
    relevance_score: Option<f32>,
    #[serde(default)]
    score: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct RerankResponse {
    #[serde(default)]
    results: Vec<RerankItem>,
    #[serde(default)]
    data: Vec<RerankItem>,
}

fn truncate_for_error(input: &str, max_chars: usize) -> String {
    if input.chars().count() <= max_chars {
        return input.to_string();
    }
    input.chars().take(max_chars).collect::<String>() + "..."
}

impl EmbeddingProvider for OpenAICompatibleEmbedder {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn embed_batch(&self, chunks: &[String], dim: usize) -> Result<Vec<Vec<f32>>> {
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        let client = Client::builder()
            .timeout(self.timeout)
            .build()
            .map_err(|e| MunindError::Internal(format!("failed to build embedding client: {e}")))?;

        let mut req = client.post(&self.endpoint).json(&json!({
            "model": self.model_id,
            "input": chunks,
        }));

        if let Some(api_key) = &self.api_key {
            req = req.bearer_auth(api_key);
        }

        let response = req
            .send()
            .map_err(|e| MunindError::Internal(format!("embedding request failed: {e}")))?;

        let status = response.status();
        let body = response.text().map_err(|e| {
            MunindError::Internal(format!("failed reading embedding response: {e}"))
        })?;

        if !status.is_success() {
            return Err(MunindError::Internal(format!(
                "embedding endpoint returned {}: {}",
                status,
                truncate_for_error(&body, 512)
            )));
        }

        let payload: EmbeddingResponse = serde_json::from_str(&body).map_err(|e| {
            MunindError::Internal(format!("failed parsing embedding response JSON: {e}"))
        })?;

        if payload.data.len() != chunks.len() {
            return Err(MunindError::Internal(format!(
                "embedding count mismatch: expected {}, got {}",
                chunks.len(),
                payload.data.len()
            )));
        }

        let mut ordered: Vec<Option<Vec<f32>>> = vec![None; chunks.len()];
        for item in payload.data {
            if item.index >= chunks.len() {
                return Err(MunindError::Internal(format!(
                    "embedding index out of range: {} (batch size {})",
                    item.index,
                    chunks.len()
                )));
            }
            if item.embedding.len() != dim {
                return Err(MunindError::DimensionMismatch {
                    expected: dim,
                    actual: item.embedding.len(),
                });
            }
            ordered[item.index] = Some(item.embedding);
        }

        let mut vectors = Vec::with_capacity(chunks.len());
        for maybe_vec in ordered {
            let vec = maybe_vec.ok_or_else(|| {
                MunindError::Internal("missing embedding row in provider response".to_string())
            })?;
            vectors.push(vec);
        }

        Ok(vectors)
    }
}

impl Reranker for OpenAICompatibleReranker {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn rerank(&self, query: &str, documents: &[String], top_k: usize) -> Result<Vec<RerankResult>> {
        if top_k == 0 || documents.is_empty() {
            return Ok(Vec::new());
        }

        let client = Client::builder()
            .timeout(self.timeout)
            .build()
            .map_err(|e| MunindError::Internal(format!("failed to build reranker client: {e}")))?;

        let mut req = client.post(&self.endpoint).json(&json!({
            "model": self.model_id,
            "query": query,
            "documents": documents,
            "top_n": top_k,
        }));

        if let Some(api_key) = &self.api_key {
            req = req.bearer_auth(api_key);
        }

        let response = req
            .send()
            .map_err(|e| MunindError::Internal(format!("rerank request failed: {e}")))?;

        let status = response.status();
        let body = response
            .text()
            .map_err(|e| MunindError::Internal(format!("failed reading rerank response: {e}")))?;

        if !status.is_success() {
            return Err(MunindError::Internal(format!(
                "rerank endpoint returned {}: {}",
                status,
                truncate_for_error(&body, 512)
            )));
        }

        let payload: RerankResponse = serde_json::from_str(&body).map_err(|e| {
            MunindError::Internal(format!("failed parsing rerank response JSON: {e}"))
        })?;

        let mut raw_items = if !payload.results.is_empty() {
            payload.results
        } else {
            payload.data
        };

        if raw_items.is_empty() {
            return Err(MunindError::Internal(
                "rerank response did not include results".to_string(),
            ));
        }

        let mut reranked = Vec::with_capacity(raw_items.len());
        for item in raw_items.drain(..) {
            if item.index >= documents.len() {
                return Err(MunindError::Internal(format!(
                    "rerank index out of range: {} (candidate count {})",
                    item.index,
                    documents.len()
                )));
            }

            let score = item.relevance_score.or(item.score).ok_or_else(|| {
                MunindError::Internal(
                    "rerank response item missing relevance_score/score".to_string(),
                )
            })?;

            reranked.push(RerankResult {
                index: item.index,
                score,
            });
        }

        reranked.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.index.cmp(&b.index))
        });
        reranked.truncate(top_k.min(reranked.len()));
        Ok(reranked)
    }
}

pub struct RagPipeline<E: EmbeddingProvider> {
    engine: MunindEngine,
    embedder: E,
    splitter: TextSplitter,
    reranker: Option<Box<dyn Reranker + Send + Sync>>,
    rerank_candidate_count: usize,
}

impl RagPipeline<DeterministicEmbedder> {
    pub fn with_deterministic_embedder(engine: MunindEngine) -> Self {
        Self::new(engine, DeterministicEmbedder::default())
    }
}

impl<E: EmbeddingProvider> RagPipeline<E> {
    pub fn new(engine: MunindEngine, embedder: E) -> Self {
        Self {
            engine,
            embedder,
            splitter: TextSplitter::new(512, 64),
            reranker: None,
            rerank_candidate_count: 100,
        }
    }

    pub fn with_reranker<R>(mut self, reranker: R) -> Self
    where
        R: Reranker + Send + Sync + 'static,
    {
        self.reranker = Some(Box::new(reranker));
        self
    }

    pub fn without_reranker(mut self) -> Self {
        self.reranker = None;
        self
    }

    pub fn with_rerank_candidate_count(mut self, count: usize) -> Self {
        self.rerank_candidate_count = count.max(1);
        self
    }

    /// Ingests a raw string document, chunks it, embeds it, and stores it in the database.
    /// `source` is persisted at both top-level (`source`) and nested (`metadata.source`) fields.
    pub fn ingest_document(
        &self,
        doc_id: &str,
        source: &str,
        content: &str,
    ) -> Result<Vec<MemoryId>> {
        self.ingest_document_with_metadata(
            doc_id,
            json!({
                "doc_id": doc_id,
                "source": source,
                "type": "document_chunk",
            }),
            content,
        )
    }

    /// Ingests a raw document with structured metadata.
    ///
    /// `metadata` must be a JSON object. Recognized metadata keys are mirrored to top-level
    /// fields for payload-index compatibility (`source`, `type`, `created_at`, `tags`, `session_id`).
    pub fn ingest_document_with_metadata(
        &self,
        doc_id: &str,
        metadata: Value,
        content: &str,
    ) -> Result<Vec<MemoryId>> {
        let Some(mut metadata_obj) = metadata.as_object().cloned() else {
            return Err(MunindError::InvalidConfig(
                "ingest_document metadata must be a JSON object".to_string(),
            ));
        };

        metadata_obj
            .entry("doc_id".to_string())
            .or_insert_with(|| Value::String(doc_id.to_string()));
        metadata_obj
            .entry("type".to_string())
            .or_insert_with(|| Value::String("document_chunk".to_string()));

        let chunks = self.splitter.split_text(content);
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        let embeddings = self
            .embedder
            .embed_batch(&chunks, self.engine.embedding_dimension())?;

        if embeddings.len() != chunks.len() {
            return Err(MunindError::Internal(
                "embedding provider returned mismatched batch size".to_string(),
            ));
        }

        let mut batch = Vec::with_capacity(chunks.len());
        for (i, (chunk, emb)) in chunks.into_iter().zip(embeddings).enumerate() {
            let mut doc = json!({
                "doc_id": doc_id,
                "metadata": Value::Object(metadata_obj.clone()),
                "chunk_idx": i,
                "text": chunk,
                "embedding_model_id": self.embedder.model_id(),
            });

            if let Some(doc_obj) = doc.as_object_mut() {
                for key in ["source", "type", "created_at", "tags", "session_id"] {
                    if let Some(value) = metadata_obj.get(key) {
                        doc_obj.insert(key.to_string(), value.clone());
                    }
                }
            }

            batch.push((emb, doc));
        }

        self.engine.insert_json_batch(batch)
    }

    /// Searches for documents relevant to a given query string.
    /// Uses hybrid retrieval and, when configured, second-stage reranking.
    pub fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchHit>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }

        let query_batch = vec![query.to_string()];
        let mut embedded = self
            .embedder
            .embed_batch(&query_batch, self.engine.embedding_dimension())?;

        let query_vec = embedded.pop().ok_or_else(|| {
            MunindError::Internal("embedding provider returned empty query batch".to_string())
        })?;

        let candidate_top_k = if self.reranker.is_some() {
            top_k.max(self.rerank_candidate_count)
        } else {
            top_k
        };

        let req = SearchRequest {
            vector: query_vec,
            top_k: candidate_top_k,
            text_query: Some(query.to_string()),
            hybrid_alpha: None,
            lexical_top_k: None,
            filter: None,
            ef_search: None,
            radius: None,
        };

        let hits = self.engine.search(req)?;
        self.rerank_hits(query, hits, top_k)
    }

    fn rerank_hits(
        &self,
        query: &str,
        mut candidates: Vec<SearchHit>,
        top_k: usize,
    ) -> Result<Vec<SearchHit>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }

        if candidates.is_empty() {
            return Ok(candidates);
        }

        let Some(reranker) = self.reranker.as_ref() else {
            candidates.truncate(top_k);
            return Ok(candidates);
        };

        let documents: Vec<String> = candidates
            .iter()
            .map(|hit| build_rerank_document(&hit.document))
            .collect();

        let mut reranked = reranker.rerank(query, &documents, top_k.min(candidates.len()))?;
        if reranked.is_empty() {
            candidates.truncate(top_k);
            return Ok(candidates);
        }

        reranked.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.index.cmp(&b.index))
        });

        let mut seen = HashSet::new();
        let mut out = Vec::with_capacity(top_k.min(candidates.len()));

        for item in reranked {
            if item.index >= candidates.len() {
                return Err(MunindError::Internal(format!(
                    "reranker '{}' returned out-of-range index {} for {} candidates",
                    reranker.model_id(),
                    item.index,
                    candidates.len()
                )));
            }

            if !seen.insert(item.index) {
                continue;
            }

            let mut hit = candidates[item.index].clone();
            hit.score = item.score;
            out.push(hit);

            if out.len() == top_k {
                return Ok(out);
            }
        }

        for (idx, hit) in candidates.into_iter().enumerate() {
            if seen.contains(&idx) {
                continue;
            }

            out.push(hit);
            if out.len() == top_k {
                break;
            }
        }

        Ok(out)
    }
}

fn build_rerank_document(doc: &Value) -> String {
    let mut parts = Vec::new();

    if let Some(title) = extract_title(doc) {
        let title = title.trim();
        if !title.is_empty() {
            parts.push(format!("title: {title}"));
        }
    }

    if let Some(text) = doc.get("text").and_then(Value::as_str) {
        let text = text.trim();
        if !text.is_empty() {
            parts.push(format!("text: {text}"));
        }
    }

    let tags = extract_tags(doc);
    if !tags.is_empty() {
        parts.push(format!("tags: {}", tags.join(", ")));
    }

    if parts.is_empty() {
        doc.to_string()
    } else {
        parts.join("\n")
    }
}

fn extract_title(doc: &Value) -> Option<String> {
    if let Some(title) = doc.get("title").and_then(Value::as_str) {
        return Some(title.to_string());
    }

    doc.get("metadata")
        .and_then(Value::as_object)
        .and_then(|m| m.get("title"))
        .and_then(Value::as_str)
        .map(ToString::to_string)
}

fn extract_tags(doc: &Value) -> Vec<String> {
    fn parse_tags(value: &Value) -> Option<Vec<String>> {
        if let Some(tag) = value.as_str() {
            return Some(vec![tag.to_string()]);
        }

        let arr = value.as_array()?;
        let mut out = Vec::new();
        for item in arr {
            if let Some(tag) = item.as_str() {
                out.push(tag.to_string());
            }
        }
        Some(out)
    }

    if let Some(tags) = doc.get("tags").and_then(parse_tags) {
        return tags;
    }

    doc.get("metadata")
        .and_then(Value::as_object)
        .and_then(|m| m.get("tags"))
        .and_then(parse_tags)
        .unwrap_or_default()
}

fn unique_terms(input: &str) -> HashSet<String> {
    let mut terms = HashSet::new();
    let mut token = String::new();

    for ch in input.chars() {
        if ch.is_alphanumeric() {
            for lc in ch.to_lowercase() {
                token.push(lc);
            }
        } else if !token.is_empty() {
            terms.insert(std::mem::take(&mut token));
        }
    }

    if !token.is_empty() {
        terms.insert(token);
    }

    terms
}

#[cfg(test)]
mod tests {
    use super::{DeterministicEmbedder, DeterministicReranker, EmbeddingProvider, RagPipeline};
    use munind_api::MunindEngine;
    use munind_core::config::EngineConfig;
    use munind_core::domain::{FilterExpression, SearchRequest};
    use munind_core::engine::VectorEngine;
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    fn test_rag_pipeline_uses_engine_embedding_dimension() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("db");
        let engine = MunindEngine::create(&db_path, 8, EngineConfig::default()).unwrap();
        let rag = RagPipeline::with_deterministic_embedder(engine);

        let ids = rag
            .ingest_document("doc-1", "test", "hello world. this is a memory.")
            .unwrap();
        assert!(!ids.is_empty());

        let hits = rag.search("hello world", 3).unwrap();
        assert!(!hits.is_empty());
        assert_eq!(hits[0].document["embedding_model_id"], "deterministic-v1");
    }

    #[test]
    fn test_ingest_document_writes_structured_metadata_for_indexes() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("db");
        let engine = MunindEngine::create(&db_path, 8, EngineConfig::default()).unwrap();

        let embedder = DeterministicEmbedder::default();
        let query_vec = embedder
            .embed_batch(&["hello world".to_string()], 8)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();

        let rag = RagPipeline::new(engine, embedder);
        let ids = rag
            .ingest_document("doc-1", "cli-ingest", "hello world. this is a test.")
            .unwrap();
        assert!(!ids.is_empty());

        let filtered = rag
            .engine
            .search(SearchRequest {
                vector: query_vec,
                top_k: 10,
                text_query: None,
                hybrid_alpha: None,
                lexical_top_k: None,
                filter: Some(FilterExpression::Eq(
                    "metadata.source".to_string(),
                    json!("cli-ingest"),
                )),
                ef_search: None,
                radius: None,
            })
            .unwrap();

        assert!(!filtered.is_empty());
        for hit in filtered {
            assert!(hit.document["metadata"].is_object());
            assert_eq!(hit.document["metadata"]["doc_id"], json!("doc-1"));
            assert_eq!(hit.document["metadata"]["source"], json!("cli-ingest"));
            assert_eq!(hit.document["source"], json!("cli-ingest"));
        }
    }

    #[test]
    fn test_reranker_reorders_top_candidate() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("db");
        let engine = MunindEngine::create(&db_path, 8, EngineConfig::default()).unwrap();

        let embedder = DeterministicEmbedder::default();
        let query = "desktop local memory";
        let mut query_emb = embedder
            .embed_batch(&[query.to_string()], 8)
            .unwrap()
            .into_iter();
        let query_vec = query_emb.next().unwrap();

        engine
            .insert_json(
                query_vec.clone(),
                json!({
                    "doc_id": "vector-best",
                    "title": "Noise",
                    "text": "totally unrelated content"
                }),
            )
            .unwrap();

        engine
            .insert_json(
                vec![0.0; 8],
                json!({
                    "doc_id": "lexical-best",
                    "title": "Desktop Notes",
                    "text": "local memory notes for desktop workflows",
                    "tags": ["desktop", "local", "memory"]
                }),
            )
            .unwrap();

        let base = engine
            .search(SearchRequest {
                vector: query_vec,
                top_k: 1,
                text_query: Some(query.to_string()),
                hybrid_alpha: None,
                lexical_top_k: None,
                filter: None,
                ef_search: None,
                radius: None,
            })
            .unwrap();
        assert_eq!(base[0].document["doc_id"], "vector-best");

        let rag = RagPipeline::new(engine, embedder)
            .with_reranker(DeterministicReranker::default())
            .with_rerank_candidate_count(50);

        let reranked = rag.search(query, 1).unwrap();
        assert_eq!(reranked[0].document["doc_id"], "lexical-best");
    }
}
