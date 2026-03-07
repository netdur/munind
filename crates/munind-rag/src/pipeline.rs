use munind_core::domain::{MemoryId, SearchRequest, SearchHit};
use munind_core::error::Result;
use munind_api::engine::MunindEngine;
use munind_core::engine::VectorEngine;
use serde_json::json;

/// A simple markdown-aware text splitter
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
            
            // If we aren't at the very end, try to find a natural break point (paragraph or sentence)
            if end_idx < total_chars {
                 if let Some(break_pos) = chunk.rfind("\n\n") {
                     chunk = chars[i..i+break_pos].iter().collect();
                 } else if let Some(break_pos) = chunk.rfind(". ") {
                     // Include the period
                     chunk = chars[i..=i+break_pos].iter().collect();
                 }
            }
            
            let chunk_len = chunk.chars().count();
            chunks.push(chunk);
            
            // Advance by chunk length minus overlap, ensuring we always advance at least 1 
            // to avoid infinite loops
            let step = std::cmp::max(1, chunk_len.saturating_sub(self.chunk_overlap));
            i += step;
        }

        chunks
    }
}

pub struct RagPipeline {
    engine: MunindEngine,
}

impl RagPipeline {
    pub fn new(engine: MunindEngine) -> Self {
        Self { engine }
    }

    /// Embeds a batch of text chunks using a local inference model or an API.
    /// MVP: A mock float generator for testing the end-to-end ingest flow.
    pub fn embed_batch_mock(&self, chunks: &[String], dim: usize) -> Result<Vec<Vec<f32>>> {
        let mut vectors = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            // Very naive "embedding" that hashes characters to floats so similar strings 
            // map roughly in the same quadrant
            let mut vec = vec![0.0; dim];
            for (i, c) in chunk.chars().enumerate().take(dim) {
                 vec[i] = (c as u32 as f32) / 255.0; // Normalize somewhat
            }
            vectors.push(vec);
        }
        Ok(vectors)
    }

    /// Ingests a raw string document, chunks it, embeds it, and stores it in the database.
    pub fn ingest_document(&self, doc_id: &str, metadata: &str, content: &str) -> Result<Vec<MemoryId>> {
        let splitter = TextSplitter::new(512, 64);
        let chunks = splitter.split_text(content);

        let embeddings = self.embed_batch_mock(&chunks, self.engine.embedding_dimension())?;
        
        let mut batch = Vec::new();
        for (i, (chunk, emb)) in chunks.into_iter().zip(embeddings).enumerate() {
            let doc = json!({
                "doc_id": doc_id,
                "metadata": metadata,
                "chunk_idx": i,
                "text": chunk
            });
            batch.push((emb, doc));
        }

        self.engine.insert_json_batch(batch)
    }

    /// Searches for documents relevant to a given query string using vector search.
    pub fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchHit>> {
        let query_vec = self
            .embed_batch_mock(&[query.to_string()], self.engine.embedding_dimension())?
            .pop()
            .unwrap();
        let req = SearchRequest {
            vector: query_vec,
            top_k,
            filter: None,
            ef_search: None,
            radius: None,
        };
        self.engine.search(req)
    }
}

#[cfg(test)]
mod tests {
    use super::RagPipeline;
    use munind_api::MunindEngine;
    use munind_core::config::EngineConfig;
    use tempfile::tempdir;

    #[test]
    fn test_rag_pipeline_uses_engine_embedding_dimension() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("db");
        let engine = MunindEngine::create(&db_path, 8, EngineConfig::default()).unwrap();
        let rag = RagPipeline::new(engine);

        let ids = rag
            .ingest_document("doc-1", "test", "hello world. this is a memory.")
            .unwrap();
        assert!(!ids.is_empty());

        let hits = rag.search("hello world", 3).unwrap();
        assert!(!hits.is_empty());
    }
}
