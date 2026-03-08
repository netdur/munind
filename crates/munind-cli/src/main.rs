use clap::{Parser, Subcommand};
use munind_api::MunindEngine;
use munind_core::config::EngineConfig;
use munind_rag::{
    DeterministicEmbedder, EmbeddingProvider, OpenAICompatibleEmbedder, OpenAICompatibleReranker,
    RagPipeline,
};
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    /// Path to the munind database directory
    #[arg(short, long, default_value = "./munind_data")]
    db: PathBuf,

    /// OpenAI-compatible embeddings endpoint (example: http://localhost:8082/v1/embeddings)
    #[arg(long)]
    embedding_endpoint: Option<String>,

    /// Embedding model id sent to provider and persisted as embedding_model_id
    #[arg(long, default_value = "deterministic-v1")]
    embedding_model: String,

    /// Optional API key for embedding endpoint authorization
    #[arg(long)]
    embedding_api_key: Option<String>,

    /// Optional reranker endpoint (example: http://localhost:8082/v1/rerank)
    #[arg(long)]
    reranker_endpoint: Option<String>,

    /// Reranker model id sent to reranker endpoint
    #[arg(long, default_value = "bge-reranker-v2-m3")]
    reranker_model: String,

    /// Optional API key for reranker endpoint authorization
    #[arg(long)]
    reranker_api_key: Option<String>,

    /// Candidate pool size retrieved before reranking
    #[arg(long, default_value_t = 100)]
    rerank_candidates: usize,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Creates a new database with a fixed embedding dimension
    Create {
        /// Embedding dimension for this database (immutable after creation)
        #[arg(long, default_value_t = 512)]
        embedding_dim: usize,
    },
    /// Ingests a text file into the vector database
    Ingest {
        /// The path to the text document
        #[arg(short, long)]
        file: PathBuf,

        /// An optional document ID
        #[arg(long, default_value = "doc")]
        doc_id: String,
    },
    /// Searches the vector database for a text query
    Search {
        /// The text query
        query: String,

        /// Number of results to return
        #[arg(short = 'k', long, default_value_t = 5)]
        top_k: usize,
    },
    /// Checks the health of the database
    CheckHealth,
}

#[derive(Clone)]
struct EmbedderConfig {
    endpoint: Option<String>,
    model: String,
    api_key: Option<String>,
}

#[derive(Clone)]
struct RerankerConfig {
    endpoint: Option<String>,
    model: String,
    api_key: Option<String>,
    candidate_count: usize,
}

#[derive(Clone)]
enum CliEmbedder {
    Deterministic(DeterministicEmbedder),
    OpenAI(OpenAICompatibleEmbedder),
}

impl EmbeddingProvider for CliEmbedder {
    fn model_id(&self) -> &str {
        match self {
            Self::Deterministic(e) => e.model_id(),
            Self::OpenAI(e) => e.model_id(),
        }
    }

    fn embed_batch(
        &self,
        chunks: &[String],
        dim: usize,
    ) -> munind_core::error::Result<Vec<Vec<f32>>> {
        match self {
            Self::Deterministic(e) => e.embed_batch(chunks, dim),
            Self::OpenAI(e) => e.embed_batch(chunks, dim),
        }
    }
}

fn build_embedder(cfg: &EmbedderConfig) -> anyhow::Result<CliEmbedder> {
    if let Some(endpoint) = &cfg.endpoint {
        if endpoint.trim().is_empty() {
            anyhow::bail!("--embedding-endpoint cannot be empty");
        }

        let mut provider = OpenAICompatibleEmbedder::new(endpoint.clone(), cfg.model.clone());
        if let Some(api_key) = &cfg.api_key {
            provider = provider.with_api_key(api_key.clone());
        }

        Ok(CliEmbedder::OpenAI(provider))
    } else {
        Ok(CliEmbedder::Deterministic(DeterministicEmbedder::new(
            cfg.model.clone(),
        )))
    }
}

fn build_reranker(cfg: &RerankerConfig) -> anyhow::Result<Option<OpenAICompatibleReranker>> {
    let Some(endpoint) = &cfg.endpoint else {
        return Ok(None);
    };

    if endpoint.trim().is_empty() {
        anyhow::bail!("--reranker-endpoint cannot be empty");
    }

    let mut reranker = OpenAICompatibleReranker::new(endpoint.clone(), cfg.model.clone());
    if let Some(api_key) = &cfg.api_key {
        reranker = reranker.with_api_key(api_key.clone());
    }

    Ok(Some(reranker))
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let base_config = EngineConfig::default();
    let embed_cfg = EmbedderConfig {
        endpoint: cli.embedding_endpoint.clone(),
        model: cli.embedding_model.clone(),
        api_key: cli.embedding_api_key.clone(),
    };
    let rerank_cfg = RerankerConfig {
        endpoint: cli.reranker_endpoint.clone(),
        model: cli.reranker_model.clone(),
        api_key: cli.reranker_api_key.clone(),
        candidate_count: cli.rerank_candidates.max(1),
    };

    match cli.command {
        Commands::Create { embedding_dim } => {
            MunindEngine::create(&cli.db, embedding_dim, base_config.clone())
                .map_err(|e| anyhow::anyhow!("Failed to create engine: {:?}", e))?;
            println!(
                "Created database at {:?} with embedding dimension {}",
                cli.db, embedding_dim
            );
        }
        Commands::Ingest { file, doc_id } => {
            let engine = MunindEngine::open(&cli.db, base_config.clone())
                .map_err(|e| anyhow::anyhow!("Failed to open engine: {:?}", e))?;
            let rag = RagPipeline::new(engine, build_embedder(&embed_cfg)?);

            let content = fs::read_to_string(&file)
                .map_err(|e| anyhow::anyhow!("Failed to read file: {}", e))?;

            println!("Ingesting file: {:?}...", file);
            let ids = rag
                .ingest_document(&doc_id, "cli-ingest", &content)
                .map_err(|e| anyhow::anyhow!("Failed to ingest: {:?}", e))?;

            println!("Successfully ingested {} chunk(s).", ids.len());
        }
        Commands::Search { query, top_k } => {
            let engine = MunindEngine::open(&cli.db, base_config.clone())
                .map_err(|e| anyhow::anyhow!("Failed to open engine: {:?}", e))?;

            let mut rag = RagPipeline::new(engine, build_embedder(&embed_cfg)?)
                .with_rerank_candidate_count(rerank_cfg.candidate_count);

            if let Some(reranker) = build_reranker(&rerank_cfg)? {
                rag = rag.with_reranker(reranker);
            }

            let results = rag
                .search(&query, top_k)
                .map_err(|e| anyhow::anyhow!("Search failed: {:?}", e))?;

            println!("Found {} results for '{}':", results.len(), query);
            for (i, hit) in results.iter().enumerate() {
                let text = hit
                    .document
                    .get("text")
                    .and_then(|t: &serde_json::Value| t.as_str())
                    .unwrap_or("No text");
                let doc_id = hit
                    .document
                    .get("doc_id")
                    .and_then(|t: &serde_json::Value| t.as_str())
                    .unwrap_or("Unknown");

                println!(
                    "{}. [Score: {:.4}] [Doc: {}] {}",
                    i + 1,
                    hit.score,
                    doc_id,
                    text.replace('\n', " ")
                );
            }
        }
        Commands::CheckHealth => {
            println!("Opening existing Munind database at {:?}", cli.db);
            let _engine = MunindEngine::open(&cli.db, base_config)
                .map_err(|e| anyhow::anyhow!("Failed to open engine: {:?}", e))?;
            println!("Database health check: OK");
        }
    }

    Ok(())
}
