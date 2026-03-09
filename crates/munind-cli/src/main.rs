use clap::{ArgAction, Parser, Subcommand};
use munind_api::MunindEngine;
use munind_core::config::EngineConfig;
use munind_core::domain::{MemoryId, OptimizeRequest};
use munind_core::engine::VectorEngine;
use munind_rag::{
    DeterministicEmbedder, EmbeddingProvider, OpenAICompatibleEmbedder, OpenAICompatibleReranker,
    RagPipeline,
};
use munind_storage::StorageEngine;
use serde_json::{Value, json};
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

        /// Print machine-readable JSON output instead of human text
        #[arg(long, default_value_t = false, action = ArgAction::SetTrue)]
        json: bool,
    },
    /// Inserts a single embedding + JSON document record
    Insert {
        /// Embedding vector as JSON array string, e.g. "[0.1,0.2]"
        #[arg(long)]
        embedding_json: Option<String>,
        /// Path to file containing embedding JSON array
        #[arg(long)]
        embedding_file: Option<PathBuf>,
        /// Document payload as JSON object string
        #[arg(long)]
        document_json: Option<String>,
        /// Path to file containing document JSON object
        #[arg(long)]
        document_file: Option<PathBuf>,
    },
    /// Gets a record by id
    Get {
        /// Record id
        #[arg(long)]
        id: u64,
        /// Include embedding in output (can be large)
        #[arg(long, default_value_t = false, action = ArgAction::SetTrue)]
        include_embedding: bool,
        /// Print compact JSON output
        #[arg(long, default_value_t = false, action = ArgAction::SetTrue)]
        json: bool,
    },
    /// Updates an existing record by id
    Update {
        /// Record id
        #[arg(long)]
        id: u64,
        /// Embedding vector as JSON array string, e.g. "[0.1,0.2]"
        #[arg(long)]
        embedding_json: Option<String>,
        /// Path to file containing embedding JSON array
        #[arg(long)]
        embedding_file: Option<PathBuf>,
        /// Document payload as JSON object string
        #[arg(long)]
        document_json: Option<String>,
        /// Path to file containing document JSON object
        #[arg(long)]
        document_file: Option<PathBuf>,
    },
    /// Deletes a record by id
    Delete {
        /// Record id
        #[arg(long)]
        id: u64,
    },
    /// Checks the health of the database
    CheckHealth,
    /// Optimizes storage and optionally repairs graph/index state
    Optimize {
        /// Disable full compaction (by default optimize compacts and truncates WAL)
        #[arg(long = "no-compact", action = ArgAction::SetFalse, default_value_t = true)]
        compact: bool,
        /// Only write checkpoint + truncate WAL (no segment rewrite)
        #[arg(long, default_value_t = false, action = ArgAction::SetTrue)]
        checkpoint_wal_only: bool,
        /// Rebuild graph/index state from storage
        #[arg(long, default_value_t = false, action = ArgAction::SetTrue)]
        repair_graph: bool,
    },
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

fn read_json_input(
    inline: Option<String>,
    file: Option<PathBuf>,
    label: &str,
) -> anyhow::Result<String> {
    match (inline, file) {
        (Some(s), None) => Ok(s),
        (None, Some(path)) => fs::read_to_string(&path)
            .map_err(|e| anyhow::anyhow!("failed to read {} file {:?}: {}", label, path, e)),
        (None, None) => anyhow::bail!(
            "missing {} input: provide either --{}-json or --{}-file",
            label,
            label,
            label
        ),
        (Some(_), Some(_)) => anyhow::bail!(
            "ambiguous {} input: provide only one of --{}-json or --{}-file",
            label,
            label,
            label
        ),
    }
}

fn parse_embedding_json(raw: &str) -> anyhow::Result<Vec<f32>> {
    let value: Value =
        serde_json::from_str(raw).map_err(|e| anyhow::anyhow!("invalid embedding json: {}", e))?;
    let arr = value
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("embedding must be a JSON array"))?;
    let mut out = Vec::with_capacity(arr.len());
    for (i, item) in arr.iter().enumerate() {
        let n = item
            .as_f64()
            .ok_or_else(|| anyhow::anyhow!("embedding[{}] must be a number", i))?;
        out.push(n as f32);
    }
    Ok(out)
}

fn parse_document_json(raw: &str) -> anyhow::Result<Value> {
    let value: Value =
        serde_json::from_str(raw).map_err(|e| anyhow::anyhow!("invalid document json: {}", e))?;
    if !value.is_object() {
        anyhow::bail!("document must be a JSON object");
    }
    Ok(value)
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
            let engine = MunindEngine::open(&cli.db)
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
        Commands::Search { query, top_k, json } => {
            let engine = MunindEngine::open(&cli.db)
                .map_err(|e| anyhow::anyhow!("Failed to open engine: {:?}", e))?;

            let mut rag = RagPipeline::new(engine, build_embedder(&embed_cfg)?)
                .with_rerank_candidate_count(rerank_cfg.candidate_count);

            if let Some(reranker) = build_reranker(&rerank_cfg)? {
                rag = rag.with_reranker(reranker);
            }

            let results = rag
                .search(&query, top_k)
                .map_err(|e| anyhow::anyhow!("Search failed: {:?}", e))?;

            if json {
                println!("{}", serde_json::to_string_pretty(&results)?);
            } else {
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
        }
        Commands::Insert {
            embedding_json,
            embedding_file,
            document_json,
            document_file,
        } => {
            let embedding_raw = read_json_input(embedding_json, embedding_file, "embedding")?;
            let document_raw = read_json_input(document_json, document_file, "document")?;
            let embedding = parse_embedding_json(&embedding_raw)?;
            let document = parse_document_json(&document_raw)?;

            let engine = MunindEngine::open(&cli.db)
                .map_err(|e| anyhow::anyhow!("Failed to open engine: {:?}", e))?;
            let id = engine
                .insert_json(embedding, document)
                .map_err(|e| anyhow::anyhow!("Insert failed: {:?}", e))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "status": "ok",
                    "id": id.0
                }))?
            );
        }
        Commands::Get {
            id,
            include_embedding,
            json,
        } => {
            let engine = MunindEngine::open(&cli.db)
                .map_err(|e| anyhow::anyhow!("Failed to open engine: {:?}", e))?;
            let Some((embedding, document)) = engine
                .get_record(MemoryId(id))
                .map_err(|e| anyhow::anyhow!("Get failed: {:?}", e))?
            else {
                anyhow::bail!("record id {} not found", id);
            };

            let output = if include_embedding {
                json!({"id": id, "embedding": embedding, "document": document})
            } else {
                json!({"id": id, "document": document})
            };

            if json {
                println!("{}", serde_json::to_string(&output)?);
            } else {
                println!("{}", serde_json::to_string_pretty(&output)?);
            }
        }
        Commands::Update {
            id,
            embedding_json,
            embedding_file,
            document_json,
            document_file,
        } => {
            let embedding_raw = read_json_input(embedding_json, embedding_file, "embedding")?;
            let document_raw = read_json_input(document_json, document_file, "document")?;
            let embedding = parse_embedding_json(&embedding_raw)?;
            let document = parse_document_json(&document_raw)?;

            let engine = MunindEngine::open(&cli.db)
                .map_err(|e| anyhow::anyhow!("Failed to open engine: {:?}", e))?;
            engine
                .update_json(MemoryId(id), embedding, document)
                .map_err(|e| anyhow::anyhow!("Update failed: {:?}", e))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "status": "ok",
                    "id": id
                }))?
            );
        }
        Commands::Delete { id } => {
            let engine = MunindEngine::open(&cli.db)
                .map_err(|e| anyhow::anyhow!("Failed to open engine: {:?}", e))?;
            engine
                .remove(MemoryId(id))
                .map_err(|e| anyhow::anyhow!("Delete failed: {:?}", e))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "status": "ok",
                    "id": id
                }))?
            );
        }
        Commands::CheckHealth => {
            println!("Opening existing Munind database at {:?}", cli.db);
            let _engine = MunindEngine::open(&cli.db)
                .map_err(|e| anyhow::anyhow!("Failed to open engine: {:?}", e))?;
            println!("Database health check: OK");
        }
        Commands::Optimize {
            compact,
            checkpoint_wal_only,
            repair_graph,
        } => {
            println!("Opening existing Munind database at {:?}", cli.db);
            let effective_compact = if checkpoint_wal_only { false } else { compact };
            println!(
                "Running optimize (compact: {}, checkpoint_wal_only: {}, repair_graph: {})...",
                effective_compact, checkpoint_wal_only, repair_graph
            );

            let report = if checkpoint_wal_only && !repair_graph && !effective_compact {
                // Fast path: storage-only checkpoint avoids rebuilding ANN/lexical indexes.
                let storage = StorageEngine::open(&cli.db)
                    .map_err(|e| anyhow::anyhow!("Failed to open storage: {:?}", e))?;
                storage
                    .optimize(OptimizeRequest {
                        force_full_compaction: false,
                        repair_graph: false,
                        checkpoint_wal_only: true,
                    })
                    .map_err(|e| anyhow::anyhow!("Optimize failed: {:?}", e))?
            } else {
                let engine = MunindEngine::open(&cli.db)
                    .map_err(|e| anyhow::anyhow!("Failed to open engine: {:?}", e))?;
                engine
                    .optimize(OptimizeRequest {
                        force_full_compaction: effective_compact,
                        repair_graph,
                        checkpoint_wal_only,
                    })
                    .map_err(|e| anyhow::anyhow!("Optimize failed: {:?}", e))?
            };

            println!("Optimization complete:");
            println!("  records_compacted: {}", report.records_compacted);
            println!("  space_reclaimed_bytes: {}", report.space_reclaimed_bytes);
            println!("  graph_edges_repaired: {}", report.graph_edges_repaired);
        }
    }

    Ok(())
}
