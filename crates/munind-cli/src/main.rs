use clap::{Parser, Subcommand};
use munind_core::config::EngineConfig;
use munind_api::MunindEngine;
use munind_rag::RagPipeline;
use std::path::PathBuf;
use std::fs;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    /// Path to the munind database directory
    #[arg(short, long, default_value = "./munind_data")]
    db: PathBuf,

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

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let base_config = EngineConfig::default();

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
            let rag = RagPipeline::new(engine);

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
            let rag = RagPipeline::new(engine);
            let results = rag
                .search(&query, top_k)
                .map_err(|e| anyhow::anyhow!("Search failed: {:?}", e))?;

            println!("Found {} results for '{}':", results.len(), query);
            for (i, hit) in results.iter().enumerate() {
                let text = hit.document.get("text")
                    .and_then(|t: &serde_json::Value| t.as_str())
                    .unwrap_or("No text");
                let doc_id = hit.document.get("doc_id")
                    .and_then(|t: &serde_json::Value| t.as_str())
                    .unwrap_or("Unknown");
                    
                println!("{}. [Score: {:.4}] [Doc: {}] {}", i + 1, hit.score, doc_id, text.replace('\n', " "));
            }
        }
        Commands::CheckHealth => {
            println!("Opening existing Munind database at {:?}", cli.db);
            let _engine = MunindEngine::open(&cli.db, base_config)
                .map_err(|e| anyhow::anyhow!("Failed to open engine: {:?}", e))?;
            println!("Database health check: OK");
            // Could add graph connectivity statistics here later
        }
    }

    Ok(())
}
