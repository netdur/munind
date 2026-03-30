use std::io::{BufRead, BufReader, Write};
use std::time::Instant;

use clap::{Parser, Subcommand};

use munind::{Index, IndexDistanceType, IndexProperty, MmapIndex, SearchOptions};

#[derive(Parser)]
#[command(name = "munind", version, about = "munind — NGT-compatible nearest neighbor index")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Create an index from a TSV data file.
    Create {
        /// Dimension of vectors.
        #[arg(short = 'd', long)]
        dimension: usize,

        /// Distance function: 1=L1, 2=L2, a=Angle, A=NormalizedAngle,
        /// c=Cosine(Normalized), C=CosineSimilarity, E=NormalizedL2,
        /// h=Hamming, j=Jaccard, p=Poincare, l=Lorentz, i=InnerProduct
        #[arg(short = 'D', long, default_value = "2")]
        distance: String,

        /// Edge size for creation.
        #[arg(short = 'E', long, default_value = "10")]
        edge_size: i32,

        /// Edge size for search.
        #[arg(short = 'S', long, default_value = "40")]
        edge_size_for_search: i32,

        /// Truncation edge limit.
        #[arg(short = 't', long, default_value = "50")]
        truncation_threshold: usize,

        /// Number of threads (unused in Phase 1, kept for CLI compat).
        #[arg(short = 'p', long, default_value = "8")]
        threads: usize,

        /// Quantize objects with TurboQuant at given bits/dim (0=off, 3/4/8=quantize).
        #[arg(short = 'q', long, default_value = "0")]
        quantize: u32,

        /// Output index path.
        index: String,

        /// Input TSV data file (optional — if omitted, creates empty index).
        data: Option<String>,
    },

    /// Search an index with queries from a TSV file or stdin.
    Search {
        /// Number of results per query.
        #[arg(short = 'n', long, default_value = "10")]
        result_size: usize,

        /// Epsilon (exploration coefficient offset).
        #[arg(short = 'e', long, default_value = "0.1")]
        epsilon: f32,

        /// Edge size for search (0 = all).
        #[arg(short = 'E', long, default_value = "0")]
        edge_size: i32,

        /// Output mode: (i)d, (d)istance, or (e)xtended.
        #[arg(short = 'o', long, default_value = "d")]
        output_mode: String,

        /// Index path.
        index: String,

        /// Query TSV file (reads from stdin if omitted).
        query: Option<String>,
    },

    /// Search using memory-mapped index (zero-copy object loading).
    SearchMmap {
        /// Number of results per query.
        #[arg(short = 'n', long, default_value = "10")]
        result_size: usize,

        /// Epsilon (exploration coefficient offset).
        #[arg(short = 'e', long, default_value = "0.1")]
        epsilon: f32,

        /// Edge size for search (0 = all).
        #[arg(short = 'E', long, default_value = "0")]
        edge_size: i32,

        /// Output mode: (i)d, (d)istance, or (e)xtended.
        #[arg(short = 'o', long, default_value = "d")]
        output_mode: String,

        /// Index path.
        index: String,

        /// Query TSV file (reads from stdin if omitted).
        query: Option<String>,
    },

    /// Print index information.
    Info {
        /// Index path.
        index: String,
    },

    /// Append vectors from a TSV file to an existing index.
    Append {
        /// Index path.
        index: String,

        /// TSV data file.
        data: String,
    },

    /// Remove objects by ID.
    Remove {
        /// Index path.
        index: String,

        /// Object IDs to remove.
        ids: Vec<u32>,
    },
}

fn parse_distance_type(s: &str) -> IndexDistanceType {
    match s {
        "1" => IndexDistanceType::L1,
        "2" => IndexDistanceType::L2,
        "a" => IndexDistanceType::Angle,
        "A" => IndexDistanceType::NormalizedAngle,
        "c" => IndexDistanceType::NormalizedCosine,
        "C" => IndexDistanceType::Cosine,
        "E" => IndexDistanceType::NormalizedL2,
        "h" => IndexDistanceType::Hamming,
        "j" => IndexDistanceType::Jaccard,
        "p" => IndexDistanceType::Poincare,
        "l" => IndexDistanceType::Lorentz,
        "i" => IndexDistanceType::InnerProduct,
        _ => {
            eprintln!("Warning: unknown distance type '{}', using L2", s);
            IndexDistanceType::L2
        }
    }
}

fn read_tsv_vectors(path: &str, dim: usize) -> Result<Vec<Vec<f32>>, String> {
    let file = std::fs::File::open(path)
        .map_err(|e| format!("Cannot open {}: {}", path, e))?;
    let reader = BufReader::new(file);
    let mut vectors = Vec::new();

    for (line_no, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("Read error at line {}: {}", line_no + 1, e))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let vals: Vec<f32> = line
            .split(|c: char| c == '\t' || c == ' ' || c == ',')
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<f32>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Parse error at line {}: {}", line_no + 1, e))?;

        if vals.len() < dim {
            return Err(format!(
                "Line {} has {} values, expected at least {}",
                line_no + 1,
                vals.len(),
                dim
            ));
        }
        vectors.push(vals[..dim].to_vec());
    }
    Ok(vectors)
}

fn cmd_create(
    dimension: usize,
    distance: String,
    edge_size: i32,
    edge_size_for_search: i32,
    truncation_threshold: usize,
    _threads: usize,
    quantize: u32,
    index_path: String,
    data_path: Option<String>,
) -> Result<(), String> {
    let mut property = IndexProperty::new(dimension);
    property.set_distance_type(parse_distance_type(&distance));
    property.edge_size_for_creation = edge_size;
    property.edge_size_for_search = edge_size_for_search;
    property.truncation_threshold = truncation_threshold;

    let mut index = Index::create(&index_path, property).map_err(|e| e.to_string())?;

    if let Some(data_path) = &data_path {
        let start = Instant::now();
        eprint!("Reading {}...", data_path);
        let vectors = read_tsv_vectors(data_path, dimension)?;
        eprintln!(" {} vectors in {:.2}s", vectors.len(), start.elapsed().as_secs_f64());

        let start = Instant::now();
        eprint!("Inserting...");
        for v in &vectors {
            index.insert(v).map_err(|e| e.to_string())?;
        }
        eprintln!(" done in {:.2}s", start.elapsed().as_secs_f64());

        let start = Instant::now();
        eprint!("Building index...");
        index.build();
        eprintln!(" done in {:.2}s", start.elapsed().as_secs_f64());
    }

    if quantize > 0 {
        // Build full-precision index first, then quantize.
        let start = Instant::now();
        eprint!("Saving native index...");
        index.save_as_directory(&index_path).map_err(|e| e.to_string())?;
        eprintln!(" done in {:.2}s", start.elapsed().as_secs_f64());

        let start = Instant::now();
        eprint!("Quantizing with TurboQuant ({}-bit)...", quantize);
        let tq = munind::tq::TqIndex::build_from_index(&index_path, quantize)
            .map_err(|e| e.to_string())?;
        eprintln!(" done in {:.2}s", start.elapsed().as_secs_f64());

        let start = Instant::now();
        eprint!("Saving TQ index...");
        tq.save(&index_path).map_err(|e| e.to_string())?;
        eprintln!(" done in {:.2}s", start.elapsed().as_secs_f64());
        eprintln!(
            "munind: created TQ-{} index at {} with {} objects",
            quantize, index_path, tq.object_count()
        );
    } else {
        let start = Instant::now();
        eprint!("Saving index...");
        index.save_as_directory(&index_path).map_err(|e| e.to_string())?;
        eprintln!(" done in {:.2}s", start.elapsed().as_secs_f64());
        eprintln!("munind: created index at {} with {} objects", index_path, index.object_count());
    }
    Ok(())
}

fn cmd_search(
    result_size: usize,
    epsilon: f32,
    edge_size: i32,
    output_mode: String,
    index_path: String,
    query_path: Option<String>,
) -> Result<(), String> {
    // Auto-detect compressed index formats.
    if munind::tq::TqIndex::is_tq_index(&index_path) {
        return cmd_search_tq(result_size, epsilon, edge_size, output_mode, index_path, query_path);
    }


    let index = Index::open_directory(&index_path).map_err(|e| e.to_string())?;
    let dim = index.object_space.as_ref().ok_or("no object space")?.dim;

    let options = SearchOptions {
        k: result_size,
        epsilon,
        edge_size: if edge_size == 0 { None } else { Some(edge_size as usize) },
    };

    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());

    let reader: Box<dyn BufRead> = match query_path {
        Some(path) => {
            let file = std::fs::File::open(&path)
                .map_err(|e| format!("Cannot open {}: {}", path, e))?;
            Box::new(BufReader::new(file))
        }
        None => Box::new(BufReader::new(std::io::stdin())),
    };

    let mut query_count = 0u64;
    let mut total_time = std::time::Duration::ZERO;

    for line in reader.lines() {
        let line = line.map_err(|e| format!("Read error: {}", e))?;
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let vals: Vec<f32> = line
            .split(|c: char| c == '\t' || c == ' ' || c == ',')
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<f32>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Parse error: {}", e))?;

        if vals.len() < dim {
            eprintln!("Warning: query has {} values, expected {}", vals.len(), dim);
            continue;
        }

        let q = &vals[..dim];
        let start = Instant::now();
        let results = index.search(q, &options).map_err(|e| e.to_string())?;
        total_time += start.elapsed();
        query_count += 1;

        // Output format matches NGT: "Query No.N"
        writeln!(out, "Query No.{}", query_count).ok();
        for (rank, r) in results.iter().enumerate() {
            let rid: u32 = r.id;
            let rdist: f32 = r.distance;
            match output_mode.as_str() {
                "i" => writeln!(out, "{}\t{}", rank + 1, rid).ok(),
                "e" => writeln!(out, "{}\t{}\t{}", rank + 1, rid, rdist).ok(),
                _ => writeln!(out, "{}\t{}\t{}", rank + 1, rid, rdist).ok(),
            };
        }
    }

    if query_count > 0 {
        let avg_ms = total_time.as_secs_f64() / query_count as f64 * 1000.0;
        eprintln!(
            "Average query time: {:.6} ms ({} queries)",
            avg_ms, query_count
        );
    }

    Ok(())
}

fn cmd_search_tq(
    result_size: usize,
    epsilon: f32,
    edge_size: i32,
    output_mode: String,
    index_path: String,
    query_path: Option<String>,
) -> Result<(), String> {
    let tq = munind::tq::TqIndex::load(&index_path).map_err(|e| e.to_string())?;
    let dim = tq.property.dimension;

    let options = SearchOptions {
        k: result_size,
        epsilon,
        edge_size: if edge_size == 0 { None } else { Some(edge_size as usize) },
    };

    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());

    let reader: Box<dyn BufRead> = match query_path {
        Some(path) => {
            let file = std::fs::File::open(&path)
                .map_err(|e| format!("Cannot open {}: {}", path, e))?;
            Box::new(BufReader::new(file))
        }
        None => Box::new(BufReader::new(std::io::stdin())),
    };

    let mut query_count = 0u64;
    let mut total_time = std::time::Duration::ZERO;

    for line in reader.lines() {
        let line = line.map_err(|e| format!("Read error: {}", e))?;
        let line = line.trim().to_string();
        if line.is_empty() { continue; }
        let vals: Vec<f32> = line
            .split(|c: char| c == '\t' || c == ' ' || c == ',')
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<f32>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Parse error: {}", e))?;
        if vals.len() < dim { continue; }
        let q = &vals[..dim];

        let start = Instant::now();
        let results = tq.search(q, &options).map_err(|e| e.to_string())?;
        total_time += start.elapsed();
        query_count += 1;

        writeln!(out, "Query No.{}", query_count).ok();
        for (rank, r) in results.iter().enumerate() {
            let rid: u32 = r.id;
            let rdist: f32 = r.distance;
            match output_mode.as_str() {
                "i" => writeln!(out, "{}\t{}", rank + 1, rid).ok(),
                _ => writeln!(out, "{}\t{}\t{}", rank + 1, rid, rdist).ok(),
            };
        }
    }

    if query_count > 0 {
        let avg_ms = total_time.as_secs_f64() / query_count as f64 * 1000.0;
        eprintln!("Average query time: {:.6} ms ({} queries, TQ)", avg_ms, query_count);
    }
    Ok(())
}

fn cmd_search_mmap(
    result_size: usize,
    epsilon: f32,
    edge_size: i32,
    output_mode: String,
    index_path: String,
    query_path: Option<String>,
) -> Result<(), String> {
    // Auto-detect compressed formats.
    if munind::tq::TqIndex::is_tq_index(&index_path) {
        return cmd_search_tq(result_size, epsilon, edge_size, output_mode, index_path, query_path);
    }


    let start = Instant::now();
    let index = MmapIndex::open(&index_path).map_err(|e| e.to_string())?;
    eprintln!("MmapIndex opened in {:.3}ms", start.elapsed().as_secs_f64() * 1000.0);

    let _dim = index.object_count(); // We need dim from property; use a workaround.
    // Read dim from property file.
    let mut ps = munind::common::PropertySet::new();
    ps.load(&format!("{}/prf", index_path)).map_err(|e| e.to_string())?;
    let dim = ps.get_i64("Dimension", 0) as usize;

    let options = SearchOptions {
        k: result_size,
        epsilon,
        edge_size: if edge_size == 0 { None } else { Some(edge_size as usize) },
    };

    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());

    let reader: Box<dyn BufRead> = match query_path {
        Some(path) => {
            let file = std::fs::File::open(&path)
                .map_err(|e| format!("Cannot open {}: {}", path, e))?;
            Box::new(BufReader::new(file))
        }
        None => Box::new(BufReader::new(std::io::stdin())),
    };

    let mut query_count = 0u64;
    let mut total_time = std::time::Duration::ZERO;

    for line in reader.lines() {
        let line = line.map_err(|e| format!("Read error: {}", e))?;
        let line = line.trim().to_string();
        if line.is_empty() { continue; }

        let vals: Vec<f32> = line
            .split(|c: char| c == '\t' || c == ' ' || c == ',')
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<f32>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Parse error: {}", e))?;

        if vals.len() < dim { continue; }
        let q = &vals[..dim];

        let start = Instant::now();
        let results = index.search(q, &options).map_err(|e| e.to_string())?;
        total_time += start.elapsed();
        query_count += 1;

        writeln!(out, "Query No.{}", query_count).ok();
        for (rank, r) in results.iter().enumerate() {
            let rid: u32 = r.id;
            let rdist: f32 = r.distance;
            match output_mode.as_str() {
                "i" => writeln!(out, "{}\t{}", rank + 1, rid).ok(),
                _ => writeln!(out, "{}\t{}\t{}", rank + 1, rid, rdist).ok(),
            };
        }
    }

    if query_count > 0 {
        let avg_ms = total_time.as_secs_f64() / query_count as f64 * 1000.0;
        eprintln!("Average query time: {:.6} ms ({} queries)", avg_ms, query_count);
    }
    Ok(())
}

fn cmd_info(index_path: String) -> Result<(), String> {
    let index = Index::open_directory(&index_path).map_err(|e| e.to_string())?;
    let os = index.object_space.as_ref().ok_or("no object space")?;
    println!("Number of objects\t{}", index.object_count());
    println!("Dimension\t\t{}", os.dim);
    println!("Graph edges\t\t{}", index.graph.edges.iter().map(Vec::len).sum::<usize>());
    println!("Leaf nodes\t\t{}", index.tree.as_ref().map_or(0, |t| t.leaves().iter().flatten().count()));
    Ok(())
}

fn cmd_append(index_path: String, data_path: String) -> Result<(), String> {
    let mut index = Index::open_directory(&index_path).map_err(|e| e.to_string())?;
    let dim = index.object_space.as_ref().ok_or("no object space")?.dim;

    let vectors = read_tsv_vectors(&data_path, dim)?;
    let before = index.object_count();

    for v in &vectors {
        index.insert(v).map_err(|e| e.to_string())?;
    }
    index.build();
    index.save_as_directory(&index_path).map_err(|e| e.to_string())?;

    eprintln!(
        "Appended {} objects ({} -> {})",
        vectors.len(),
        before,
        index.object_count()
    );
    Ok(())
}

fn cmd_remove(index_path: String, ids: Vec<u32>) -> Result<(), String> {
    let mut index = Index::open_directory(&index_path).map_err(|e| e.to_string())?;
    let removed = index.delete_batch(&ids).map_err(|e| e.to_string())?;
    index.save_as_directory(&index_path).map_err(|e| e.to_string())?;
    eprintln!("Removed {} objects", removed);
    Ok(())
}

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Create {
            dimension,
            distance,
            edge_size,
            edge_size_for_search,
            truncation_threshold,
            threads,
            quantize,
            index,
            data,
        } => cmd_create(
            dimension,
            distance,
            edge_size,
            edge_size_for_search,
            truncation_threshold,
            threads,
            quantize,
            index,
            data,
        ),
        Command::Search {
            result_size,
            epsilon,
            edge_size,
            output_mode,
            index,
            query,
        } => cmd_search(result_size, epsilon, edge_size, output_mode, index, query),
        Command::SearchMmap {
            result_size,
            epsilon,
            edge_size,
            output_mode,
            index,
            query,
        } => cmd_search_mmap(result_size, epsilon, edge_size, output_mode, index, query),
        Command::Info { index } => cmd_info(index),
        Command::Append { index, data } => cmd_append(index, data),
        Command::Remove { index, ids } => cmd_remove(index, ids),
    };

    if let Err(e) = result {
        eprintln!("munind: error: {}", e);
        std::process::exit(1);
    }
}
