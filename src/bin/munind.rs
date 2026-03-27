use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{Args, Parser, Subcommand};
use munind::index::IndexType;
use munind::{Index, IndexDistanceType, IndexProperty, ObjectDistance, SearchOptions};

#[derive(Parser, Debug)]
#[command(name = "munind")]
#[command(about = "Rust NGT-like CLI for the munind index", long_about = None)]
struct Cli {
    #[arg(short = 'X', global = true, default_value_t = 0)]
    debug: u32,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Create(CreateArgs),
    Search(SearchArgs),
    Append(AppendArgs),
    Info(InfoArgs),
    Export(TransferArgs),
    Import(TransferArgs),
    ExportGraph(IndexOnlyArgs),
    ExportObjects(IndexOnlyArgs),
    Rebuild(IndexOnlyArgs),
    Remove(UnsupportedArgs),
    Prune(UnsupportedArgs),
    ReconstructGraph(UnsupportedArgs),
    OptimizeSearchParameters(UnsupportedArgs),
    #[command(name = "optimize-#-of-edges")]
    OptimizeNumberOfEdges(UnsupportedArgs),
    Repair(UnsupportedArgs),
    Eval(UnsupportedArgs),
    #[command(name = "refine-anng")]
    RefineAnng(UnsupportedArgs),
    #[command(name = "prep-pq")]
    PrepPq(UnsupportedArgs),
    #[command(name = "extract-query")]
    ExtractQuery(UnsupportedArgs),
    #[command(name = "adjust-edge-size")]
    AdjustEdgeSize(UnsupportedArgs),
}

#[derive(Args, Debug)]
struct UnsupportedArgs {
    #[arg(trailing_var_arg = true)]
    args: Vec<String>,
}

#[derive(Args, Debug)]
struct IndexOnlyArgs {
    index: PathBuf,
}

#[derive(Args, Debug)]
struct TransferArgs {
    index: PathBuf,
    file: PathBuf,
}

#[derive(Args, Debug)]
struct CreateArgs {
    #[arg(short = 'd')]
    dimension: usize,
    #[arg(short = 'p', default_value_t = 24)]
    thread_pool_size: usize,
    #[arg(short = 'i', default_value = "t")]
    index_type: String,
    #[arg(short = 'g', default_value = "a")]
    graph_type: String,
    #[arg(short = 't', default_value_t = 0)]
    truncation_threshold: isize,
    #[arg(short = 'E', default_value_t = 10)]
    edge_size_for_creation: usize,
    #[arg(short = 'S', default_value_t = 40)]
    edge_size_for_search: usize,
    #[arg(short = 'e', default_value_t = 0.1)]
    epsilon: f32,
    #[arg(short = 'D', default_value = "2")]
    distance_type: String,
    #[arg(short = 'P', default_value_t = 0)]
    path_adjustment_interval: isize,
    #[arg(short = 'B', default_value_t = 30)]
    dynamic_edge_size_base: usize,
    #[arg(short = 'A', default_value = "f")]
    object_alignment: String,
    #[arg(short = 'T', default_value_t = 0.0)]
    build_time_limit: f32,
    #[arg(short = 'O')]
    outgoing_incoming: Option<String>,
    #[arg(short = 'b', default_value_t = 200)]
    batch_size_for_creation: usize,
    #[arg(short = 'L')]
    leaf_internal_sizes: Option<String>,
    #[arg(short = 'o', default_value = "f")]
    object_type: String,
    #[arg(short = 's', default_value = "-")]
    seed_type: String,
    #[arg(short = 'M', default_value = "-")]
    epsilon_type: String,
    #[arg(short = 'r', default_value = "-")]
    identical_object_edge_type: String,
    #[arg(short = 'l')]
    insertion_order: Option<String>,
    #[arg(short = 'c', default_value_t = 0.0)]
    clipping_rate: f32,
    #[arg(short = 'v', default_value_t = false)]
    quiet: bool,
    index: PathBuf,
    data: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct SearchArgs {
    #[arg(short = 'n', default_value_t = 20)]
    size: usize,
    #[arg(short = 'e', default_value_t = 0.1)]
    epsilon: f32,
    #[arg(short = 'E', default_value_t = -1)]
    edge_size: isize,
    #[arg(short = 'o', default_value = "-")]
    output_mode: String,
    index: PathBuf,
    query: PathBuf,
}

#[derive(Args, Debug)]
struct AppendArgs {
    #[arg(short = 'v', default_value_t = false)]
    quiet: bool,
    index: PathBuf,
    data: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct InfoArgs {
    index: PathBuf,
}

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Create(args) => create(args, cli.debug),
        Command::Search(args) => search(args, cli.debug),
        Command::Append(args) => append(args, cli.debug),
        Command::Info(args) => info(args),
        Command::Export(args) => export_index(args),
        Command::Import(args) => import_index(args),
        Command::ExportGraph(args) => export_graph(args),
        Command::ExportObjects(args) => export_objects(args),
        Command::Rebuild(args) => rebuild(args),
        Command::Remove(_) => unsupported("remove"),
        Command::Prune(_) => unsupported("prune"),
        Command::ReconstructGraph(_) => unsupported("reconstruct-graph"),
        Command::OptimizeSearchParameters(_) => unsupported("optimize-search-parameters"),
        Command::OptimizeNumberOfEdges(_) => unsupported("optimize-#-of-edges"),
        Command::Repair(_) => unsupported("repair"),
        Command::Eval(_) => unsupported("eval"),
        Command::RefineAnng(_) => unsupported("refine-anng"),
        Command::PrepPq(_) => unsupported("prep-pq"),
        Command::ExtractQuery(_) => unsupported("extract-query"),
        Command::AdjustEdgeSize(_) => unsupported("adjust-edge-size"),
    };

    if let Err(err) = result {
        eprintln!("munind: Error: {err}");
        std::process::exit(1);
    }
}

fn unsupported(command: &str) -> Result<(), String> {
    Err(format!(
        "command `{command}` is not implemented in the Rust port yet"
    ))
}

fn create(args: CreateArgs, debug: u32) -> Result<(), String> {
    let mut property = IndexProperty::new(args.dimension);
    property.thread_pool_size = args.thread_pool_size;
    property.edge_size_for_creation = args.edge_size_for_creation;
    property.edge_size_for_search = args.edge_size_for_search;
    property.batch_size_for_creation = args.batch_size_for_creation;
    property.insertion_radius_coefficient = args.epsilon as f64 + 1.0;
    property.truncation_threshold = args.truncation_threshold;
    property.path_adjustment_interval = args.path_adjustment_interval;
    property.dynamic_edge_size_base = args.dynamic_edge_size_base;
    property.build_time_limit = args.build_time_limit;
    property.clipping_rate = args.clipping_rate;
    property.distance_type = parse_distance_type(&args.distance_type)?;
    property.index_type = parse_index_type(&args.index_type)?;
    property.graph_type = parse_graph_type(&args.graph_type)?;
    property.object_alignment = parse_object_alignment(&args.object_alignment)?;
    property.seed_type = parse_seed_type(&args.seed_type)?;
    property.epsilon_type = parse_epsilon_type(&args.epsilon_type)?;
    property.identical_object_edge_type =
        parse_identical_object_edge_type(&args.identical_object_edge_type)?;
    apply_leaf_internal_sizes(&mut property, args.leaf_internal_sizes.as_deref())?;
    apply_outgoing_incoming(&mut property, args.outgoing_incoming.as_deref())?;
    apply_insertion_order(&mut property, args.insertion_order.as_deref())?;
    apply_object_type(&mut property, &args.object_type)?;

    if debug >= 1 {
        eprintln!("munind: command=create");
        eprintln!("munind: dimension={}", property.dimension);
        eprintln!("munind: index_type={:?}", property.index_type);
        eprintln!("munind: distance_type={:?}", property.distance_type);
    }

    let mut index = Index::create(&args.index, property)?;
    if let Some(data) = args.data.as_ref() {
        load_vectors_into_index(data, index.property.dimension, &mut index)?;
        index.build_index();
    }
    save_auto(&mut index, &args.index)?;
    if !args.quiet {
        eprintln!(
            "munind: created index at {} with {} objects",
            args.index.display(),
            index.object_count()
        );
    }
    Ok(())
}

fn append(args: AppendArgs, debug: u32) -> Result<(), String> {
    let mut index = open_auto(&args.index)?;
    if let Some(data) = args.data.as_ref() {
        load_vectors_into_index(data, index.property.dimension, &mut index)?;
    }
    index.build_index();
    save_auto(&mut index, &args.index)?;
    if debug >= 1 {
        eprintln!("munind: command=append");
    }
    if !args.quiet {
        eprintln!(
            "munind: appended objects, index now contains {} objects",
            index.object_count()
        );
    }
    Ok(())
}

fn search(args: SearchArgs, debug: u32) -> Result<(), String> {
    let index = open_auto(&args.index)?;
    let queries = load_vectors(&args.query, index.property.dimension)?;
    let mut total = 0.0f64;

    for (query_no, query) in queries.iter().enumerate() {
        let started = Instant::now();
        let results = index.search(
            query,
            &SearchOptions {
                k: args.size,
                epsilon: args.epsilon,
                edge_size: Some(args.edge_size),
            },
        )?;
        total += started.elapsed().as_secs_f64() * 1000.0;
        print_query_results(query_no + 1, &results, &args.output_mode);
    }

    let average = if queries.is_empty() {
        0.0
    } else {
        total / queries.len() as f64
    };
    println!(
        "Average Query Time=0.0 (sec), {:.6} (msec), queries={}",
        average,
        queries.len()
    );

    if debug >= 1 {
        eprintln!("munind: command=search");
        eprintln!("munind: queries={}", queries.len());
    }
    Ok(())
}

fn info(args: InfoArgs) -> Result<(), String> {
    let index = open_auto(&args.index)?;
    println!("munind index: {}", args.index.display());
    println!("Dimension: {}", index.property.dimension);
    println!("Objects: {}", index.object_count());
    println!("IndexType: {:?}", index.property.index_type);
    println!("DistanceType: {:?}", index.property.distance_type);
    println!(
        "EdgeSizeForCreation: {}",
        index.property.edge_size_for_creation
    );
    println!("EdgeSizeForSearch: {}", index.property.edge_size_for_search);
    println!("LeafNodeSize: {}", index.property.leaf_node_size);
    println!(
        "InternalChildrenSize: {}",
        index.property.internal_children_size
    );
    println!("GraphNodes: {}", index.graph.edges.len());
    println!(
        "TreeLeaves: {}",
        index
            .tree
            .as_ref()
            .map(|tree| tree.leaves.len())
            .unwrap_or(0)
    );
    Ok(())
}

fn export_index(args: TransferArgs) -> Result<(), String> {
    let mut index = open_auto(&args.index)?;
    if is_index_dir(&args.file) || !args.file.extension().is_some() {
        index.save_as_directory(&args.file)
    } else {
        index.save(Some(&args.file))
    }
}

fn import_index(args: TransferArgs) -> Result<(), String> {
    let mut index = open_auto(&args.file)?;
    save_auto(&mut index, &args.index)
}

fn export_graph(args: IndexOnlyArgs) -> Result<(), String> {
    let index = open_auto(&args.index)?;
    for (node_idx, edges) in index.graph.edges.iter().enumerate() {
        print!("{}", node_idx + 1);
        for edge in edges {
            print!("\t{}\t{}", edge.id, edge.distance);
        }
        println!();
    }
    Ok(())
}

fn export_objects(args: IndexOnlyArgs) -> Result<(), String> {
    let index = open_auto(&args.index)?;
    for object in index.all_objects() {
        let line = object
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join("\t");
        println!("{line}");
    }
    Ok(())
}

fn rebuild(args: IndexOnlyArgs) -> Result<(), String> {
    let mut index = open_auto(&args.index)?;
    index.build_index();
    save_auto(&mut index, &args.index)
}

fn open_auto(path: &Path) -> Result<Index, String> {
    if is_index_dir(path) {
        Index::open_directory(path)
    } else {
        Index::open(path)
    }
}

fn save_auto(index: &mut Index, path: &Path) -> Result<(), String> {
    if is_index_dir(path) {
        index.save_as_directory(path)
    } else {
        index.save(Some(path))
    }
}

fn is_index_dir(path: &Path) -> bool {
    path.is_dir() || path.extension().is_none()
}

fn load_vectors(path: &Path, dimension: usize) -> Result<Vec<Vec<f32>>, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let reader = BufReader::new(file);
    let mut vectors = Vec::new();
    for (lineno, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| e.to_string())?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let values = trimmed
            .split(|c: char| c == '\t' || c == ',' || c.is_ascii_whitespace())
            .filter(|token| !token.is_empty())
            .map(|token| {
                token.parse::<f32>().map_err(|err| {
                    format!(
                        "failed to parse float at {}:{}: {err}",
                        path.display(),
                        lineno + 1
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if values.len() != dimension {
            return Err(format!(
                "invalid vector size at {}:{}: expected {}, got {}",
                path.display(),
                lineno + 1,
                dimension,
                values.len()
            ));
        }
        vectors.push(values);
    }
    Ok(vectors)
}

fn load_vectors_into_index(path: &Path, dimension: usize, index: &mut Index) -> Result<(), String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let reader = BufReader::new(file);
    for (lineno, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| e.to_string())?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let values = trimmed
            .split(|c: char| c == '\t' || c == ',' || c.is_ascii_whitespace())
            .filter(|token| !token.is_empty())
            .map(|token| {
                token.parse::<f32>().map_err(|err| {
                    format!(
                        "failed to parse float at {}:{}: {err}",
                        path.display(),
                        lineno + 1
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if values.len() != dimension {
            return Err(format!(
                "invalid vector size at {}:{}: expected {}, got {}",
                path.display(),
                lineno + 1,
                dimension,
                values.len()
            ));
        }
        index.insert(&values)?;
    }
    Ok(())
}

fn print_query_results(query_no: usize, results: &[ObjectDistance], output_mode: &str) {
    if output_mode != "t" {
        println!("Query No.={query_no}");
    }
    for (rank, result) in results.iter().enumerate() {
        println!("{} {} {}", rank + 1, result.id, result.distance);
    }
}

fn parse_distance_type(value: &str) -> Result<IndexDistanceType, String> {
    match value {
        "1" => Ok(IndexDistanceType::L1),
        "2" | "e" => Ok(IndexDistanceType::L2),
        "c" => Ok(IndexDistanceType::Cosine),
        "a" => Ok(IndexDistanceType::Angle),
        "p" => Ok(IndexDistanceType::DotProduct),
        other => Err(format!("unsupported distance type: {other}")),
    }
}

fn parse_index_type(value: &str) -> Result<IndexType, String> {
    match value {
        "t" => Ok(IndexType::GraphAndTree),
        "g" => Ok(IndexType::Graph),
        other => Err(format!("unsupported index type: {other}")),
    }
}

fn parse_graph_type(value: &str) -> Result<munind::index::GraphType, String> {
    match value {
        "a" => Ok(munind::index::GraphType::ANNG),
        "k" => Ok(munind::index::GraphType::KNNG),
        "b" => Ok(munind::index::GraphType::BKNNG),
        "o" => Ok(munind::index::GraphType::ONNG),
        "i" => Ok(munind::index::GraphType::IANNG),
        "d" => Ok(munind::index::GraphType::DNNG),
        "r" => Ok(munind::index::GraphType::RANNG),
        "R" => Ok(munind::index::GraphType::RIANNG),
        other => Err(format!("unsupported graph type: {other}")),
    }
}

fn parse_object_alignment(value: &str) -> Result<munind::index::ObjectAlignment, String> {
    match value {
        "t" => Ok(munind::index::ObjectAlignment::True),
        "f" => Ok(munind::index::ObjectAlignment::False),
        other => Err(format!("unsupported object alignment: {other}")),
    }
}

fn parse_seed_type(value: &str) -> Result<munind::index::SeedType, String> {
    let mode = value.chars().next().unwrap_or('-');
    match mode {
        'f' => Ok(munind::index::SeedType::FixedNodes),
        '1' => Ok(munind::index::SeedType::FirstNode),
        'r' => Ok(munind::index::SeedType::RandomNodes),
        'l' => Ok(munind::index::SeedType::AllLeafNodes),
        '-' => Ok(munind::index::SeedType::None),
        other => Err(format!("unsupported seed type: {other}")),
    }
}

fn parse_epsilon_type(value: &str) -> Result<munind::index::EpsilonType, String> {
    match value {
        "q" => Ok(munind::index::EpsilonType::ByQuery),
        "n" | "-" => Ok(munind::index::EpsilonType::None),
        other => Err(format!("unsupported epsilon type: {other}")),
    }
}

fn parse_identical_object_edge_type(
    value: &str,
) -> Result<munind::index::IdenticalObjectEdgeType, String> {
    match value {
        "d" => Ok(munind::index::IdenticalObjectEdgeType::DirectedEdge),
        "u" => Ok(munind::index::IdenticalObjectEdgeType::UndirectedEdge),
        "-" => Ok(munind::index::IdenticalObjectEdgeType::None),
        other => Err(format!("unsupported identical-object-edge type: {other}")),
    }
}

fn apply_leaf_internal_sizes(
    property: &mut IndexProperty,
    value: Option<&str>,
) -> Result<(), String> {
    if let Some(value) = value {
        let parts: Vec<_> = value.split(':').collect();
        match parts.as_slice() {
            [leaf] => {
                property.leaf_node_size = leaf
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?;
            }
            [leaf, internal] => {
                property.leaf_node_size = leaf
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?;
                property.internal_children_size = internal
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?;
            }
            _ => return Err(format!("invalid -L value: {value}")),
        }
    }
    Ok(())
}

fn apply_outgoing_incoming(
    property: &mut IndexProperty,
    value: Option<&str>,
) -> Result<(), String> {
    if let Some(value) = value {
        let parts: Vec<_> = value.split('x').collect();
        if parts.len() != 2 {
            return Err(format!("invalid -O value: {value}"));
        }
        property.outgoing_edge = parts[0]
            .parse()
            .map_err(|e: std::num::ParseIntError| e.to_string())?;
        property.incoming_edge = parts[1]
            .parse()
            .map_err(|e: std::num::ParseIntError| e.to_string())?;
    } else if matches!(
        property.graph_type,
        munind::index::GraphType::ANNG
            | munind::index::GraphType::ONNG
            | munind::index::GraphType::IANNG
            | munind::index::GraphType::RANNG
            | munind::index::GraphType::RIANNG
    ) {
        property.outgoing_edge = 10;
        property.incoming_edge = 100;
    }
    Ok(())
}

fn apply_insertion_order(property: &mut IndexProperty, value: Option<&str>) -> Result<(), String> {
    if let Some(value) = value {
        let parts: Vec<_> = value.split(':').collect();
        match parts.as_slice() {
            [neighbors] => {
                property.number_of_neighbors_for_insertion_order = neighbors
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?;
            }
            [neighbors, epsilon] => {
                property.number_of_neighbors_for_insertion_order = neighbors
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?;
                property.epsilon_for_insertion_order = epsilon
                    .parse()
                    .map_err(|e: std::num::ParseFloatError| e.to_string())?;
            }
            _ => return Err(format!("invalid -l value: {value}")),
        }
    }
    Ok(())
}

fn apply_object_type(property: &mut IndexProperty, value: &str) -> Result<(), String> {
    match value {
        "f" => {
            property.object_type = munind::object_space::ObjectType::Float;
            Ok(())
        }
        "c" => {
            property.object_type = munind::object_space::ObjectType::Uint8;
            Ok(())
        }
        other => Err(format!("unsupported object type: {other}")),
    }
}
