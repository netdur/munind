use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use rand::Rng;
use rand::thread_rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;

use crate::graph::{NeighborhoodGraph, SearchContainer};
use crate::mmap_index::{GRAPH_MAGIC, OBJECT_HEADER_SIZE, OBJECT_MAGIC};
use crate::node::ObjectDistance;
use crate::object_space::{DistanceType, ObjectSpace, ObjectType};
use crate::tree::DvpTree;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexType {
    None,
    GraphAndTree,
    Graph,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatabaseType {
    None,
    Memory,
    MemoryMappedFile,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObjectAlignment {
    None,
    True,
    False,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphType {
    None,
    ANNG,
    KNNG,
    BKNNG,
    ONNG,
    IANNG,
    DNNG,
    RANNG,
    RIANNG,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SeedType {
    None,
    RandomNodes,
    FixedNodes,
    FirstNode,
    AllLeafNodes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EpsilonType {
    None,
    ByQuery,
    ResultSize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum IdenticalObjectEdgeType {
    None,
    DirectedEdge,
    UndirectedEdge,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum IndexDistanceType {
    L1,
    L2,
    Cosine,
    Angle,
    DotProduct,
}

impl Into<DistanceType> for IndexDistanceType {
    fn into(self) -> DistanceType {
        match self {
            IndexDistanceType::L1 => DistanceType::L1,
            IndexDistanceType::L2 => DistanceType::L2,
            IndexDistanceType::Cosine => DistanceType::Cosine,
            IndexDistanceType::Angle => DistanceType::Angle,
            IndexDistanceType::DotProduct => DistanceType::DotProduct,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IndexProperty {
    pub dimension: usize,
    pub thread_pool_size: usize,
    pub distance_type: IndexDistanceType,
    pub object_type: ObjectType,
    pub index_type: IndexType,
    pub database_type: DatabaseType,
    pub object_alignment: ObjectAlignment,
    pub path_adjustment_interval: isize,
    pub prefetch_offset: isize,
    pub prefetch_size: isize,
    pub max_magnitude: f32,
    pub quantization_scale: f32,
    pub quantization_offset: f32,
    pub clipping_rate: f32,
    pub number_of_neighbors_for_insertion_order: isize,
    pub epsilon_for_insertion_order: f32,
    pub leaf_node_size: usize,
    pub internal_children_size: usize,
    pub truncation_threshold: isize,
    pub edge_size_for_creation: usize,
    pub edge_size_for_search: usize,
    pub edge_size_limit_for_creation: usize,
    pub insertion_radius_coefficient: f64,
    pub seed_size: usize,
    pub seed_type: SeedType,
    pub truncation_thread_pool_size: usize,
    pub batch_size_for_creation: usize,
    pub graph_type: GraphType,
    pub dynamic_edge_size_base: usize,
    pub dynamic_edge_size_rate: usize,
    pub build_time_limit: f32,
    pub outgoing_edge: usize,
    pub incoming_edge: usize,
    pub epsilon_type: EpsilonType,
    pub identical_object_edge_type: IdenticalObjectEdgeType,
}

impl IndexProperty {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            thread_pool_size: 32,
            distance_type: IndexDistanceType::L2,
            object_type: ObjectType::Float,
            index_type: IndexType::GraphAndTree,
            database_type: DatabaseType::Memory,
            object_alignment: ObjectAlignment::False,
            path_adjustment_interval: 0,
            prefetch_offset: 0,
            prefetch_size: 0,
            max_magnitude: -1.0,
            quantization_scale: 0.0,
            quantization_offset: 0.0,
            clipping_rate: 0.0,
            number_of_neighbors_for_insertion_order: 0,
            epsilon_for_insertion_order: 0.1,
            leaf_node_size: 100,
            internal_children_size: 5,
            truncation_threshold: 0,
            edge_size_for_creation: 10,
            edge_size_for_search: 0,
            edge_size_limit_for_creation: 5,
            insertion_radius_coefficient: 1.1,
            seed_size: 10,
            seed_type: SeedType::None,
            truncation_thread_pool_size: 8,
            batch_size_for_creation: 200,
            graph_type: GraphType::ANNG,
            dynamic_edge_size_base: 30,
            dynamic_edge_size_rate: 20,
            build_time_limit: 0.0,
            outgoing_edge: 10,
            incoming_edge: 80,
            epsilon_type: EpsilonType::None,
            identical_object_edge_type: IdenticalObjectEdgeType::None,
        }
    }

    pub fn set_distance_type(&mut self, dt: IndexDistanceType) {
        self.distance_type = dt;
    }
}

#[derive(Clone, Debug)]
pub struct SearchOptions {
    pub k: usize,
    pub epsilon: f32,
    pub edge_size: Option<isize>,
}

struct IndexRandom {
    state: u32,
}

impl IndexRandom {
    fn new(seed: u32) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        let value = (self.state / 65536) % 32768;
        (f64::from(value) + 1.0) / 32769.0
    }

    fn pick_index(&mut self, upper: usize) -> usize {
        ((upper as f64) * self.next_f64()).floor() as usize
    }
}

#[derive(Serialize, Deserialize)]
pub struct Index {
    pub property: IndexProperty,
    pub objects: Vec<Vec<f32>>,
    #[serde(skip)]
    pub object_space: Option<ObjectSpace>,
    pub graph: NeighborhoodGraph,
    pub tree: Option<DvpTree>,
    pub path: String,
}

#[derive(Serialize)]
struct SerializableIndex<'a> {
    property: &'a IndexProperty,
    objects: Vec<Vec<f32>>,
    graph: &'a NeighborhoodGraph,
    tree: &'a Option<DvpTree>,
    path: &'a String,
}

impl Index {
    pub fn create<P: AsRef<Path>>(path: P, property: IndexProperty) -> Result<Self, String> {
        let object_space = ObjectSpace::new(
            property.dimension,
            property.distance_type.into(),
            property.object_type,
        );

        let instance = Self {
            property,
            objects: vec![],
            object_space: Some(object_space),
            graph: NeighborhoodGraph::new(),
            tree: None,
            path: path.as_ref().to_string_lossy().into_owned(),
        };
        Ok(instance)
    }

    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| e.to_string())?;
        let reader = BufReader::new(file);
        let mut index: Index = bincode::deserialize_from(reader).map_err(|e| e.to_string())?;

        let mut object_space = ObjectSpace::new(
            index.property.dimension,
            index.property.distance_type.into(),
            index.property.object_type,
        );
        for obj in index.objects.iter() {
            object_space.insert_prepared(obj.clone())?;
        }
        index.object_space = Some(object_space);
        Ok(index)
    }

    fn reconstruct_object_space(&mut self) -> Result<(), String> {
        let mut object_space = ObjectSpace::new(
            self.property.dimension,
            self.property.distance_type.into(),
            self.property.object_type,
        );
        for obj in self.objects.iter() {
            object_space.insert_prepared(obj.clone())?;
        }
        self.object_space = Some(object_space);
        Ok(())
    }

    pub fn save<P: AsRef<Path>>(&mut self, path: Option<P>) -> Result<(), String> {
        let target_path = if let Some(p) = path {
            p.as_ref().to_string_lossy().into_owned()
        } else {
            self.path.clone()
        };

        let file = File::create(target_path).map_err(|e| e.to_string())?;
        let writer = BufWriter::new(file);
        let snapshot = self.serializable_snapshot();
        bincode::serialize_into(writer, &snapshot).map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn save_as_directory<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let path = path.as_ref();
        fs::create_dir_all(path).map_err(|e| e.to_string())?;

        self.save_property_file(path.join("prf"))?;
        self.serialize_into_file(path.join("obj"), &self.materialize_objects())?;
        self.serialize_into_file(path.join("grp"), &self.graph)?;
        self.serialize_into_file(path.join("tre"), &self.tree)?;

        let robj = File::create(path.join("robj")).map_err(|e| e.to_string())?;
        let mut writer = BufWriter::new(robj);
        writer
            .write_all(&0u64.to_le_bytes())
            .map_err(|e| e.to_string())?;
        writer.flush().map_err(|e| e.to_string())
    }

    pub fn save_as_mmap<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let path = path.as_ref();
        fs::create_dir_all(path).map_err(|e| e.to_string())?;

        self.save_property_file(path.join("prf"))?;
        self.save_mmap_objects(path.join("obj.mmap"))?;
        self.save_mmap_graph(path.join("grp.mmap"))?;
        self.serialize_into_file(path.join("tre.bin"), &self.tree)?;
        Ok(())
    }

    pub fn open_directory<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path = path.as_ref();
        let property = Self::load_property_file(path.join("prf"))?;
        let objects: Vec<Vec<f32>> = Self::deserialize_from_file(path.join("obj"))?;
        let graph: NeighborhoodGraph = Self::deserialize_from_file(path.join("grp"))?;
        let tree: Option<DvpTree> = Self::deserialize_from_file(path.join("tre"))?;

        let mut index = Self {
            property,
            objects,
            object_space: None,
            graph,
            tree,
            path: path.to_string_lossy().into_owned(),
        };
        index.reconstruct_object_space()?;
        Ok(index)
    }

    fn serialize_into_file<T: Serialize, P: AsRef<Path>>(
        &self,
        path: P,
        value: &T,
    ) -> Result<(), String> {
        let file = File::create(path).map_err(|e| e.to_string())?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, value).map_err(|e| e.to_string())
    }

    fn deserialize_from_file<T: for<'de> Deserialize<'de>, P: AsRef<Path>>(
        path: P,
    ) -> Result<T, String> {
        let file = File::open(path).map_err(|e| e.to_string())?;
        let reader = BufReader::new(file);
        bincode::deserialize_from(reader).map_err(|e| e.to_string())
    }

    pub(crate) fn save_property_file<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let mut lines = Vec::new();
        lines.push(("AccuracyTable", "".to_string()));
        lines.push((
            "BatchSizeForCreation",
            self.property.batch_size_for_creation.to_string(),
        ));
        lines.push(("BuildTimeLimit", self.property.build_time_limit.to_string()));
        lines.push((
            "DatabaseType",
            match self.property.database_type {
                DatabaseType::None => "None",
                DatabaseType::Memory => "Memory",
                DatabaseType::MemoryMappedFile => "MemoryMappedFile",
            }
            .to_string(),
        ));
        lines.push(("Dimension", self.property.dimension.to_string()));
        lines.push((
            "DistanceType",
            match self.property.distance_type {
                IndexDistanceType::L1 => "L1",
                IndexDistanceType::L2 => "L2",
                IndexDistanceType::Cosine => "Cosine",
                IndexDistanceType::Angle => "Angle",
                IndexDistanceType::DotProduct => "DotProduct",
            }
            .to_string(),
        ));
        lines.push((
            "DynamicEdgeSizeBase",
            self.property.dynamic_edge_size_base.to_string(),
        ));
        lines.push((
            "DynamicEdgeSizeRate",
            self.property.dynamic_edge_size_rate.to_string(),
        ));
        lines.push((
            "EdgeSizeForCreation",
            self.property.edge_size_for_creation.to_string(),
        ));
        lines.push((
            "EdgeSizeForSearch",
            self.property.edge_size_for_search.to_string(),
        ));
        lines.push((
            "EdgeSizeLimitForCreation",
            self.property.edge_size_limit_for_creation.to_string(),
        ));
        lines.push((
            "EpsilonForCreation",
            (self.property.insertion_radius_coefficient - 1.0).to_string(),
        ));
        lines.push((
            "EpsilonForInsertionOrder",
            self.property.epsilon_for_insertion_order.to_string(),
        ));
        lines.push((
            "EpsilonType",
            match self.property.epsilon_type {
                EpsilonType::None => "None",
                EpsilonType::ByQuery => "ByQuery",
                EpsilonType::ResultSize => "ResultSize",
            }
            .to_string(),
        ));
        lines.push((
            "GraphType",
            match self.property.graph_type {
                GraphType::None => "None",
                GraphType::ANNG => "ANNG",
                GraphType::KNNG => "KNNG",
                GraphType::BKNNG => "BKNNG",
                GraphType::ONNG => "ONNG",
                GraphType::IANNG => "IANNG",
                GraphType::DNNG => "DNNG",
                GraphType::RANNG => "RANNG",
                GraphType::RIANNG => "RIANNG",
            }
            .to_string(),
        ));
        lines.push((
            "IdenticalObjectEdgeType",
            match self.property.identical_object_edge_type {
                IdenticalObjectEdgeType::None => "None",
                IdenticalObjectEdgeType::DirectedEdge => "DirectedEdge",
                IdenticalObjectEdgeType::UndirectedEdge => "UndirectedEdge",
            }
            .to_string(),
        ));
        lines.push(("IncomingEdge", self.property.incoming_edge.to_string()));
        lines.push((
            "IncrimentalEdgeSizeLimitForTruncation",
            self.property.truncation_threshold.to_string(),
        ));
        lines.push((
            "IndexType",
            match self.property.index_type {
                IndexType::None => "None",
                IndexType::GraphAndTree => "GraphAndTree",
                IndexType::Graph => "Graph",
            }
            .to_string(),
        ));
        lines.push((
            "InternalChildrenSize",
            self.property.internal_children_size.to_string(),
        ));
        lines.push(("LeafNodeSize", self.property.leaf_node_size.to_string()));
        lines.push(("MaxMagnitude", self.property.max_magnitude.to_string()));
        lines.push((
            "NumberOfNeighborsForInsertionOrder",
            self.property
                .number_of_neighbors_for_insertion_order
                .to_string(),
        ));
        lines.push((
            "ObjectAlignment",
            match self.property.object_alignment {
                ObjectAlignment::None => "None",
                ObjectAlignment::True => "True",
                ObjectAlignment::False => "False",
            }
            .to_string(),
        ));
        lines.push((
            "ObjectType",
            match self.property.object_type {
                ObjectType::Uint8 => "Integer-1",
                ObjectType::Float => "Float-4",
                ObjectType::None => "None",
                ObjectType::Unset => "Unset",
            }
            .to_string(),
        ));
        lines.push(("OutgoingEdge", self.property.outgoing_edge.to_string()));
        lines.push((
            "PathAdjustmentInterval",
            self.property.path_adjustment_interval.to_string(),
        ));
        lines.push(("PrefetchOffset", self.property.prefetch_offset.to_string()));
        lines.push(("PrefetchSize", self.property.prefetch_size.to_string()));
        lines.push((
            "QuantizationClippingRate",
            self.property.clipping_rate.to_string(),
        ));
        lines.push((
            "QuantizationOffset",
            self.property.quantization_offset.to_string(),
        ));
        lines.push((
            "QuantizationScale",
            self.property.quantization_scale.to_string(),
        ));
        lines.push(("RefinementObjectType", "Float-4".to_string()));
        lines.push(("SeedSize", self.property.seed_size.to_string()));
        lines.push((
            "SeedType",
            match self.property.seed_type {
                SeedType::None => "None",
                SeedType::RandomNodes => "RandomNodes",
                SeedType::FixedNodes => "FixedNodes",
                SeedType::FirstNode => "FirstNode",
                SeedType::AllLeafNodes => "AllLeafNodes",
            }
            .to_string(),
        ));
        lines.push(("ThreadPoolSize", self.property.thread_pool_size.to_string()));
        lines.push((
            "TruncationThreadPoolSize",
            self.property.truncation_thread_pool_size.to_string(),
        ));

        let file = File::create(path).map_err(|e| e.to_string())?;
        let mut writer = BufWriter::new(file);
        for (key, value) in lines {
            writer
                .write_all(format!("{key}\t{value}\n").as_bytes())
                .map_err(|e| e.to_string())?;
        }
        writer.flush().map_err(|e| e.to_string())
    }

    pub(crate) fn load_property_file<P: AsRef<Path>>(path: P) -> Result<IndexProperty, String> {
        let file = File::open(path).map_err(|e| e.to_string())?;
        let reader = BufReader::new(file);
        let mut property = IndexProperty::new(0);
        for line in reader.lines() {
            let line = line.map_err(|e| e.to_string())?;
            if line.trim().is_empty() {
                continue;
            }
            let mut parts = line.splitn(2, '\t');
            let key = parts.next().unwrap_or_default();
            let value = parts.next().unwrap_or_default().trim();
            match key {
                "Dimension" => {
                    property.dimension = value
                        .parse()
                        .map_err(|e: std::num::ParseIntError| e.to_string())?
                }
                "ThreadPoolSize" => {
                    property.thread_pool_size = value
                        .parse()
                        .map_err(|e: std::num::ParseIntError| e.to_string())?
                }
                "DatabaseType" => {
                    property.database_type = match value {
                        "None" => DatabaseType::None,
                        "Memory" => DatabaseType::Memory,
                        "MemoryMappedFile" => DatabaseType::MemoryMappedFile,
                        other => return Err(format!("Unsupported DatabaseType: {other}")),
                    }
                }
                "DistanceType" => {
                    property.distance_type = match value {
                        "L1" => IndexDistanceType::L1,
                        "L2" => IndexDistanceType::L2,
                        "Cosine" => IndexDistanceType::Cosine,
                        "Angle" => IndexDistanceType::Angle,
                        "DotProduct" => IndexDistanceType::DotProduct,
                        other => return Err(format!("Unsupported DistanceType: {other}")),
                    }
                }
                "EdgeSizeForCreation" => {
                    property.edge_size_for_creation = value
                        .parse()
                        .map_err(|e: std::num::ParseIntError| e.to_string())?
                }
                "EdgeSizeForSearch" => {
                    property.edge_size_for_search = value
                        .parse()
                        .map_err(|e: std::num::ParseIntError| e.to_string())?
                }
                "EdgeSizeLimitForCreation" => {
                    property.edge_size_limit_for_creation = value
                        .parse()
                        .map_err(|e: std::num::ParseIntError| e.to_string())?
                }
                "EpsilonForCreation" => {
                    let epsilon_for_creation: f64 = value
                        .parse()
                        .map_err(|e: std::num::ParseFloatError| e.to_string())?;
                    property.insertion_radius_coefficient = epsilon_for_creation + 1.0;
                }
                "EpsilonForInsertionOrder" => {
                    property.epsilon_for_insertion_order = value
                        .parse()
                        .map_err(|e: std::num::ParseFloatError| e.to_string())?
                }
                "EpsilonType" => {
                    property.epsilon_type = match value {
                        "None" => EpsilonType::None,
                        "ByQuery" => EpsilonType::ByQuery,
                        "ResultSize" => EpsilonType::ResultSize,
                        other => return Err(format!("Unsupported EpsilonType: {other}")),
                    }
                }
                "LeafNodeSize" => {
                    property.leaf_node_size = value
                        .parse()
                        .map_err(|e: std::num::ParseIntError| e.to_string())?
                }
                "InternalChildrenSize" => {
                    property.internal_children_size = value
                        .parse()
                        .map_err(|e: std::num::ParseIntError| e.to_string())?
                }
                "GraphType" => {
                    property.graph_type = match value {
                        "ANNG" => GraphType::ANNG,
                        "KNNG" => GraphType::KNNG,
                        "BKNNG" => GraphType::BKNNG,
                        "ONNG" => GraphType::ONNG,
                        "IANNG" => GraphType::IANNG,
                        "DNNG" => GraphType::DNNG,
                        "RANNG" => GraphType::RANNG,
                        "RIANNG" => GraphType::RIANNG,
                        "None" => GraphType::None,
                        other => return Err(format!("Unsupported GraphType: {other}")),
                    }
                }
                "IndexType" => {
                    property.index_type = match value {
                        "GraphAndTree" => IndexType::GraphAndTree,
                        "Graph" => IndexType::Graph,
                        "None" => IndexType::None,
                        other => return Err(format!("Unsupported IndexType: {other}")),
                    }
                }
                "ObjectType" => {
                    property.object_type = match value {
                        "Integer-1" => ObjectType::Uint8,
                        "Float-4" => ObjectType::Float,
                        "Unset" => ObjectType::Unset,
                        "None" => ObjectType::None,
                        other => return Err(format!("Unsupported ObjectType: {other}")),
                    }
                }
                "ObjectAlignment" => {
                    property.object_alignment = match value {
                        "None" => ObjectAlignment::None,
                        "True" => ObjectAlignment::True,
                        "False" => ObjectAlignment::False,
                        other => return Err(format!("Unsupported ObjectAlignment: {other}")),
                    }
                }
                "PathAdjustmentInterval" => {
                    property.path_adjustment_interval = value
                        .parse()
                        .map_err(|e: std::num::ParseIntError| e.to_string())?
                }
                "PrefetchOffset" => {
                    property.prefetch_offset = value
                        .parse()
                        .map_err(|e: std::num::ParseIntError| e.to_string())?
                }
                "PrefetchSize" => {
                    property.prefetch_size = value
                        .parse()
                        .map_err(|e: std::num::ParseIntError| e.to_string())?
                }
                "MaxMagnitude" => {
                    property.max_magnitude = value
                        .parse()
                        .map_err(|e: std::num::ParseFloatError| e.to_string())?
                }
                "QuantizationScale" => {
                    property.quantization_scale = value
                        .parse()
                        .map_err(|e: std::num::ParseFloatError| e.to_string())?
                }
                "QuantizationOffset" => {
                    property.quantization_offset = value
                        .parse()
                        .map_err(|e: std::num::ParseFloatError| e.to_string())?
                }
                "QuantizationClippingRate" => {
                    property.clipping_rate = value
                        .parse()
                        .map_err(|e: std::num::ParseFloatError| e.to_string())?
                }
                "NumberOfNeighborsForInsertionOrder" => {
                    property.number_of_neighbors_for_insertion_order = value
                        .parse()
                        .map_err(|e: std::num::ParseIntError| e.to_string())?
                }
                "SeedSize" => {
                    property.seed_size = value
                        .parse()
                        .map_err(|e: std::num::ParseIntError| e.to_string())?
                }
                "SeedType" => {
                    property.seed_type = match value {
                        "None" => SeedType::None,
                        "RandomNodes" => SeedType::RandomNodes,
                        "FixedNodes" => SeedType::FixedNodes,
                        "FirstNode" => SeedType::FirstNode,
                        "AllLeafNodes" => SeedType::AllLeafNodes,
                        other => return Err(format!("Unsupported SeedType: {other}")),
                    }
                }
                "BatchSizeForCreation" => {
                    property.batch_size_for_creation = value
                        .parse()
                        .map_err(|e: std::num::ParseIntError| e.to_string())?
                }
                "TruncationThreadPoolSize" => {
                    property.truncation_thread_pool_size = value
                        .parse()
                        .map_err(|e: std::num::ParseIntError| e.to_string())?
                }
                "IncrimentalEdgeSizeLimitForTruncation" => {
                    property.truncation_threshold = value
                        .parse()
                        .map_err(|e: std::num::ParseIntError| e.to_string())?
                }
                "DynamicEdgeSizeBase" => {
                    property.dynamic_edge_size_base = value
                        .parse()
                        .map_err(|e: std::num::ParseIntError| e.to_string())?
                }
                "DynamicEdgeSizeRate" => {
                    property.dynamic_edge_size_rate = value
                        .parse()
                        .map_err(|e: std::num::ParseIntError| e.to_string())?
                }
                "BuildTimeLimit" => {
                    property.build_time_limit = value
                        .parse()
                        .map_err(|e: std::num::ParseFloatError| e.to_string())?
                }
                "OutgoingEdge" => {
                    property.outgoing_edge = value
                        .parse()
                        .map_err(|e: std::num::ParseIntError| e.to_string())?
                }
                "IncomingEdge" => {
                    property.incoming_edge = value
                        .parse()
                        .map_err(|e: std::num::ParseIntError| e.to_string())?
                }
                "IdenticalObjectEdgeType" => {
                    property.identical_object_edge_type = match value {
                        "None" => IdenticalObjectEdgeType::None,
                        "DirectedEdge" => IdenticalObjectEdgeType::DirectedEdge,
                        "UndirectedEdge" => IdenticalObjectEdgeType::UndirectedEdge,
                        other => {
                            return Err(format!("Unsupported IdenticalObjectEdgeType: {other}"));
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(property)
    }

    pub fn insert(&mut self, object: &[f32]) -> Result<usize, String> {
        let os = self
            .object_space
            .as_mut()
            .ok_or_else(|| "Object space is not initialized".to_string())?;
        let stored = os.prepare_for_insert(object)?;
        let id = os.insert_prepared(stored)?;
        self.graph.insert_node(id as u32);
        Ok(id)
    }

    pub fn build_index(&mut self) {
        self.build_index_with_debug(0);
    }

    pub fn build_index_with_debug(&mut self, debug: u32) {
        let mut graph = NeighborhoodGraph::new();
        graph.edge_size_for_creation = self.property.edge_size_for_creation;
        graph.edge_size_for_search = self.property.edge_size_for_search;
        graph.insertion_exploration_coefficient = self.property.insertion_radius_coefficient;
        graph.dynamic_edge_size_base = self.property.dynamic_edge_size_base;
        graph.dynamic_edge_size_rate = self.property.dynamic_edge_size_rate;
        let num_objects = self.object_count();
        graph.edges = vec![Vec::new(); num_objects];

        self.graph = graph;
        self.tree = if matches!(self.property.index_type, IndexType::GraphAndTree) {
            Some(DvpTree::new(
                self.property.leaf_node_size,
                self.property.internal_children_size,
            ))
        } else {
            None
        };

        if self.property.thread_pool_size > 1 && self.property.batch_size_for_creation > 1 {
            let thread_pool = rayon::ThreadPoolBuilder::new()
                .num_threads(self.property.thread_pool_size)
                .build();
            match thread_pool {
                Ok(pool) => pool.install(|| self.build_index_in_batches(num_objects, debug)),
                Err(_) => self.build_index_sequential(num_objects, debug),
            }
        } else {
            self.build_index_sequential(num_objects, debug);
        }
    }

    pub fn search(
        &self,
        query: &[f32],
        options: &SearchOptions,
    ) -> Result<Vec<ObjectDistance>, String> {
        let mut sc = SearchContainer {
            object: query,
            radius: f32::MAX,
            size: options.k,
            exploration_coefficient: 1.0 + options.epsilon as f64,
            edge_size: options.edge_size.unwrap_or(-1),
        };

        let prepared_query = self
            .object_space
            .as_ref()
            .ok_or_else(|| "Object space is not initialized".to_string())?
            .prepare_query(query)?;
        sc.object = &prepared_query;

        let mut seeds = self.get_seeds(sc.object, options.k)?;

        let results = self
            .graph
            .search(self.object_space.as_ref().unwrap(), &mut sc, &mut seeds);
        Ok(results)
    }

    pub fn linear_search(&self, query: &[f32], k: usize) -> Result<Vec<ObjectDistance>, String> {
        let mut results = BinaryHeap::new();
        let os = self
            .object_space
            .as_ref()
            .ok_or_else(|| "Object space is not initialized".to_string())?;
        let prepared_query = os.prepare_query(query)?;

        for i in 1..=self.object_count() {
            let dist = os
                .compare_to_id(&prepared_query, i)
                .ok_or_else(|| format!("Object {i} is missing"))?;
            results.push(crate::graph::MaxDistanceNode(ObjectDistance {
                id: i as u32,
                distance: dist,
            }));
            if results.len() > k {
                results.pop();
            }
        }

        let mut final_res = Vec::with_capacity(results.len());
        while let Some(node) = results.pop() {
            final_res.push(node.0);
        }
        final_res.reverse();
        Ok(final_res)
    }

    fn get_seeds(&self, prepared_query: &[f32], k: usize) -> Result<Vec<ObjectDistance>, String> {
        if matches!(self.property.index_type, IndexType::GraphAndTree) {
            let tree_seeds = self.get_seeds_from_tree(prepared_query, k)?;
            if !tree_seeds.is_empty() {
                return Ok(tree_seeds);
            }
        }

        self.get_seeds_from_graph()
    }

    fn get_seeds_from_tree(
        &self,
        prepared_query: &[f32],
        k: usize,
    ) -> Result<Vec<ObjectDistance>, String> {
        if let Some(tree) = &self.tree {
            if let Some(object_space) = &self.object_space {
                if let Some(leaf_id) = tree.greedy_leaf_for_query(prepared_query, object_space) {
                    let mut seeds = tree.get_object_ids_from_leaf(leaf_id);
                    if !seeds.is_empty() {
                        self.thin_tree_seeds(&mut seeds, false, k);
                    }
                    if !seeds.is_empty() {
                        self.setup_seed_distances(prepared_query, &mut seeds);
                        seeds.sort_by(|a, b| {
                            a.distance
                                .partial_cmp(&b.distance)
                                .unwrap_or(std::cmp::Ordering::Equal)
                                .then_with(|| a.id.cmp(&b.id))
                        });
                        let target_seed_count = self.effective_seed_count(k).max(1);
                        if seeds.len() > target_seed_count {
                            seeds.truncate(target_seed_count);
                        }
                        return Ok(seeds);
                    }
                }
            }
        }
        Ok(Vec::new())
    }

    fn get_seeds_from_graph(&self) -> Result<Vec<ObjectDistance>, String> {
        let repository_size = self.object_count();
        if repository_size == 0 {
            return Ok(Vec::new());
        }

        let mut seeds = Vec::new();
        match self.property.seed_type {
            SeedType::FixedNodes => {
                for id in 1..=self
                    .effective_seed_count(self.property.edge_size_for_creation)
                    .min(repository_size)
                {
                    seeds.push(ObjectDistance {
                        id: id as u32,
                        distance: 0.0,
                    });
                }
            }
            SeedType::FirstNode => {
                seeds.push(ObjectDistance {
                    id: 1,
                    distance: 0.0,
                });
            }
            _ => {
                self.supplement_random_graph_seeds(
                    &mut seeds,
                    self.effective_seed_count(self.property.edge_size_for_creation),
                );
            }
        }
        Ok(seeds)
    }

    fn effective_seed_count(&self, k: usize) -> usize {
        if self.property.seed_size == 0 {
            k.max(1)
        } else {
            self.property.seed_size
        }
    }

    fn thin_tree_seeds(
        &self,
        seeds: &mut Vec<ObjectDistance>,
        use_all_nodes_in_leaf: bool,
        k: usize,
    ) {
        if seeds.is_empty()
            || use_all_nodes_in_leaf
            || matches!(self.property.seed_type, SeedType::AllLeafNodes)
        {
            return;
        }

        match self.property.seed_type {
            SeedType::None => {
                let seed_size = self.effective_seed_count(k).min(k.max(1));
                if seeds.len() > seed_size {
                    self.random_thin_seeds(seeds, seed_size, true);
                }
            }
            SeedType::FixedNodes => {
                let seed_size = self.effective_seed_count(k);
                if seeds.len() > seed_size {
                    seeds.truncate(seed_size);
                }
            }
            SeedType::RandomNodes => {
                let seed_size = self.effective_seed_count(k);
                if seeds.len() > seed_size {
                    self.random_thin_seeds(seeds, seed_size, false);
                }
            }
            SeedType::FirstNode => seeds.truncate(1),
            SeedType::AllLeafNodes => {}
        }
    }

    fn setup_seed_distances(&self, prepared_query: &[f32], seeds: &mut [ObjectDistance]) {
        let Some(object_space) = &self.object_space else {
            return;
        };
        for seed in seeds.iter_mut() {
            seed.distance = object_space
                .compare_to_id(prepared_query, seed.id as usize)
                .unwrap_or(f32::MAX);
        }
    }

    fn random_thin_seeds(
        &self,
        seeds: &mut Vec<ObjectDistance>,
        seed_size: usize,
        deterministic: bool,
    ) {
        if seeds.len() <= seed_size {
            return;
        }
        let mut rng = IndexRandom::new(if deterministic {
            seeds[0].id
        } else {
            rand::random::<u32>()
        });
        for i in (seed_size + 1..=seeds.len()).rev() {
            let idx = rng.pick_index(i);
            seeds[idx] = seeds[i - 1];
        }
        seeds.truncate(seed_size);
    }

    fn supplement_random_graph_seeds(&self, seeds: &mut Vec<ObjectDistance>, requested: usize) {
        let repository_size = self.object_count();
        if repository_size == 0 || seeds.len() >= requested {
            return;
        }
        let needed = requested - seeds.len();
        if needed >= repository_size {
            for id in 1..=repository_size as u32 {
                if !seeds.iter().any(|seed| seed.id == id) {
                    seeds.push(ObjectDistance { id, distance: 0.0 });
                }
            }
            return;
        }
        let mut rng = thread_rng();
        while seeds.len() < requested {
            let id = rng.gen_range(1..=repository_size as u32);
            if !seeds.iter().any(|seed| seed.id == id) {
                seeds.push(ObjectDistance { id, distance: 0.0 });
            }
        }
    }

    fn get_insertion_seeds(&self, id: usize, use_all_nodes_in_leaf: bool) -> Vec<ObjectDistance> {
        if id <= 1 {
            return Vec::new();
        }
        if matches!(self.property.index_type, IndexType::GraphAndTree) {
            if let (Some(tree), Some(object_space)) = (&self.tree, &self.object_space) {
                if !tree.is_empty() {
                    if let Some(query_object) = self.object(id) {
                        if let Some(leaf_id) = tree.leaf_for_query(&query_object, object_space) {
                            let mut seeds = tree.get_object_ids_from_leaf(leaf_id);
                            self.thin_tree_seeds(
                                &mut seeds,
                                use_all_nodes_in_leaf,
                                self.property.edge_size_for_creation,
                            );
                            seeds.retain(|seed| seed.id < id as u32);
                            if !seeds.is_empty() {
                                return seeds;
                            }
                        }
                    }
                }
            }
        }

        let mut seeds = Vec::new();
        let max_id = id - 1;
        match self.property.seed_type {
            SeedType::FixedNodes => {
                for seed_id in 1..=self
                    .effective_seed_count(self.property.edge_size_for_creation)
                    .min(max_id)
                {
                    seeds.push(ObjectDistance {
                        id: seed_id as u32,
                        distance: 0.0,
                    });
                }
            }
            SeedType::FirstNode => seeds.push(ObjectDistance {
                id: 1,
                distance: 0.0,
            }),
            _ => {
                let requested = self
                    .effective_seed_count(self.property.edge_size_for_creation)
                    .min(max_id);
                if requested == max_id {
                    for seed_id in 1..=max_id as u32 {
                        seeds.push(ObjectDistance {
                            id: seed_id,
                            distance: 0.0,
                        });
                    }
                } else {
                    let mut rng = thread_rng();
                    while seeds.len() < requested {
                        let seed_id = rng.gen_range(1..=max_id as u32);
                        if !seeds.iter().any(|seed| seed.id == seed_id) {
                            seeds.push(ObjectDistance {
                                id: seed_id,
                                distance: 0.0,
                            });
                        }
                    }
                }
            }
        }
        seeds
    }

    fn search_for_nng_insertion(&self, id: usize) -> Vec<ObjectDistance> {
        if id <= 1 {
            return Vec::new();
        }
        let Some(query_object) = self.object(id) else {
            return Vec::new();
        };
        let mut sc = SearchContainer {
            object: &query_object,
            radius: f32::MAX,
            size: self.property.edge_size_for_creation,
            exploration_coefficient: self.property.insertion_radius_coefficient,
            edge_size: -1,
        };
        let mut seeds = self.get_insertion_seeds(id, true);
        if seeds.is_empty() {
            return Vec::new();
        }
        let os = self.object_space.as_ref().expect("object space");
        let mut result = self.graph.search(os, &mut sc, &mut seeds);
        if result.len() < self.property.edge_size_for_creation && result.len() < id - 1 {
            sc.edge_size = 0;
            let mut retry_seeds = self.get_insertion_seeds(id, true);
            result = self.graph.search(os, &mut sc, &mut retry_seeds);
        }
        result
    }

    fn search_for_knng_insertion(&self, id: usize) -> Vec<ObjectDistance> {
        if id <= 1 {
            return Vec::new();
        }
        let os = self.object_space.as_ref().expect("object space");
        let Some(query) = self.object(id) else {
            return Vec::new();
        };
        let mut results = BinaryHeap::new();
        for other_id in 1..id {
            let Some(dist) = os.compare_to_id(&query, other_id) else {
                continue;
            };
            results.push(crate::graph::MaxDistanceNode(ObjectDistance {
                id: other_id as u32,
                distance: dist,
            }));
            if results.len() > self.property.edge_size_for_creation {
                results.pop();
            }
        }
        let mut final_res = Vec::with_capacity(results.len());
        while let Some(node) = results.pop() {
            final_res.push(node.0);
        }
        final_res.reverse();
        final_res
    }

    fn build_index_sequential(&mut self, num_objects: usize, debug: u32) {
        let start = Instant::now();
        let report_every = progress_interval(num_objects);
        for id in 1..=num_objects {
            self.insert_indexed_object(id, self.search_neighbors_for_insertion(id));
            if debug >= 1 && (id == 1 || id == num_objects || id % report_every == 0) {
                eprintln!(
                    "munind: build_progress inserted={id}/{num_objects} elapsed_ms={:.3}",
                    start.elapsed().as_secs_f64() * 1000.0
                );
            }
        }
    }

    fn build_index_in_batches(&mut self, num_objects: usize, debug: u32) {
        let batch_size = self.property.batch_size_for_creation.max(1);
        let mut start_id = 1usize;
        let start = Instant::now();
        while start_id <= num_objects {
            let end_id = (start_id + batch_size - 1).min(num_objects);
            let ids: Vec<usize> = (start_id..=end_id).collect();
            let this: &Index = &*self;
            let mut batch_results: Vec<(usize, Vec<ObjectDistance>)> = ids
                .par_iter()
                .map(|&id| (id, this.search_neighbors_for_insertion(id)))
                .collect();
            batch_results.sort_by_key(|(id, _)| *id);
            self.enrich_batch_results(&mut batch_results);
            for (id, neighbors) in batch_results {
                self.insert_indexed_object(id, neighbors);
            }
            if debug >= 1 {
                eprintln!(
                    "munind: build_progress inserted={end_id}/{num_objects} elapsed_ms={:.3}",
                    start.elapsed().as_secs_f64() * 1000.0
                );
            }
            start_id = end_id + 1;
        }
    }

    fn search_neighbors_for_insertion(&self, id: usize) -> Vec<ObjectDistance> {
        match self.property.graph_type {
            GraphType::ANNG | GraphType::IANNG | GraphType::RANNG => {
                self.search_for_nng_insertion(id)
            }
            _ => self.search_for_knng_insertion(id),
        }
    }

    fn enrich_batch_results(&self, batch_results: &mut [(usize, Vec<ObjectDistance>)]) {
        if !matches!(
            self.property.graph_type,
            GraphType::ANNG
                | GraphType::IANNG
                | GraphType::ONNG
                | GraphType::RANNG
                | GraphType::RIANNG
        ) {
            return;
        }
        let Some(object_space) = &self.object_space else {
            return;
        };
        let max_edges = self.property.edge_size_for_creation;
        for idx in 0..batch_results.len() {
            let (previous, current_and_rest) = batch_results.split_at_mut(idx);
            let (id, neighbors) = &mut current_and_rest[0];
            for (prev_id, _) in previous.iter() {
                let distance = object_space
                    .compare_ids(*id, *prev_id)
                    .expect("batch object missing");
                neighbors.push(ObjectDistance {
                    id: *prev_id as u32,
                    distance,
                });
            }
            neighbors.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.id.cmp(&b.id))
            });
            neighbors.retain(|neighbor| neighbor.id != *id as u32);
            neighbors.dedup_by(|a, b| a.id == b.id);
            if neighbors.len() > max_edges {
                neighbors.truncate(max_edges);
            }
        }
    }

    fn insert_indexed_object(&mut self, id: usize, neighbors: Vec<ObjectDistance>) {
        let insert_into_tree = self.should_insert_into_tree(id, &neighbors);
        self.insert_graph_node(id, neighbors);
        if insert_into_tree {
            if let (Some(tree), Some(object_space)) = (&mut self.tree, &self.object_space) {
                tree.insert(id as u32, object_space);
            }
        }
    }

    fn should_insert_into_tree(&self, id: usize, neighbors: &[ObjectDistance]) -> bool {
        let Some(object_space) = &self.object_space else {
            return false;
        };
        let Some(nearest) = neighbors.first() else {
            return true;
        };
        if nearest.distance != 0.0 {
            return true;
        }
        if !object_space.is_normalized_distance() {
            return false;
        }
        let Some(object) = object_space.materialize_object(id) else {
            return true;
        };
        let Some(neighbor) = object_space.materialize_object(nearest.id as usize) else {
            return true;
        };
        object_space.compare_l1(&object, &neighbor) != 0.0
    }

    fn insert_graph_node(&mut self, id: usize, mut neighbors: Vec<ObjectDistance>) {
        neighbors.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        neighbors.retain(|neighbor| neighbor.id != id as u32);
        neighbors.dedup_by(|a, b| a.id == b.id);

        match self.property.graph_type {
            GraphType::KNNG => {
                self.graph.edges[id - 1] = neighbors;
            }
            GraphType::BKNNG => {
                let mut merged = self.graph.edges[id - 1].clone();
                merged.extend(neighbors.iter().copied());
                merged.sort_by(|a, b| {
                    a.distance
                        .partial_cmp(&b.distance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.id.cmp(&b.id))
                });
                merged.dedup_by(|a, b| a.id == b.id);
                self.graph.edges[id - 1] = merged;
                for neighbor in neighbors {
                    let _ = self.graph.add_edge(
                        (neighbor.id - 1) as usize,
                        ObjectDistance {
                            id: id as u32,
                            distance: neighbor.distance,
                        },
                        false,
                    );
                }
            }
            GraphType::ONNG => {
                for neighbor in neighbors.iter().take(self.property.incoming_edge) {
                    let _ = self.graph.add_edge(
                        (neighbor.id - 1) as usize,
                        ObjectDistance {
                            id: id as u32,
                            distance: neighbor.distance,
                        },
                        true,
                    );
                }
                let mut outgoing = neighbors;
                if outgoing.len() > self.property.outgoing_edge {
                    outgoing.truncate(self.property.outgoing_edge);
                }
                self.graph.edges[id - 1] = outgoing;
            }
            GraphType::ANNG | GraphType::RANNG => {
                if matches!(self.property.graph_type, GraphType::ANNG)
                    && !matches!(
                        self.property.identical_object_edge_type,
                        IdenticalObjectEdgeType::None
                    )
                {
                    self.filter_duplicates_in_results(&mut neighbors);
                    if self.try_insert_identical_object_node(id, &neighbors) {
                        return;
                    }
                }
                self.graph.edges[id - 1] = neighbors.clone();
                for neighbor in neighbors {
                    let _ = self.graph.add_edge(
                        (neighbor.id - 1) as usize,
                        ObjectDistance {
                            id: id as u32,
                            distance: neighbor.distance,
                        },
                        true,
                    );
                }
            }
            GraphType::IANNG | GraphType::RIANNG => {
                self.graph.edges[id - 1] = neighbors.clone();
                let reverse_limit = self
                    .property
                    .incoming_edge
                    .max(self.property.outgoing_edge)
                    .max(self.property.edge_size_for_creation)
                    .max(1);
                for neighbor in neighbors {
                    let _ = self.graph.add_edge_with_deletion(
                        (neighbor.id - 1) as usize,
                        ObjectDistance {
                            id: id as u32,
                            distance: neighbor.distance,
                        },
                        reverse_limit,
                        true,
                    );
                }
            }
            _ => {
                self.graph.edges[id - 1] = neighbors.clone();
                for neighbor in neighbors {
                    let _ = self.graph.add_edge(
                        (neighbor.id - 1) as usize,
                        ObjectDistance {
                            id: id as u32,
                            distance: neighbor.distance,
                        },
                        true,
                    );
                }
            }
        }
    }

    fn filter_duplicates_in_results(&self, results: &mut Vec<ObjectDistance>) {
        if results.len() <= 1 {
            return;
        }
        let Some(object_space) = &self.object_space else {
            return;
        };
        let mut filtered = Vec::with_capacity(results.len());
        let mut i = 0usize;
        while i < results.len() {
            let current_distance = results[i].distance;
            let mut j = i + 1;
            while j < results.len() && results[j].distance == current_distance {
                j += 1;
            }
            if j - i == 1 {
                filtered.push(results[i]);
            } else {
                let mut groups: Vec<(u32, usize)> = Vec::new();
                for candidate in &results[i..j] {
                    let Some(candidate_object) =
                        object_space.materialize_object(candidate.id as usize)
                    else {
                        continue;
                    };
                    let candidate_degree = self.graph.edges[(candidate.id - 1) as usize].len();
                    let mut found = false;
                    for group in groups.iter_mut() {
                        let group_object = object_space
                            .materialize_object(group.0 as usize)
                            .expect("group object missing");
                        if object_space.compare_l1(&group_object, &candidate_object) == 0.0 {
                            if candidate_degree > group.1 {
                                *group = (candidate.id, candidate_degree);
                            }
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        groups.push((candidate.id, candidate_degree));
                    }
                }
                for (group_id, _) in groups {
                    filtered.push(ObjectDistance {
                        id: group_id,
                        distance: current_distance,
                    });
                }
            }
            i = j;
        }
        *results = filtered;
    }

    fn try_insert_identical_object_node(
        &mut self,
        id: usize,
        neighbors: &[ObjectDistance],
    ) -> bool {
        if matches!(
            self.property.identical_object_edge_type,
            IdenticalObjectEdgeType::None
        ) || neighbors.is_empty()
            || neighbors[0].distance != 0.0
        {
            return false;
        }

        let object_space = match &self.object_space {
            Some(object_space) => object_space,
            None => return false,
        };
        let inserted = match object_space.materialize_object(id) {
            Some(object) => object,
            None => return false,
        };

        let mut max_id = None;
        let mut max_degree = 0usize;
        for neighbor in neighbors {
            if neighbor.distance != 0.0 {
                break;
            }
            let other_id = neighbor.id as usize;
            let Some(other) = object_space.materialize_object(other_id) else {
                continue;
            };
            if object_space.compare_l1(&inserted, &other) != 0.0 {
                break;
            }
            let degree = self.graph.edges[other_id - 1].len();
            if degree > max_degree || max_id.is_none() {
                max_degree = degree;
                max_id = Some(neighbor.id);
            }
        }

        let Some(max_id) = max_id else {
            return false;
        };

        self.graph.edges[id - 1] = if matches!(
            self.property.identical_object_edge_type,
            IdenticalObjectEdgeType::UndirectedEdge
        ) {
            vec![ObjectDistance {
                id: max_id,
                distance: 0.0,
            }]
        } else {
            Vec::new()
        };
        let _ = self.graph.add_edge(
            (max_id - 1) as usize,
            ObjectDistance {
                id: id as u32,
                distance: 0.0,
            },
            true,
        );
        true
    }

    pub fn all_objects(&self) -> Vec<Vec<f32>> {
        self.materialize_objects()
    }

    pub fn object_count(&self) -> usize {
        self.object_space
            .as_ref()
            .map(|object_space| object_space.repository.len().saturating_sub(1))
            .unwrap_or(self.objects.len())
    }

    fn object(&self, id: usize) -> Option<Vec<f32>> {
        self.object_space
            .as_ref()
            .and_then(|object_space| object_space.materialize_object(id))
            .or_else(|| self.objects.get(id.saturating_sub(1)).cloned())
    }

    fn materialize_objects(&self) -> Vec<Vec<f32>> {
        if let Some(object_space) = &self.object_space {
            object_space.repository.materialize()
        } else {
            self.objects.clone()
        }
    }

    fn serializable_snapshot(&self) -> SerializableIndex<'_> {
        SerializableIndex {
            property: &self.property,
            objects: self.materialize_objects(),
            graph: &self.graph,
            tree: &self.tree,
            path: &self.path,
        }
    }

    fn save_mmap_objects<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let file = File::create(path).map_err(|e| e.to_string())?;
        let mut writer = BufWriter::new(file);
        let objects = self.materialize_objects();
        let object_count = objects.len() as u64;
        writer.write_all(OBJECT_MAGIC).map_err(|e| e.to_string())?;
        writer
            .write_all(&object_count.to_le_bytes())
            .map_err(|e| e.to_string())?;
        writer
            .write_all(&(self.property.dimension as u64).to_le_bytes())
            .map_err(|e| e.to_string())?;
        let max_magnitude = self
            .object_space
            .as_ref()
            .map(|object_space| object_space.max_magnitude)
            .unwrap_or(-1.0);
        writer
            .write_all(&max_magnitude.to_le_bytes())
            .map_err(|e| e.to_string())?;
        writer
            .write_all(&0u32.to_le_bytes())
            .map_err(|e| e.to_string())?;
        writer
            .write_all(&0u32.to_le_bytes())
            .map_err(|e| e.to_string())?;
        debug_assert_eq!(OBJECT_HEADER_SIZE, 36);
        for object in objects {
            for value in object {
                writer
                    .write_all(&value.to_le_bytes())
                    .map_err(|e| e.to_string())?;
            }
        }
        writer.flush().map_err(|e| e.to_string())
    }

    fn save_mmap_graph<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let file = File::create(path).map_err(|e| e.to_string())?;
        let mut writer = BufWriter::new(file);
        let node_count = self.graph.edges.len() as u64;
        let mut offsets = Vec::with_capacity(self.graph.edges.len() + 1);
        offsets.push(0u64);
        let mut edge_total = 0u64;
        for edges in &self.graph.edges {
            edge_total += edges.len() as u64;
            offsets.push(edge_total);
        }
        writer.write_all(GRAPH_MAGIC).map_err(|e| e.to_string())?;
        writer
            .write_all(&node_count.to_le_bytes())
            .map_err(|e| e.to_string())?;
        writer
            .write_all(&edge_total.to_le_bytes())
            .map_err(|e| e.to_string())?;
        writer
            .write_all(&0u64.to_le_bytes())
            .map_err(|e| e.to_string())?;
        for offset in offsets {
            writer
                .write_all(&offset.to_le_bytes())
                .map_err(|e| e.to_string())?;
        }
        for edges in &self.graph.edges {
            for edge in edges {
                writer
                    .write_all(&edge.id.to_le_bytes())
                    .map_err(|e| e.to_string())?;
                writer
                    .write_all(&edge.distance.to_le_bytes())
                    .map_err(|e| e.to_string())?;
            }
        }
        writer.flush().map_err(|e| e.to_string())
    }
}

fn progress_interval(num_objects: usize) -> usize {
    if num_objects >= 1_000_000 {
        50_000
    } else if num_objects >= 100_000 {
        10_000
    } else if num_objects >= 10_000 {
        1_000
    } else {
        100
    }
}
