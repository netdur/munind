use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;

use crate::graph::{NeighborhoodGraph, SearchContainer};
use crate::node::ObjectDistance;
use crate::object_space::{DistanceType, ObjectSpace, ObjectType};
use crate::tree::DvpTree;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NgtIndexType {
    None,
    GraphAndTree,
    Graph,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NgtDatabaseType {
    None,
    Memory,
    MemoryMappedFile,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NgtObjectAlignment {
    None,
    True,
    False,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NgtGraphType {
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
pub enum NgtSeedType {
    None,
    RandomNodes,
    FixedNodes,
    FirstNode,
    AllLeafNodes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NgtEpsilonType {
    None,
    ByQuery,
    ResultSize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NgtIdenticalObjectEdgeType {
    None,
    DirectedEdge,
    UndirectedEdge,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum NgtDistanceType {
    L1,
    L2,
    Cosine,
    Angle,
    DotProduct,
}

impl Into<DistanceType> for NgtDistanceType {
    fn into(self) -> DistanceType {
        match self {
            NgtDistanceType::L1 => DistanceType::L1,
            NgtDistanceType::L2 => DistanceType::L2,
            NgtDistanceType::Cosine => DistanceType::Cosine,
            NgtDistanceType::Angle => DistanceType::Angle,
            NgtDistanceType::DotProduct => DistanceType::DotProduct,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NgtProperty {
    pub dimension: usize,
    pub thread_pool_size: usize,
    pub distance_type: NgtDistanceType,
    pub object_type: ObjectType,
    pub index_type: NgtIndexType,
    pub database_type: NgtDatabaseType,
    pub object_alignment: NgtObjectAlignment,
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
    pub seed_type: NgtSeedType,
    pub truncation_thread_pool_size: usize,
    pub batch_size_for_creation: usize,
    pub graph_type: NgtGraphType,
    pub dynamic_edge_size_base: usize,
    pub dynamic_edge_size_rate: usize,
    pub build_time_limit: f32,
    pub outgoing_edge: usize,
    pub incoming_edge: usize,
    pub epsilon_type: NgtEpsilonType,
    pub identical_object_edge_type: NgtIdenticalObjectEdgeType,
}

impl NgtProperty {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            thread_pool_size: 32,
            distance_type: NgtDistanceType::L2,
            object_type: ObjectType::Float,
            index_type: NgtIndexType::GraphAndTree,
            database_type: NgtDatabaseType::Memory,
            object_alignment: NgtObjectAlignment::False,
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
            seed_type: NgtSeedType::None,
            truncation_thread_pool_size: 8,
            batch_size_for_creation: 200,
            graph_type: NgtGraphType::ANNG,
            dynamic_edge_size_base: 30,
            dynamic_edge_size_rate: 20,
            build_time_limit: 0.0,
            outgoing_edge: 10,
            incoming_edge: 80,
            epsilon_type: NgtEpsilonType::None,
            identical_object_edge_type: NgtIdenticalObjectEdgeType::None,
        }
    }

    pub fn set_distance_type(&mut self, dt: NgtDistanceType) {
        self.distance_type = dt;
    }
}

#[derive(Clone, Debug)]
pub struct NgtSearchOptions {
    pub k: usize,
    pub epsilon: f32,
    pub edge_size: Option<isize>,
}

struct NgtRandom {
    state: u32,
}

impl NgtRandom {
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
pub struct NgtIndex {
    pub property: NgtProperty,
    pub objects: Vec<Vec<f32>>,
    #[serde(skip)]
    pub object_space: Option<ObjectSpace>,
    pub graph: NeighborhoodGraph,
    pub tree: Option<DvpTree>,
    pub path: String,
}

#[derive(Serialize)]
struct SerializableNgtIndex<'a> {
    property: &'a NgtProperty,
    objects: Vec<Vec<f32>>,
    graph: &'a NeighborhoodGraph,
    tree: &'a Option<DvpTree>,
    path: &'a String,
}

impl NgtIndex {
    pub fn create_graph_and_tree<P: AsRef<Path>>(
        path: P,
        property: NgtProperty,
    ) -> Result<Self, String> {
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
        let mut index: NgtIndex = bincode::deserialize_from(reader).map_err(|e| e.to_string())?;

        // Reconstruct object_space correctly
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

    pub fn save_as_ngt<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
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

    pub fn open_ngt<P: AsRef<Path>>(path: P) -> Result<Self, String> {
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

    fn save_property_file<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
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
                NgtDatabaseType::None => "None",
                NgtDatabaseType::Memory => "Memory",
                NgtDatabaseType::MemoryMappedFile => "MemoryMappedFile",
            }
            .to_string(),
        ));
        lines.push(("Dimension", self.property.dimension.to_string()));
        lines.push((
            "DistanceType",
            match self.property.distance_type {
                NgtDistanceType::L1 => "L1",
                NgtDistanceType::L2 => "L2",
                NgtDistanceType::Cosine => "Cosine",
                NgtDistanceType::Angle => "Angle",
                NgtDistanceType::DotProduct => "DotProduct",
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
                NgtEpsilonType::None => "None",
                NgtEpsilonType::ByQuery => "ByQuery",
                NgtEpsilonType::ResultSize => "ResultSize",
            }
            .to_string(),
        ));
        lines.push((
            "GraphType",
            match self.property.graph_type {
                NgtGraphType::None => "None",
                NgtGraphType::ANNG => "ANNG",
                NgtGraphType::KNNG => "KNNG",
                NgtGraphType::BKNNG => "BKNNG",
                NgtGraphType::ONNG => "ONNG",
                NgtGraphType::IANNG => "IANNG",
                NgtGraphType::DNNG => "DNNG",
                NgtGraphType::RANNG => "RANNG",
                NgtGraphType::RIANNG => "RIANNG",
            }
            .to_string(),
        ));
        lines.push((
            "IdenticalObjectEdgeType",
            match self.property.identical_object_edge_type {
                NgtIdenticalObjectEdgeType::None => "None",
                NgtIdenticalObjectEdgeType::DirectedEdge => "DirectedEdge",
                NgtIdenticalObjectEdgeType::UndirectedEdge => "UndirectedEdge",
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
                NgtIndexType::None => "None",
                NgtIndexType::GraphAndTree => "GraphAndTree",
                NgtIndexType::Graph => "Graph",
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
                NgtObjectAlignment::None => "None",
                NgtObjectAlignment::True => "True",
                NgtObjectAlignment::False => "False",
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
                NgtSeedType::None => "None",
                NgtSeedType::RandomNodes => "RandomNodes",
                NgtSeedType::FixedNodes => "FixedNodes",
                NgtSeedType::FirstNode => "FirstNode",
                NgtSeedType::AllLeafNodes => "AllLeafNodes",
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

    fn load_property_file<P: AsRef<Path>>(path: P) -> Result<NgtProperty, String> {
        let file = File::open(path).map_err(|e| e.to_string())?;
        let reader = BufReader::new(file);
        let mut property = NgtProperty::new(0);
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
                "DistanceType" => {
                    property.distance_type = match value {
                        "L1" => NgtDistanceType::L1,
                        "L2" => NgtDistanceType::L2,
                        "Cosine" => NgtDistanceType::Cosine,
                        "Angle" => NgtDistanceType::Angle,
                        "DotProduct" => NgtDistanceType::DotProduct,
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
                        "ANNG" => NgtGraphType::ANNG,
                        "KNNG" => NgtGraphType::KNNG,
                        "BKNNG" => NgtGraphType::BKNNG,
                        "ONNG" => NgtGraphType::ONNG,
                        "IANNG" => NgtGraphType::IANNG,
                        "DNNG" => NgtGraphType::DNNG,
                        "RANNG" => NgtGraphType::RANNG,
                        "RIANNG" => NgtGraphType::RIANNG,
                        "None" => NgtGraphType::None,
                        other => return Err(format!("Unsupported GraphType: {other}")),
                    }
                }
                "IndexType" => {
                    property.index_type = match value {
                        "GraphAndTree" => NgtIndexType::GraphAndTree,
                        "Graph" => NgtIndexType::Graph,
                        "None" => NgtIndexType::None,
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
        let mut graph = NeighborhoodGraph::new();
        graph.edge_size_for_creation = self.property.edge_size_for_creation;
        graph.edge_size_for_search = self.property.edge_size_for_search;
        graph.insertion_exploration_coefficient = self.property.dynamic_edge_size_rate as f64;
        graph.dynamic_edge_size_base = self.property.dynamic_edge_size_base;
        graph.dynamic_edge_size_rate = self.property.dynamic_edge_size_rate;
        let num_objects = self.object_count();
        graph.edges = vec![Vec::new(); num_objects];

        self.graph = graph;
        self.tree = if matches!(self.property.index_type, NgtIndexType::GraphAndTree) {
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
                Ok(pool) => pool.install(|| self.build_index_in_batches(num_objects)),
                Err(_) => self.build_index_sequential(num_objects),
            }
        } else {
            self.build_index_sequential(num_objects);
        }
    }

    pub fn search(
        &self,
        query: &[f32],
        options: &NgtSearchOptions,
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
            let object = self
                .object(i)
                .ok_or_else(|| format!("Object {i} is missing"))?;
            let dist = os.compare(&prepared_query, object) as f32;
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
        if matches!(self.property.index_type, NgtIndexType::GraphAndTree) {
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
            NgtSeedType::FixedNodes => {
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
            NgtSeedType::FirstNode => {
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
            || matches!(self.property.seed_type, NgtSeedType::AllLeafNodes)
        {
            return;
        }

        match self.property.seed_type {
            NgtSeedType::None => {
                let seed_size = self.effective_seed_count(k).min(k.max(1));
                if seeds.len() > seed_size {
                    self.random_thin_seeds(seeds, seed_size, true);
                }
            }
            NgtSeedType::FixedNodes => {
                let seed_size = self.effective_seed_count(k);
                if seeds.len() > seed_size {
                    seeds.truncate(seed_size);
                }
            }
            NgtSeedType::RandomNodes => {
                let seed_size = self.effective_seed_count(k);
                if seeds.len() > seed_size {
                    self.random_thin_seeds(seeds, seed_size, false);
                }
            }
            NgtSeedType::FirstNode => seeds.truncate(1),
            NgtSeedType::AllLeafNodes => {}
        }
    }

    fn setup_seed_distances(&self, prepared_query: &[f32], seeds: &mut [ObjectDistance]) {
        let Some(object_space) = &self.object_space else {
            return;
        };
        for seed in seeds.iter_mut() {
            seed.distance = object_space
                .get_object(seed.id as usize)
                .map(|object| object_space.compare(prepared_query, object) as f32)
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
        let mut rng = NgtRandom::new(if deterministic {
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
        let mut candidates: Vec<u32> = (1..=repository_size as u32)
            .filter(|id| !seeds.iter().any(|seed| seed.id == *id))
            .collect();
        let mut rng = thread_rng();
        candidates.shuffle(&mut rng);
        for id in candidates.into_iter().take(requested - seeds.len()) {
            seeds.push(ObjectDistance { id, distance: 0.0 });
        }
    }

    fn get_insertion_seeds(&self, id: usize, use_all_nodes_in_leaf: bool) -> Vec<ObjectDistance> {
        if id <= 1 {
            return Vec::new();
        }
        if matches!(self.property.index_type, NgtIndexType::GraphAndTree) {
            if let (Some(tree), Some(object_space)) = (&self.tree, &self.object_space) {
                if !tree.is_empty() {
                    if let Some(query_object) = self.object(id) {
                        if let Some(leaf_id) = tree.leaf_for_query(query_object, object_space) {
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
            NgtSeedType::FixedNodes => {
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
            NgtSeedType::FirstNode => seeds.push(ObjectDistance {
                id: 1,
                distance: 0.0,
            }),
            _ => {
                let mut candidates: Vec<u32> = (1..=max_id as u32).collect();
                let mut rng = thread_rng();
                candidates.shuffle(&mut rng);
                for seed_id in candidates
                    .into_iter()
                    .take(self.effective_seed_count(self.property.edge_size_for_creation))
                {
                    seeds.push(ObjectDistance {
                        id: seed_id,
                        distance: 0.0,
                    });
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
            object: query_object,
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
            let Some(other) = self.object(other_id) else {
                continue;
            };
            let dist = os.compare(query, other) as f32;
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

    fn build_index_sequential(&mut self, num_objects: usize) {
        for id in 1..=num_objects {
            self.insert_indexed_object(id, self.search_neighbors_for_insertion(id));
        }
    }

    fn build_index_in_batches(&mut self, num_objects: usize) {
        let batch_size = self.property.batch_size_for_creation.max(1);
        let mut start_id = 1usize;
        while start_id <= num_objects {
            let end_id = (start_id + batch_size - 1).min(num_objects);
            let ids: Vec<usize> = (start_id..=end_id).collect();
            let this: &NgtIndex = &*self;
            let mut batch_results: Vec<(usize, Vec<ObjectDistance>)> = ids
                .par_iter()
                .map(|&id| (id, this.search_neighbors_for_insertion(id)))
                .collect();
            batch_results.sort_by_key(|(id, _)| *id);
            self.enrich_batch_results(&mut batch_results);
            for (id, neighbors) in batch_results {
                self.insert_indexed_object(id, neighbors);
            }
            start_id = end_id + 1;
        }
    }

    fn search_neighbors_for_insertion(&self, id: usize) -> Vec<ObjectDistance> {
        match self.property.graph_type {
            NgtGraphType::ANNG | NgtGraphType::IANNG | NgtGraphType::RANNG => {
                self.search_for_nng_insertion(id)
            }
            _ => self.search_for_knng_insertion(id),
        }
    }

    fn enrich_batch_results(&self, batch_results: &mut [(usize, Vec<ObjectDistance>)]) {
        if !matches!(
            self.property.graph_type,
            NgtGraphType::ANNG
                | NgtGraphType::IANNG
                | NgtGraphType::ONNG
                | NgtGraphType::RANNG
                | NgtGraphType::RIANNG
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
                let distance = object_space.compare(
                    self.object(*id).expect("current object missing"),
                    self.object(*prev_id).expect("previous object missing"),
                );
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
        let Some(object) = object_space.get_object(id) else {
            return true;
        };
        let Some(neighbor) = object_space.get_object(nearest.id as usize) else {
            return true;
        };
        object_space.compare_l1(object, neighbor) != 0.0
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
            NgtGraphType::KNNG => {
                self.graph.edges[id - 1] = neighbors;
            }
            NgtGraphType::BKNNG => {
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
            NgtGraphType::ONNG => {
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
            NgtGraphType::ANNG | NgtGraphType::RANNG => {
                if matches!(self.property.graph_type, NgtGraphType::ANNG)
                    && !matches!(
                        self.property.identical_object_edge_type,
                        NgtIdenticalObjectEdgeType::None
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
            NgtGraphType::IANNG | NgtGraphType::RIANNG => {
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
                    let Some(candidate_object) = object_space.get_object(candidate.id as usize)
                    else {
                        continue;
                    };
                    let candidate_degree = self.graph.edges[(candidate.id - 1) as usize].len();
                    let mut found = false;
                    for group in groups.iter_mut() {
                        let group_object = object_space
                            .get_object(group.0 as usize)
                            .expect("group object missing");
                        if object_space.compare_l1(group_object, candidate_object) == 0.0 {
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
            NgtIdenticalObjectEdgeType::None
        ) || neighbors.is_empty()
            || neighbors[0].distance != 0.0
        {
            return false;
        }

        let object_space = match &self.object_space {
            Some(object_space) => object_space,
            None => return false,
        };
        let inserted = match object_space.get_object(id) {
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
            let Some(other) = object_space.get_object(other_id) else {
                continue;
            };
            if object_space.compare_l1(inserted, other) != 0.0 {
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
            NgtIdenticalObjectEdgeType::UndirectedEdge
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

    fn object(&self, id: usize) -> Option<&[f32]> {
        self.object_space
            .as_ref()
            .and_then(|object_space| object_space.get_object(id))
            .or_else(|| self.objects.get(id.saturating_sub(1)).map(Vec::as_slice))
    }

    fn materialize_objects(&self) -> Vec<Vec<f32>> {
        if let Some(object_space) = &self.object_space {
            object_space
                .repository
                .objects
                .iter()
                .skip(1)
                .cloned()
                .collect()
        } else {
            self.objects.clone()
        }
    }

    fn serializable_snapshot(&self) -> SerializableNgtIndex<'_> {
        SerializableNgtIndex {
            property: &self.property,
            objects: self.materialize_objects(),
            graph: &self.graph,
            tree: &self.tree,
            path: &self.path,
        }
    }
}
