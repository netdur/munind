use munind_core::config::{DistanceMetric, EngineConfig};
use munind_core::domain::MemoryId;
use munind_core::error::{MunindError, Result};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::exact::{ExactSearcher, ScoredHit};
use crate::graph::GraphIndex;
use crate::lookup::VectorLookup;
use crate::search::AnnSearcher;

/// Orchestrates the graph and vector storage together.
pub struct IndexEngine {
    pub graph: GraphIndex,
    vectors: HashMap<MemoryId, Vec<f32>>,
    metric: DistanceMetric,
    ef_construction: usize,
    ef_search: usize,
}

const SNAPSHOT_MAGIC: &[u8; 8] = b"MNDIDX01";
const SNAPSHOT_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct IdDigest {
    count: u64,
    xor: u64,
    sum: u64,
}

fn mix_id(id: u64) -> u64 {
    let mut x = id.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn digest_ids_slice(ids: &[MemoryId]) -> IdDigest {
    let mut digest = IdDigest {
        count: 0,
        xor: 0,
        sum: 0,
    };
    for id in ids {
        let mixed = mix_id(id.0);
        digest.count = digest.count.saturating_add(1);
        digest.xor ^= mixed;
        digest.sum = digest.sum.wrapping_add(mixed);
    }
    digest
}

fn digest_ids_refs<'a, I>(ids: I) -> IdDigest
where
    I: IntoIterator<Item = &'a MemoryId>,
{
    let mut digest = IdDigest {
        count: 0,
        xor: 0,
        sum: 0,
    };
    for id in ids {
        let mixed = mix_id(id.0);
        digest.count = digest.count.saturating_add(1);
        digest.xor ^= mixed;
        digest.sum = digest.sum.wrapping_add(mixed);
    }
    digest
}

fn metric_to_tag(metric: &DistanceMetric) -> u8 {
    match metric {
        DistanceMetric::Cosine => 1,
        DistanceMetric::L2 => 2,
        DistanceMetric::InnerProduct => 3,
    }
}

fn metric_from_tag(tag: u8) -> Result<DistanceMetric> {
    match tag {
        1 => Ok(DistanceMetric::Cosine),
        2 => Ok(DistanceMetric::L2),
        3 => Ok(DistanceMetric::InnerProduct),
        _ => Err(MunindError::Corruption(format!(
            "unknown index snapshot metric tag: {}",
            tag
        ))),
    }
}

fn write_u8<W: Write>(w: &mut W, value: u8) -> Result<()> {
    w.write_all(&[value]).map_err(MunindError::Io)
}

fn write_u32<W: Write>(w: &mut W, value: u32) -> Result<()> {
    w.write_all(&value.to_le_bytes()).map_err(MunindError::Io)
}

fn write_u64<W: Write>(w: &mut W, value: u64) -> Result<()> {
    w.write_all(&value.to_le_bytes()).map_err(MunindError::Io)
}

fn write_f32<W: Write>(w: &mut W, value: f32) -> Result<()> {
    w.write_all(&value.to_le_bytes()).map_err(MunindError::Io)
}

fn read_exact<R: Read>(r: &mut R, buf: &mut [u8]) -> Result<()> {
    r.read_exact(buf).map_err(MunindError::Io)
}

fn read_u8<R: Read>(r: &mut R) -> Result<u8> {
    let mut buf = [0u8; 1];
    read_exact(r, &mut buf)?;
    Ok(buf[0])
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    read_exact(r, &mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    read_exact(r, &mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_f32<R: Read>(r: &mut R) -> Result<f32> {
    let mut buf = [0u8; 4];
    read_exact(r, &mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

impl IndexEngine {
    pub fn new(config: &EngineConfig) -> Self {
        Self {
            graph: GraphIndex::new(&config.index),
            vectors: HashMap::new(),
            metric: config.index.metric.clone(),
            ef_construction: config.index.ef_construction,
            ef_search: config.index.ef_search,
        }
    }

    pub fn insert(&mut self, id: MemoryId, vector: Vec<f32>) {
        // Replace semantics for duplicate IDs keeps graph/vector storage consistent.
        if self.vectors.insert(id, vector).is_some() {
            self.graph.remove_node(id);
        }

        let (graph, vectors) = (&mut self.graph, &self.vectors);
        let Some(vec_ref) = vectors.get_vector(id) else {
            return;
        };

        graph.insert(
            id,
            vec_ref,
            self.metric.clone(),
            self.ef_construction,
            vectors,
        );
    }

    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<ScoredHit> {
        self.search_with_ef(query, top_k, self.ef_search)
    }

    pub fn search_with_ef(&self, query: &[f32], top_k: usize, ef_search: usize) -> Vec<ScoredHit> {
        let best_seed = self.graph.entry_point;
        let searcher = AnnSearcher::new(&self.graph, query, &self.metric, &self.vectors);
        searcher.search(best_seed, top_k, ef_search)
    }

    /// Exact scoring restricted to a candidate ID set; useful for payload-indexed filters.
    pub fn exact_search_filtered(
        &self,
        query: &[f32],
        allowed_ids: &HashSet<MemoryId>,
        top_k: usize,
    ) -> Vec<ScoredHit> {
        if top_k == 0 || allowed_ids.is_empty() {
            return Vec::new();
        }

        let mut searcher = ExactSearcher::new(query, self.metric.clone(), top_k);
        for id in allowed_ids {
            if let Some(vector) = self.vectors.get_vector(*id) {
                searcher.push(*id, vector);
            }
        }

        searcher.take_results()
    }

    pub fn default_ef_search(&self) -> usize {
        self.ef_search
    }

    pub fn vector_count(&self) -> usize {
        self.vectors.len()
    }

    pub fn delete(&mut self, id: MemoryId) {
        self.vectors.remove(&id);
        self.graph.remove_node(id);
    }

    pub fn save_snapshot<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent).map_err(MunindError::Io)?;
        }

        let file = File::create(path).map_err(MunindError::Io)?;
        let mut writer = BufWriter::new(file);

        let vector_dim = self.vectors.values().next().map(|v| v.len()).unwrap_or(0);
        for vector in self.vectors.values() {
            if vector.len() != vector_dim {
                return Err(MunindError::Corruption(
                    "index snapshot save failed: inconsistent vector dimensions".to_string(),
                ));
            }
        }

        let digest = digest_ids_refs(self.vectors.keys());

        writer.write_all(SNAPSHOT_MAGIC).map_err(MunindError::Io)?;
        write_u32(&mut writer, SNAPSHOT_VERSION)?;
        write_u8(&mut writer, metric_to_tag(&self.metric))?;
        write_u64(&mut writer, self.ef_construction as u64)?;
        write_u64(&mut writer, self.ef_search as u64)?;
        write_u32(&mut writer, vector_dim as u32)?;
        write_u64(&mut writer, digest.count)?;
        write_u64(&mut writer, digest.xor)?;
        write_u64(&mut writer, digest.sum)?;

        write_u64(&mut writer, self.vectors.len() as u64)?;
        for (id, vector) in &self.vectors {
            write_u64(&mut writer, id.0)?;
            for value in vector {
                write_f32(&mut writer, *value)?;
            }
        }

        write_u8(&mut writer, self.graph.max_layer)?;
        match self.graph.entry_point {
            Some(id) => {
                write_u8(&mut writer, 1)?;
                write_u64(&mut writer, id.0)?;
            }
            None => {
                write_u8(&mut writer, 0)?;
            }
        }
        write_u64(&mut writer, self.graph.m as u64)?;
        write_u64(&mut writer, self.graph.m0 as u64)?;
        write_f32(&mut writer, self.graph.ml)?;

        write_u32(&mut writer, self.graph.layers.len() as u32)?;
        for (layer, nodes) in &self.graph.layers {
            write_u8(&mut writer, *layer)?;
            write_u64(&mut writer, nodes.len() as u64)?;
            for (node_id, neighbors) in nodes {
                write_u64(&mut writer, node_id.0)?;
                write_u32(&mut writer, neighbors.len() as u32)?;
                for neighbor in neighbors {
                    write_u64(&mut writer, neighbor.0)?;
                }
            }
        }

        writer.flush().map_err(MunindError::Io)?;
        Ok(())
    }

    pub fn load_snapshot<P: AsRef<Path>>(
        path: P,
        expected_ids: &[MemoryId],
        config: &EngineConfig,
    ) -> Result<Option<Self>> {
        if !path.as_ref().exists() {
            return Ok(None);
        }

        let file = File::open(path).map_err(MunindError::Io)?;
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 8];
        read_exact(&mut reader, &mut magic)?;
        if &magic != SNAPSHOT_MAGIC {
            return Ok(None);
        }

        let version = read_u32(&mut reader)?;
        if version != SNAPSHOT_VERSION {
            return Ok(None);
        }

        let snapshot_metric = metric_from_tag(read_u8(&mut reader)?)?;
        if snapshot_metric != config.index.metric {
            return Ok(None);
        }

        let ef_construction = read_u64(&mut reader)? as usize;
        let ef_search = read_u64(&mut reader)? as usize;
        let vector_dim = read_u32(&mut reader)? as usize;

        let expected_digest = digest_ids_slice(expected_ids);
        let snapshot_digest = IdDigest {
            count: read_u64(&mut reader)?,
            xor: read_u64(&mut reader)?,
            sum: read_u64(&mut reader)?,
        };
        if snapshot_digest != expected_digest {
            return Ok(None);
        }

        let vector_count = read_u64(&mut reader)? as usize;
        if vector_count != snapshot_digest.count as usize {
            return Ok(None);
        }

        let mut vectors = HashMap::with_capacity(vector_count);
        for _ in 0..vector_count {
            let id = MemoryId(read_u64(&mut reader)?);
            let mut vector = Vec::with_capacity(vector_dim);
            for _ in 0..vector_dim {
                vector.push(read_f32(&mut reader)?);
            }
            vectors.insert(id, vector);
        }

        let max_layer = read_u8(&mut reader)?;
        let entry_point = if read_u8(&mut reader)? == 1 {
            Some(MemoryId(read_u64(&mut reader)?))
        } else {
            None
        };

        let m = read_u64(&mut reader)? as usize;
        let m0 = read_u64(&mut reader)? as usize;
        let ml = read_f32(&mut reader)?;

        if m != config.index.m
            || m0 != config.index.m0
            || (ml - config.index.ml).abs() > f32::EPSILON
            || ef_construction != config.index.ef_construction
            || ef_search != config.index.ef_search
        {
            return Ok(None);
        }

        let layer_count = read_u32(&mut reader)? as usize;
        let mut layers = HashMap::with_capacity(layer_count);
        for _ in 0..layer_count {
            let layer_id = read_u8(&mut reader)?;
            let node_count = read_u64(&mut reader)? as usize;
            let mut nodes = HashMap::with_capacity(node_count);
            for _ in 0..node_count {
                let node_id = MemoryId(read_u64(&mut reader)?);
                let neighbor_count = read_u32(&mut reader)? as usize;
                let mut neighbors = Vec::with_capacity(neighbor_count);
                for _ in 0..neighbor_count {
                    neighbors.push(MemoryId(read_u64(&mut reader)?));
                }
                nodes.insert(node_id, neighbors);
            }
            layers.insert(layer_id, nodes);
        }

        Ok(Some(Self {
            graph: GraphIndex {
                layers,
                max_layer,
                entry_point,
                m,
                m0,
                ml,
            },
            vectors,
            metric: snapshot_metric,
            ef_construction,
            ef_search,
        }))
    }
}
