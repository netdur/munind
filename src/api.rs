/// Clean public API for munind.

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::common::ObjectDistance;
use crate::index::{
    Index as InnerIndex, IndexProperty, IndexDistanceType,
};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Distance metric.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Distance {
    L1,
    L2,
    Cosine,
    InnerProduct,
    Angle,
    NormalizedAngle,
    NormalizedL2,
    Hamming,
    Jaccard,
    Poincare,
    Lorentz,
}

impl Distance {
    fn to_inner(self) -> IndexDistanceType {
        match self {
            Distance::L1 => IndexDistanceType::L1,
            Distance::L2 => IndexDistanceType::L2,
            Distance::Cosine => IndexDistanceType::Cosine,
            Distance::InnerProduct => IndexDistanceType::InnerProduct,
            Distance::Angle => IndexDistanceType::Angle,
            Distance::NormalizedAngle => IndexDistanceType::NormalizedAngle,
            Distance::NormalizedL2 => IndexDistanceType::NormalizedL2,
            Distance::Hamming => IndexDistanceType::Hamming,
            Distance::Jaccard => IndexDistanceType::Jaccard,
            Distance::Poincare => IndexDistanceType::Poincare,
            Distance::Lorentz => IndexDistanceType::Lorentz,
        }
    }
}

/// Search result: object ID + distance.
#[derive(Clone, Copy, Debug)]
pub struct SearchResult {
    pub id: u32,
    pub distance: f32,
}

impl From<ObjectDistance> for SearchResult {
    fn from(od: ObjectDistance) -> Self {
        SearchResult {
            id: od.id,
            distance: od.distance,
        }
    }
}

/// Index configuration.
#[derive(Clone, Debug)]
pub struct IndexConfig {
    pub dimension: usize,
    pub distance: Distance,
    pub edge_size_for_creation: i32,
    pub edge_size_for_search: i32,
    pub truncation_threshold: usize,
}

impl IndexConfig {
    pub fn new(dimension: usize, distance: Distance) -> Self {
        IndexConfig {
            dimension,
            distance,
            edge_size_for_creation: 10,
            edge_size_for_search: 40,
            truncation_threshold: 50,
        }
    }
}

// ---------------------------------------------------------------------------
// Index
// ---------------------------------------------------------------------------

/// A nearest-neighbor search index.
///
/// # Example
/// ```no_run
/// use munind::api::{Index, IndexConfig, Distance};
///
/// let config = IndexConfig::new(4, Distance::L2);
/// let mut index = Index::create(config).unwrap();
/// index.insert(&[1.0, 2.0, 3.0, 4.0]).unwrap();
/// index.insert(&[5.0, 6.0, 7.0, 8.0]).unwrap();
/// index.build().unwrap();
/// let results = index.search(&[1.1, 2.1, 3.1, 4.1], 1).unwrap();
/// assert_eq!(results[0].id, 1);
/// ```
pub struct Index {
    inner: InnerIndex,
}

impl Index {
    /// Create a new empty index.
    pub fn create(config: IndexConfig) -> Result<Self> {
        if config.dimension == 0 {
            return Err(Error::ZeroDimension);
        }
        let mut prop = IndexProperty::new(config.dimension);
        prop.set_distance_type(config.distance.to_inner());
        prop.edge_size_for_creation = config.edge_size_for_creation;
        prop.edge_size_for_search = config.edge_size_for_search;
        prop.truncation_threshold = config.truncation_threshold;
        let inner = InnerIndex::create("", prop)?;
        Ok(Index { inner })
    }

    /// Open an existing index from a directory.
    pub fn open(path: &str) -> Result<Self> {
        let inner = InnerIndex::open_directory(path)?;
        Ok(Index { inner })
    }

    /// Save the index to a directory.
    pub fn save(&self, path: &str) -> Result<()> {
        self.inner.save_as_directory(path)?;
        Ok(())
    }

    // -- Mutation --

    /// Insert a vector. Returns the assigned 1-based ID.
    pub fn insert(&mut self, vector: &[f32]) -> Result<u32> {
        let id = self.inner.insert(vector)?;
        Ok(id)
    }

    /// Insert multiple vectors. Returns their IDs.
    pub fn insert_batch(&mut self, vectors: &[Vec<f32>]) -> Result<Vec<u32>> {
        let mut ids = Vec::with_capacity(vectors.len());
        for v in vectors {
            ids.push(self.inner.insert(v)?);
        }
        Ok(ids)
    }

    /// Build the graph index for all inserted objects.
    pub fn build(&mut self) -> Result<()> {
        self.inner.build();
        Ok(())
    }

    /// Remove an object by ID.
    pub fn remove(&mut self, id: u32) -> Result<()> {
        self.inner.delete_batch(&[id])?;
        Ok(())
    }

    // -- Search --

    /// Search for the `k` nearest neighbors.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let opts = crate::common::SearchOptions {
            k,
            epsilon: 0.1,
            edge_size: None,
        };
        let results = self.inner.search(query, &opts)?;
        Ok(results.into_iter().map(SearchResult::from).collect())
    }

    /// Search with custom parameters.
    pub fn search_with(
        &self,
        query: &[f32],
        k: usize,
        epsilon: f32,
        edge_size: Option<usize>,
    ) -> Result<Vec<SearchResult>> {
        let opts = crate::common::SearchOptions { k, epsilon, edge_size };
        let results = self.inner.search(query, &opts)?;
        Ok(results.into_iter().map(SearchResult::from).collect())
    }

    /// Search multiple queries in parallel.
    pub fn search_batch(
        &self,
        queries: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<Vec<SearchResult>>> {
        let results: Vec<Result<Vec<SearchResult>>> = queries
            .par_iter()
            .map(|q| self.search(q, k))
            .collect();
        results.into_iter().collect()
    }

    /// Brute-force linear search (exact, no graph).
    pub fn linear_search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let results = self.inner.linear_search(query, k)?;
        Ok(results.into_iter().map(SearchResult::from).collect())
    }

    // -- Info --

    /// Number of objects in the index.
    pub fn len(&self) -> usize {
        self.inner.object_count()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.object_count() == 0
    }

    /// Vector dimension.
    pub fn dimension(&self) -> usize {
        self.inner.property.dimension
    }

    /// Retrieve a stored vector by ID.
    pub fn get(&self, id: u32) -> Result<Vec<f32>> {
        let os = self.inner.object_space.as_ref()
            .ok_or(Error::EmptyIndex)?;
        let slice = os.get_object(id)?;
        Ok(slice.to_vec())
    }
}
