use crate::primitive_comparator::PrimitiveComparator;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum DistanceType {
    None = -1,
    L1 = 0,
    L2 = 1,
    Hamming = 2,
    Angle = 3,
    Cosine = 4,
    NormalizedAngle = 5,
    NormalizedCosine = 6,
    Jaccard = 7,
    SparseJaccard = 8,
    NormalizedL2 = 9,
    InnerProduct = 10,
    DotProduct = 11,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ObjectType {
    None = 0,
    Uint8 = 1,
    Float = 2,
    Unset = 127,
}

// NGT reserves object ID 0, so the repository keeps a sentinel entry at index 0.
pub struct ObjectRepository {
    objects: Vec<Vec<f32>>,
}

impl ObjectRepository {
    pub fn new() -> Self {
        Self {
            objects: vec![Vec::new()],
        }
    }

    pub fn len(&self) -> usize {
        self.objects.len()
    }

    pub fn push_float(&mut self, obj: Vec<f32>) -> usize {
        let id = self.objects.len();
        self.objects.push(obj);
        id
    }

    pub fn get(&self, id: usize) -> Option<&[f32]> {
        let object = self.objects.get(id)?;
        if object.is_empty() {
            None
        } else {
            Some(object.as_slice())
        }
    }

    pub fn delete(&mut self, id: usize) {
        if id < self.objects.len() {
            self.objects[id].clear();
        }
    }

    pub fn materialize(&self) -> Vec<Vec<f32>> {
        self.objects.iter().skip(1).cloned().collect()
    }
}

pub struct ObjectSpace {
    pub dimension: usize,
    pub distance_type: DistanceType,
    pub object_type: ObjectType,
    pub repository: ObjectRepository,
    pub padded_byte_size: usize,
    pub normalization: bool,
    pub max_magnitude: f32,
}

impl ObjectSpace {
    pub fn new(dimension: usize, distance_type: DistanceType, object_type: ObjectType) -> Self {
        let padded_dimension = if dimension == 0 {
            0
        } else {
            ((dimension - 1) / 16 + 1) * 16
        };
        Self {
            dimension,
            distance_type,
            object_type,
            repository: ObjectRepository::new(),
            padded_byte_size: padded_dimension * 4,
            normalization: matches!(
                distance_type,
                DistanceType::Angle
                    | DistanceType::Cosine
                    | DistanceType::NormalizedAngle
                    | DistanceType::NormalizedCosine
                    | DistanceType::NormalizedL2
            ),
            max_magnitude: -1.0,
        }
    }

    pub fn compare_l2(&self, a: &[f32], b: &[f32]) -> f32 {
        PrimitiveComparator::compare_l2_f32(a, b)
    }

    pub fn compare_l1(&self, a: &[f32], b: &[f32]) -> f32 {
        PrimitiveComparator::compare_l1_f32(a, b)
    }

    pub fn compare_cosine(&self, a: &[f32], b: &[f32]) -> f32 {
        PrimitiveComparator::compare_cosine_f32(a, b)
    }

    pub fn compare_normalized_l2(&self, a: &[f32], b: &[f32]) -> f32 {
        PrimitiveComparator::compare_normalized_l2_f32(a, b)
    }

    pub fn is_normalized_distance(&self) -> bool {
        self.normalization
    }

    pub fn normalize(object: &mut [f32]) -> Result<(), String> {
        let sum = object.iter().map(|v| *v * *v).sum::<f32>();
        if sum == 0.0 {
            return Err(
                "ObjectSpace::normalize: zero vector is invalid for normalized distances"
                    .to_string(),
            );
        }
        let norm = sum.sqrt();
        for value in object.iter_mut() {
            *value /= norm;
        }
        Ok(())
    }

    pub fn prepare_for_insert(&self, obj: &[f32]) -> Result<Vec<f32>, String> {
        if obj.len() != self.dimension {
            return Err(format!(
                "Invalid dimensionality. Expected {}, got {}",
                self.dimension,
                obj.len()
            ));
        }
        let mut stored = obj.to_vec();
        if self.is_normalized_distance() {
            Self::normalize(&mut stored)?;
        }
        Ok(stored)
    }

    pub fn prepare_query(&self, obj: &[f32]) -> Result<Vec<f32>, String> {
        self.prepare_for_insert(obj)
    }

    pub fn insert_prepared(&mut self, obj: Vec<f32>) -> Result<usize, String> {
        if obj.len() != self.dimension {
            return Err(format!(
                "Invalid dimensionality. Expected {}, got {}",
                self.dimension,
                obj.len()
            ));
        }
        if matches!(self.distance_type, DistanceType::DotProduct) {
            let magnitude = PrimitiveComparator::compare_dot_product_f32(&obj, &obj);
            if magnitude > self.max_magnitude {
                self.max_magnitude = magnitude;
            }
        }
        Ok(self.repository.push_float(obj))
    }

    pub fn compare(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.distance_type {
            DistanceType::L2 => self.compare_l2(a, b),
            DistanceType::L1 => self.compare_l1(a, b),
            DistanceType::Cosine => {
                if self.normalization {
                    PrimitiveComparator::compare_normalized_cosine_similarity_f32(a, b)
                } else {
                    1.0 - self.compare_cosine(a, b)
                }
            }
            DistanceType::NormalizedCosine => {
                PrimitiveComparator::compare_normalized_cosine_similarity_f32(a, b)
            }
            DistanceType::Angle => {
                if self.normalization {
                    PrimitiveComparator::compare_normalized_angle_distance_f32(a, b)
                } else {
                    PrimitiveComparator::compare_angle_distance_f32(a, b)
                }
            }
            DistanceType::NormalizedAngle => {
                PrimitiveComparator::compare_normalized_angle_distance_f32(a, b)
            }
            DistanceType::NormalizedL2 => self.compare_normalized_l2(a, b),
            DistanceType::InnerProduct => -PrimitiveComparator::compare_dot_product_f32(a, b),
            DistanceType::DotProduct => {
                let magnitude = if self.max_magnitude > 0.0 {
                    self.max_magnitude
                } else {
                    PrimitiveComparator::compare_dot_product_f32(b, b)
                };
                magnitude - PrimitiveComparator::compare_dot_product_f32(a, b)
            }
            _ => self.compare_l2(a, b),
        }
    }

    pub fn compare_to_id(&self, query: &[f32], id: usize) -> Option<f32> {
        self.get_object(id)
            .map(|object| self.compare(query, object))
    }

    pub fn compare_ids(&self, left: usize, right: usize) -> Option<f32> {
        let left_object = self.get_object(left)?;
        let right_object = self.get_object(right)?;
        Some(self.compare(left_object, right_object))
    }

    pub fn materialize_object(&self, id: usize) -> Option<Vec<f32>> {
        self.get_object(id).map(|object| object.to_vec())
    }

    pub fn insert(&mut self, obj: &[f32]) -> Result<usize, String> {
        let prepared = self.prepare_for_insert(obj)?;
        self.insert_prepared(prepared)
    }

    pub fn get_object(&self, id: usize) -> Option<&[f32]> {
        self.repository.get(id)
    }
}
