use crate::common::{Distance, ObjectID};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum NodeType {
    Internal = 0,
    Leaf = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct NodeID {
    id: u32,
}

impl NodeID {
    pub fn new() -> Self {
        Self { id: 0 }
    }

    pub fn from_id(id: u32) -> Self {
        let mut n = Self::new();
        n.set_id(id);
        n
    }

    pub fn get_type(&self) -> NodeType {
        if (self.id & 0x80000000) != 0 {
            NodeType::Leaf
        } else {
            NodeType::Internal
        }
    }

    pub fn get_id(&self) -> u32 {
        self.id & 0x7FFFFFFF
    }

    pub fn get(&self) -> u32 {
        self.id
    }

    pub fn set_id(&mut self, i: u32) {
        self.id = (self.id & 0x80000000) | (i & 0x7FFFFFFF);
    }

    pub fn set_type(&mut self, t: NodeType) {
        self.id = ((t as u32) << 31) | self.get_id();
    }

    pub fn set_raw(&mut self, i: u32) {
        self.id = i;
    }

    pub fn set_null(&mut self) {
        self.id = 0;
    }
}

// In standard NGT without Shared Memory Allocator,
// Node uses a pointer to a pivot Object. We'll simplify this for the initial port
// to just an index/ID if it refers to the persistent space, or keep it abstract.
// Actually, NGT Node has an Object* or off_t pivot. We'll use Option<usize> or similar
// representing the offset/ID in the object space. In C++, `Object *pivot` holds the actual vector or a pointer to it.
// We'll leave it out until Phase 2 when ObjectSpace is built, or use a bare pointer/index.

#[derive(Clone, Debug)]
pub struct NodeObject {
    pub id: ObjectID,
    // Note: PersistentObject pointer is omitted for memory safety.
    // We will access vectors through the ObjectSpace using the id.
    pub distance: Distance,
    pub leaf_distance: Distance,
    pub cluster_id: i32,
}

impl NodeObject {
    pub fn new(id: ObjectID, distance: Distance) -> Self {
        Self {
            id,
            distance,
            leaf_distance: 0.0,
            cluster_id: 0,
        }
    }
}

// Implement partial ordering based on distance for NodeObject
impl PartialEq for NodeObject {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl PartialOrd for NodeObject {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

// Basic Node trait or struct
#[derive(Clone, Debug)]
pub struct Node {
    pub id: NodeID,
    pub parent: NodeID,
    // pivot: Option<ObjectID>, // To be implemented with ObjectSpace
}

impl Node {
    pub fn new() -> Self {
        Self {
            id: NodeID::new(),
            parent: NodeID::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct InternalNode {
    pub base: Node,
    pub children: Vec<NodeID>,
    pub borders: Vec<Distance>,
}

impl InternalNode {
    pub fn new(children_size: usize) -> Self {
        let mut n = Self {
            base: Node::new(),
            children: vec![NodeID::new(); children_size],
            borders: vec![0.0; children_size.saturating_sub(1)],
        };
        n.base.id.set_type(NodeType::Internal);
        n
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObjectDistance {
    pub id: u32,
    pub distance: Distance,
}

#[derive(Clone, Debug)]
pub struct LeafNode {
    pub base: Node,
    pub object_ids: Vec<ObjectDistance>, // Using NGT_NODE_USE_VECTOR equivalent
}

impl LeafNode {
    pub fn new(n_of_objects: usize) -> Self {
        let mut n = Self {
            base: Node::new(),
            object_ids: Vec::with_capacity(n_of_objects),
        };
        n.base.id.set_type(NodeType::Leaf);
        n
    }
}
