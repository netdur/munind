use munind_core::domain::{FilterExpression, MemoryId};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

const INDEXED_PATHS: &[&str] = &[
    "doc_id",
    "source",
    "type",
    "created_at",
    "tags",
    "session_id",
    "metadata.doc_id",
    "metadata.source",
    "metadata.type",
    "metadata.created_at",
    "metadata.tags",
    "metadata.session_id",
];

#[derive(Debug, Clone)]
struct DocEntry {
    keys: Vec<(String, String)>,
}

/// Candidate narrowing plan derived from payload indexes.
#[derive(Debug, Clone)]
pub struct PayloadFilterPlan {
    pub candidate_ids: HashSet<MemoryId>,
    /// True when every filter clause was resolved by indexes.
    pub fully_indexed: bool,
}

/// In-memory exact-match payload index for frequently filtered fields.
#[derive(Debug, Clone)]
pub struct PayloadIndex {
    postings: HashMap<String, HashMap<String, HashSet<MemoryId>>>,
    docs: HashMap<MemoryId, DocEntry>,
}

impl Default for PayloadIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl PayloadIndex {
    pub fn new() -> Self {
        Self {
            postings: HashMap::new(),
            docs: HashMap::new(),
        }
    }

    pub fn insert(&mut self, id: MemoryId, doc: &Value) {
        self.remove(id);

        let entry = Self::build_doc_entry(doc);
        for (path, value_key) in &entry.keys {
            self.postings
                .entry(path.clone())
                .or_default()
                .entry(value_key.clone())
                .or_default()
                .insert(id);
        }

        self.docs.insert(id, entry);
    }

    pub fn remove(&mut self, id: MemoryId) {
        let Some(entry) = self.docs.remove(&id) else {
            return;
        };

        for (path, value_key) in entry.keys {
            if let Some(path_map) = self.postings.get_mut(&path) {
                if let Some(ids) = path_map.get_mut(&value_key) {
                    ids.remove(&id);
                    if ids.is_empty() {
                        path_map.remove(&value_key);
                    }
                }

                if path_map.is_empty() {
                    self.postings.remove(&path);
                }
            }
        }
    }

    /// Builds a filter plan:
    /// - `None`: no indexed clauses were usable.
    /// - `Some(plan)` with `fully_indexed=true`: full filter resolved by index.
    /// - `Some(plan)` with `fully_indexed=false`: partial narrowing; caller must post-filter.
    pub fn plan_filter(&self, filter: &FilterExpression) -> Option<PayloadFilterPlan> {
        match filter {
            FilterExpression::Eq(path, expected) => self.eq_plan(path, expected),
            FilterExpression::And(parts) => {
                let mut combined: Option<HashSet<MemoryId>> = None;
                let mut fully_indexed = true;

                for part in parts {
                    if let Some(plan) = self.plan_filter(part) {
                        if let Some(current) = &mut combined {
                            intersect_in_place(current, &plan.candidate_ids);
                        } else {
                            combined = Some(plan.candidate_ids);
                        }
                        fully_indexed &= plan.fully_indexed;
                    } else {
                        fully_indexed = false;
                    }
                }

                combined.map(|candidate_ids| PayloadFilterPlan {
                    candidate_ids,
                    fully_indexed,
                })
            }
        }
    }

    fn eq_plan(&self, path: &str, expected: &Value) -> Option<PayloadFilterPlan> {
        if !is_indexed_path(path) {
            return None;
        }

        let value_key = encode_value(expected)?;
        let ids = self
            .postings
            .get(path)
            .and_then(|values| values.get(&value_key))
            .cloned()
            .unwrap_or_default();

        Some(PayloadFilterPlan {
            candidate_ids: ids,
            fully_indexed: true,
        })
    }

    fn build_doc_entry(doc: &Value) -> DocEntry {
        let mut keys = Vec::new();

        for path in INDEXED_PATHS {
            let Some(value) = json_path_get(doc, path) else {
                continue;
            };

            let Some(value_key) = encode_value(value) else {
                continue;
            };

            keys.push(((*path).to_string(), value_key));
        }

        DocEntry { keys }
    }
}

fn intersect_in_place(left: &mut HashSet<MemoryId>, right: &HashSet<MemoryId>) {
    if left.len() <= right.len() {
        left.retain(|id| right.contains(id));
        return;
    }

    let mut reduced = HashSet::with_capacity(right.len());
    for id in right {
        if left.contains(id) {
            reduced.insert(*id);
        }
    }
    *left = reduced;
}

fn encode_value(value: &Value) -> Option<String> {
    serde_json::to_string(value).ok()
}

fn is_indexed_path(path: &str) -> bool {
    INDEXED_PATHS.contains(&path)
}

fn json_path_get<'a>(doc: &'a Value, path: &str) -> Option<&'a Value> {
    let mut current = doc;
    for key in path.split('.') {
        current = current.get(key)?;
    }
    Some(current)
}

#[cfg(test)]
mod tests {
    use super::PayloadIndex;
    use munind_core::domain::{FilterExpression, MemoryId};
    use serde_json::json;
    use std::collections::HashSet;

    #[test]
    fn indexed_eq_plan_returns_expected_ids() {
        let mut idx = PayloadIndex::new();
        let id1 = MemoryId(1);
        let id2 = MemoryId(2);

        idx.insert(
            id1,
            &json!({"doc_id": "a", "source": "desk", "session_id": "s1"}),
        );
        idx.insert(
            id2,
            &json!({"doc_id": "b", "source": "mobile", "session_id": "s2"}),
        );

        let plan = idx
            .plan_filter(&FilterExpression::Eq("doc_id".to_string(), json!("b")))
            .expect("expected indexed plan");

        assert!(plan.fully_indexed);
        assert_eq!(plan.candidate_ids, HashSet::from([id2]));
    }

    #[test]
    fn and_plan_can_be_partial() {
        let mut idx = PayloadIndex::new();
        let id1 = MemoryId(1);
        let id2 = MemoryId(2);
        let id3 = MemoryId(3);

        idx.insert(id1, &json!({"source": "desk", "x": "keep"}));
        idx.insert(id2, &json!({"source": "desk", "x": "skip"}));
        idx.insert(id3, &json!({"source": "mobile", "x": "keep"}));

        let filter = FilterExpression::And(vec![
            FilterExpression::Eq("source".to_string(), json!("desk")),
            FilterExpression::Eq("x".to_string(), json!("keep")),
        ]);

        let plan = idx
            .plan_filter(&filter)
            .expect("expected partial indexed plan");
        assert!(!plan.fully_indexed);
        assert_eq!(plan.candidate_ids, HashSet::from([id1, id2]));
    }

    #[test]
    fn remove_updates_index() {
        let mut idx = PayloadIndex::new();
        let id = MemoryId(10);

        idx.insert(id, &json!({"doc_id": "abc", "source": "desk"}));
        assert!(
            idx.plan_filter(&FilterExpression::Eq("doc_id".to_string(), json!("abc")))
                .unwrap()
                .candidate_ids
                .contains(&id)
        );

        idx.remove(id);
        assert!(
            idx.plan_filter(&FilterExpression::Eq("doc_id".to_string(), json!("abc")))
                .unwrap()
                .candidate_ids
                .is_empty()
        );
    }
}
