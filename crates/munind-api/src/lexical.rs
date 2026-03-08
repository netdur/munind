use munind_core::domain::MemoryId;
use serde_json::Value;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

const FIELD_TEXT: usize = 0;
const FIELD_TITLE: usize = 1;
const FIELD_TAGS: usize = 2;
const FIELD_COUNT: usize = 3;

type FieldTfs = [u16; FIELD_COUNT];
type FieldLens = [u32; FIELD_COUNT];

#[derive(Debug, Clone)]
struct DocEntry {
    terms: HashMap<String, FieldTfs>,
    lengths: FieldLens,
}

/// In-memory BM25F lexical index over text/title/tags fields.
#[derive(Debug, Clone)]
pub struct LexicalIndex {
    postings: HashMap<String, HashMap<MemoryId, FieldTfs>>,
    docs: HashMap<MemoryId, DocEntry>,
    total_lengths: [u64; FIELD_COUNT],
    field_weights: [f32; FIELD_COUNT],
    field_b: [f32; FIELD_COUNT],
    k1: f32,
}

impl Default for LexicalIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl LexicalIndex {
    pub fn new() -> Self {
        Self {
            postings: HashMap::new(),
            docs: HashMap::new(),
            total_lengths: [0; FIELD_COUNT],
            field_weights: [1.0, 1.6, 1.3],
            field_b: [0.75, 0.75, 0.75],
            k1: 1.2,
        }
    }

    pub fn insert(&mut self, id: MemoryId, doc: &Value) {
        self.remove(id);

        let entry = Self::build_doc_entry(doc);
        for i in 0..FIELD_COUNT {
            self.total_lengths[i] += u64::from(entry.lengths[i]);
        }

        for (term, tfs) in &entry.terms {
            self.postings
                .entry(term.clone())
                .or_default()
                .insert(id, *tfs);
        }

        self.docs.insert(id, entry);
    }

    pub fn remove(&mut self, id: MemoryId) {
        let Some(entry) = self.docs.remove(&id) else {
            return;
        };

        for i in 0..FIELD_COUNT {
            self.total_lengths[i] =
                self.total_lengths[i].saturating_sub(u64::from(entry.lengths[i]));
        }

        for term in entry.terms.keys() {
            if let Some(postings) = self.postings.get_mut(term) {
                postings.remove(&id);
                if postings.is_empty() {
                    self.postings.remove(term);
                }
            }
        }
    }

    pub fn search(&self, query: &str, top_k: usize) -> Vec<(MemoryId, f32)> {
        self.search_internal(query, top_k, None)
    }

    pub fn search_filtered(
        &self,
        query: &str,
        top_k: usize,
        allowed_ids: &HashSet<MemoryId>,
    ) -> Vec<(MemoryId, f32)> {
        if allowed_ids.is_empty() {
            return Vec::new();
        }
        self.search_internal(query, top_k, Some(allowed_ids))
    }

    fn search_internal(
        &self,
        query: &str,
        top_k: usize,
        allowed_ids: Option<&HashSet<MemoryId>>,
    ) -> Vec<(MemoryId, f32)> {
        if top_k == 0 || self.docs.is_empty() {
            return Vec::new();
        }

        let query_terms = term_freqs(&tokenize(query));
        if query_terms.is_empty() {
            return Vec::new();
        }

        let doc_count = self.docs.len() as f32;
        let mut scores: HashMap<MemoryId, f32> = HashMap::new();

        for (term, query_tf) in query_terms {
            let Some(postings) = self.postings.get(&term) else {
                continue;
            };

            let df = postings.len() as f32;
            let idf = (((doc_count - df + 0.5) / (df + 0.5)) + 1.0).ln();

            for (doc_id, field_tfs) in postings {
                if let Some(ids) = allowed_ids
                    && !ids.contains(doc_id)
                {
                    continue;
                }

                let Some(doc_entry) = self.docs.get(doc_id) else {
                    continue;
                };

                let tf = self.bm25f_tf(field_tfs, &doc_entry.lengths);
                if tf <= 0.0 {
                    continue;
                }

                let term_score = idf * ((tf * (self.k1 + 1.0)) / (tf + self.k1));
                *scores.entry(*doc_id).or_insert(0.0) += term_score * query_tf as f32;
            }
        }

        let mut ranked: Vec<(MemoryId, f32)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        ranked.truncate(top_k);
        ranked
    }

    fn bm25f_tf(&self, field_tfs: &FieldTfs, field_lens: &FieldLens) -> f32 {
        let mut sum = 0.0_f32;
        let docs = self.docs.len().max(1) as f32;

        for f in 0..FIELD_COUNT {
            let tf = field_tfs[f] as f32;
            if tf <= 0.0 {
                continue;
            }

            let avg_len = (self.total_lengths[f] as f32 / docs).max(1e-6);
            let len = field_lens[f] as f32;
            let norm = (1.0 - self.field_b[f]) + self.field_b[f] * (len / avg_len);

            sum += self.field_weights[f] * (tf / norm.max(1e-6));
        }

        sum
    }

    fn build_doc_entry(doc: &Value) -> DocEntry {
        let mut terms: HashMap<String, FieldTfs> = HashMap::new();
        let mut lengths: FieldLens = [0; FIELD_COUNT];

        for token in tokenize(&extract_text_field(doc)) {
            lengths[FIELD_TEXT] = lengths[FIELD_TEXT].saturating_add(1);
            let entry = terms.entry(token).or_insert([0; FIELD_COUNT]);
            entry[FIELD_TEXT] = entry[FIELD_TEXT].saturating_add(1);
        }

        for token in tokenize(&extract_title_field(doc)) {
            lengths[FIELD_TITLE] = lengths[FIELD_TITLE].saturating_add(1);
            let entry = terms.entry(token).or_insert([0; FIELD_COUNT]);
            entry[FIELD_TITLE] = entry[FIELD_TITLE].saturating_add(1);
        }

        for token in tokenize(&extract_tags_field(doc)) {
            lengths[FIELD_TAGS] = lengths[FIELD_TAGS].saturating_add(1);
            let entry = terms.entry(token).or_insert([0; FIELD_COUNT]);
            entry[FIELD_TAGS] = entry[FIELD_TAGS].saturating_add(1);
        }

        DocEntry { terms, lengths }
    }
}

fn extract_text_field(doc: &Value) -> String {
    doc.get("text")
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .unwrap_or_default()
}

fn extract_title_field(doc: &Value) -> String {
    if let Some(title) = doc.get("title").and_then(Value::as_str) {
        return title.to_string();
    }

    doc.get("metadata")
        .and_then(Value::as_object)
        .and_then(|m| m.get("title"))
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .unwrap_or_default()
}

fn extract_tags_field(doc: &Value) -> String {
    fn tags_to_string(value: &Value) -> Option<String> {
        if let Some(s) = value.as_str() {
            return Some(s.to_string());
        }

        let arr = value.as_array()?;
        let mut out = String::new();
        for (i, tag) in arr.iter().enumerate() {
            let Some(s) = tag.as_str() else {
                continue;
            };
            if i > 0 && !out.is_empty() {
                out.push(' ');
            }
            out.push_str(s);
        }
        Some(out)
    }

    if let Some(tags) = doc.get("tags").and_then(tags_to_string) {
        return tags;
    }

    doc.get("metadata")
        .and_then(Value::as_object)
        .and_then(|m| m.get("tags"))
        .and_then(tags_to_string)
        .unwrap_or_default()
}

fn term_freqs(tokens: &[String]) -> HashMap<String, u16> {
    let mut tf = HashMap::new();
    for token in tokens {
        let entry = tf.entry(token.clone()).or_insert(0_u16);
        *entry = entry.saturating_add(1);
    }
    tf
}

fn tokenize(input: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::new();

    for ch in input.chars() {
        if ch.is_alphanumeric() {
            for lc in ch.to_lowercase() {
                buf.push(lc);
            }
        } else if !buf.is_empty() {
            out.push(std::mem::take(&mut buf));
        }
    }

    if !buf.is_empty() {
        out.push(buf);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::LexicalIndex;
    use munind_core::domain::MemoryId;
    use serde_json::json;
    use std::collections::HashSet;

    #[test]
    fn title_and_tags_are_indexed() {
        let mut idx = LexicalIndex::new();
        let id_title = MemoryId(1);
        let id_other = MemoryId(2);

        idx.insert(
            id_title,
            &json!({
                "title": "Apple Guide",
                "text": "neutral text",
                "tags": ["fruit", "orchard"]
            }),
        );
        idx.insert(id_other, &json!({"title": "Banana", "text": "other"}));

        let hits = idx.search("apple orchard", 5);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].0, id_title);
    }

    #[test]
    fn remove_excludes_document_from_results() {
        let mut idx = LexicalIndex::new();
        let id = MemoryId(10);

        idx.insert(id, &json!({"text": "hello memory world"}));
        assert_eq!(idx.search("memory", 5).first().map(|x| x.0), Some(id));

        idx.remove(id);
        assert!(idx.search("memory", 5).is_empty());
    }

    #[test]
    fn metadata_fallback_for_title_and_tags() {
        let mut idx = LexicalIndex::new();
        let id = MemoryId(42);

        idx.insert(
            id,
            &json!({
                "text": "base",
                "metadata": {
                    "title": "Desktop Notes",
                    "tags": ["local", "rag"]
                }
            }),
        );

        let hits = idx.search("desktop rag", 5);
        assert_eq!(hits.first().map(|x| x.0), Some(id));
    }

    #[test]
    fn filtered_search_honors_allow_list() {
        let mut idx = LexicalIndex::new();
        let id1 = MemoryId(1);
        let id2 = MemoryId(2);

        idx.insert(id1, &json!({"text": "apple orchard notes"}));
        idx.insert(id2, &json!({"text": "apple orchard guide"}));

        let allowed = HashSet::from([id2]);
        let hits = idx.search_filtered("apple", 5, &allowed);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, id2);
    }
}
