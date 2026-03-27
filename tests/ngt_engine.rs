use munind::index::{IdenticalObjectEdgeType, IndexType};
use munind::{Index, IndexDistanceType, IndexProperty, MmapIndex, SearchOptions};

#[test]
fn test_basic_insert_search() {
    let mut property = IndexProperty::new(2);
    property.set_distance_type(IndexDistanceType::L2);
    property.edge_size_for_creation = 4;
    property.edge_size_for_search = 4;

    let mut index = Index::create("./target/ngt_test", property).expect("create index");
    assert_eq!(index.insert(&[0.0, 0.0]).unwrap(), 1);
    assert_eq!(index.insert(&[1.0, 0.0]).unwrap(), 2);
    assert_eq!(index.insert(&[0.0, 1.0]).unwrap(), 3);
    assert_eq!(index.insert(&[1.0, 1.0]).unwrap(), 4);

    index.build_index();

    let options = SearchOptions {
        k: 2,
        epsilon: 0.0,
        edge_size: Some(4),
    };
    let res = index.search(&[0.9, 0.9], &options).expect("search");
    assert!(!res.is_empty());
    assert_eq!(res[0].id, 4);

    let brute = index.linear_search(&[0.9, 0.9], 2).expect("linear");
    assert_eq!(res[0].id, brute[0].id);
}

#[test]
fn test_save_open() {
    let mut property = IndexProperty::new(2);
    property.edge_size_for_creation = 3;
    property.edge_size_for_search = 3;

    let mut index = Index::create("./target/ngt_test_save", property).unwrap();
    index.insert(&[0.0, 0.0]).unwrap();
    index.insert(&[1.0, 1.0]).unwrap();
    index.build_index();
    index.save(Some("./target/ngt_test_save.bin")).unwrap();

    let loaded = Index::open("./target/ngt_test_save.bin").unwrap();
    assert_eq!(loaded.objects.len(), 2);
    assert_eq!(loaded.graph.edges.len(), 2);
}

#[test]
fn test_cosine_normalizes_inserted_objects_and_queries() {
    let mut property = IndexProperty::new(2);
    property.set_distance_type(IndexDistanceType::Cosine);
    property.edge_size_for_creation = 4;
    property.edge_size_for_search = 4;

    let mut index = Index::create("./target/ngt_test_cosine", property).unwrap();
    assert_eq!(index.insert(&[10.0, 0.0]).unwrap(), 1);
    assert_eq!(index.insert(&[0.0, 5.0]).unwrap(), 2);
    index.build_index();

    let options = SearchOptions {
        k: 1,
        epsilon: 0.0,
        edge_size: Some(4),
    };
    let res = index.search(&[2.0, 0.0], &options).unwrap();
    assert_eq!(res[0].id, 1);

    let stored = index.object_space.as_ref().unwrap().get_object(1).unwrap();
    assert!((stored[0] - 1.0).abs() < 1e-6);
    assert!(stored[1].abs() < 1e-6);
}

#[test]
fn test_default_property_matches_ngt_defaults() {
    let property = IndexProperty::new(16);
    assert_eq!(property.thread_pool_size, 32);
    assert_eq!(property.edge_size_for_creation, 10);
    assert_eq!(property.edge_size_for_search, 0);
    assert_eq!(property.seed_size, 10);
    assert_eq!(property.batch_size_for_creation, 200);
    assert_eq!(property.outgoing_edge, 10);
    assert_eq!(property.incoming_edge, 80);
}

#[test]
fn test_save_open_directory_directory_layout() {
    let mut property = IndexProperty::new(2);
    property.set_distance_type(IndexDistanceType::Cosine);
    property.edge_size_for_creation = 3;
    property.edge_size_for_search = 3;

    let mut index = Index::create("./target/ngt_dir", property).unwrap();
    index.insert(&[1.0, 0.0]).unwrap();
    index.insert(&[0.0, 1.0]).unwrap();
    index.insert(&[1.0, 1.0]).unwrap();
    index.build_index();
    index.save_as_directory("./target/ngt_dir").unwrap();

    let loaded = Index::open_directory("./target/ngt_dir").unwrap();
    assert_eq!(loaded.objects.len(), 3);
    assert!(loaded.tree.is_some());

    let options = munind::SearchOptions {
        k: 1,
        epsilon: 0.0,
        edge_size: Some(3),
    };
    let result = loaded.search(&[2.0, 0.0], &options).unwrap();
    assert_eq!(result[0].id, 1);
}

#[test]
fn test_identical_object_directed_edge_behavior() {
    let mut property = IndexProperty::new(2);
    property.identical_object_edge_type = IdenticalObjectEdgeType::DirectedEdge;

    let mut index = Index::create("./target/ngt_identical", property).unwrap();
    index.insert(&[1.0, 0.0]).unwrap();
    index.insert(&[1.0, 0.0]).unwrap();
    index.insert(&[0.0, 1.0]).unwrap();
    index.build_index();

    assert!(
        !index.graph.edges[1]
            .iter()
            .any(|edge| edge.id == 1 && edge.distance == 0.0)
    );
    assert!(
        index.graph.edges[0]
            .iter()
            .any(|edge| edge.id == 2 && edge.distance == 0.0)
    );
}

#[test]
fn test_tree_splits_and_returns_leaf_seeds() {
    let mut property = IndexProperty::new(2);
    property.leaf_node_size = 2;
    property.internal_children_size = 2;
    property.edge_size_for_creation = 2;
    property.edge_size_for_search = 2;

    let mut index = Index::create("./target/ngt_tree_split", property).unwrap();
    index.insert(&[0.0, 0.0]).unwrap();
    index.insert(&[0.0, 1.0]).unwrap();
    index.insert(&[10.0, 10.0]).unwrap();
    index.insert(&[10.0, 11.0]).unwrap();
    index.build_index();

    let tree = index.tree.as_ref().expect("tree must exist");
    let object_space = index.object_space.as_ref().expect("object space");
    let leaf_id = tree.leaf_for_query(&[10.0, 10.2], object_space).unwrap();
    let seeds = tree.get_object_ids_from_leaf(leaf_id);

    assert!(tree.leaves.iter().flatten().count() >= 2);
    assert!(seeds.iter().any(|seed| seed.id == 3));
    assert!(seeds.iter().any(|seed| seed.id == 4));
}

#[test]
fn test_graph_only_build_does_not_create_tree() {
    let mut property = IndexProperty::new(2);
    property.index_type = IndexType::Graph;
    property.edge_size_for_creation = 3;

    let mut index = Index::create("./target/ngt_graph_only", property).unwrap();
    index.insert(&[0.0, 0.0]).unwrap();
    index.insert(&[1.0, 0.0]).unwrap();
    index.insert(&[0.0, 1.0]).unwrap();
    index.build_index();

    assert!(index.tree.is_none());
    assert_eq!(index.graph.edges.len(), 3);
}

#[test]
fn test_tree_guided_insertion_uses_current_object_leaf() {
    let mut property = IndexProperty::new(2);
    property.leaf_node_size = 2;
    property.internal_children_size = 2;
    property.edge_size_for_creation = 1;
    property.edge_size_for_search = 1;

    let mut index = Index::create("./target/ngt_tree_insertion", property).unwrap();
    index.insert(&[0.0, 0.0]).unwrap();
    index.insert(&[0.0, 1.0]).unwrap();
    index.insert(&[10.0, 10.0]).unwrap();
    index.insert(&[10.0, 11.0]).unwrap();
    index.build_index();

    assert_eq!(index.graph.edges[3][0].id, 3);
}

#[test]
fn test_parallel_batch_build_produces_graph() {
    let mut property = IndexProperty::new(2);
    property.thread_pool_size = 4;
    property.batch_size_for_creation = 3;
    property.edge_size_for_creation = 2;
    property.edge_size_for_search = 2;

    let mut index = Index::create("./target/ngt_parallel_build", property).unwrap();
    index.insert(&[0.0, 0.0]).unwrap();
    index.insert(&[1.0, 0.0]).unwrap();
    index.insert(&[0.0, 1.0]).unwrap();
    index.insert(&[1.0, 1.0]).unwrap();
    index.insert(&[2.0, 2.0]).unwrap();
    index.build_index();

    assert_eq!(index.graph.edges.len(), 5);
    assert!(index.graph.edges.iter().map(Vec::len).sum::<usize>() > 0);
}

#[test]
fn test_save_open_mmap_directory_layout() {
    let mut property = IndexProperty::new(2);
    property.set_distance_type(IndexDistanceType::Cosine);
    property.edge_size_for_creation = 3;
    property.edge_size_for_search = 3;

    let mut index = Index::create("./target/ngt_mmap_dir", property).unwrap();
    index.insert(&[1.0, 0.0]).unwrap();
    index.insert(&[0.0, 1.0]).unwrap();
    index.insert(&[1.0, 1.0]).unwrap();
    index.build_index();
    index.save_as_mmap("./target/ngt_mmap_dir").unwrap();

    let loaded = MmapIndex::open("./target/ngt_mmap_dir").unwrap();
    assert_eq!(loaded.object_count(), 3);

    let options = SearchOptions {
        k: 1,
        epsilon: 0.0,
        edge_size: Some(3),
    };
    let result = loaded.search(&[2.0, 0.0], &options).unwrap();
    assert_eq!(result[0].id, 1);

    let linear = loaded.linear_search(&[0.0, 2.0], 1).unwrap();
    assert_eq!(linear[0].id, 2);
}
