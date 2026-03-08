use munind_core::config::DistanceMetric;

pub fn calculate_distance(metric: &DistanceMetric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        DistanceMetric::L2 => l2_distance(a, b),
        DistanceMetric::Cosine => cosine_distance(a, b),
        DistanceMetric::InnerProduct => inner_product_distance(a, b),
    }
}

pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
}

pub fn inner_product_distance(a: &[f32], b: &[f32]) -> f32 {
    // Return negative IP so lower score is "better" (like distance)
    -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
}

pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }

    // Cosine distance = 1.0 - cosine_similarity
    1.0 - (dot / (norm_a.sqrt() * norm_b.sqrt()))
}
