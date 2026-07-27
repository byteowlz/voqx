//! Pure embedding math: cosine similarity and L2-normalized centroids.
//!
//! No ML dependencies — these operate on plain `f32` vectors produced elsewhere.

/// Cosine similarity of two vectors in `[-1, 1]`.
///
/// Returns `0.0` if the lengths differ, either vector is empty, or either has
/// zero norm.
#[must_use]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < f32::EPSILON || nb < f32::EPSILON {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// The L2-normalized mean of the given embeddings.
///
/// Returns an empty vector if `embeddings` is empty. Vectors shorter than the
/// first are zero-padded implicitly by only summing the overlapping prefix; the
/// caller is responsible for providing consistent dimensionality.
#[must_use]
pub fn centroid(embeddings: &[Vec<f32>]) -> Vec<f32> {
    let Some(dim) = embeddings.first().map(Vec::len) else {
        return Vec::new();
    };
    let mut mean = vec![0.0_f32; dim];
    for embedding in embeddings {
        for (m, v) in mean.iter_mut().zip(embedding) {
            *m += *v;
        }
    }
    let count = embeddings.len() as f32;
    for m in &mut mean {
        *m /= count;
    }
    let norm: f32 = mean.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm >= f32::EPSILON {
        for m in &mut mean {
            *m /= norm;
        }
    }
    mean
}

#[cfg(test)]
mod tests {
    use super::{centroid, cosine};

    #[test]
    fn cosine_basics() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert!(cosine(&[], &[]).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[1.0]).abs() < 1e-6);
    }

    #[test]
    fn centroid_is_unit_norm() {
        let c = centroid(&[vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]);
        let norm: f32 = c.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn centroid_of_empty_is_empty() {
        assert!(centroid(&[]).is_empty());
    }
}
