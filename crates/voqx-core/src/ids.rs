//! Stable identifier and slug helpers.

/// Prefix for all voice identity ids.
pub const VOICE_ID_PREFIX: &str = "vox";

/// Slugify a display name into a lowercase ASCII `-`-separated slug.
///
/// Non-alphanumeric runs collapse to a single `-`; leading/trailing separators
/// are trimmed. Returns `"voice"` if nothing usable remains.
#[must_use]
pub fn slugify(name: &str) -> String {
    let slug = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .split('-')
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    if slug.is_empty() {
        "voice".to_string()
    } else {
        slug
    }
}

/// Allocate a `vox-<slug>` id unique under `exists`.
///
/// `exists` returns `true` when a candidate id is already taken. Numeric
/// suffixes are appended on collision.
pub fn allocate_voice_id(name: &str, exists: impl Fn(&str) -> bool) -> String {
    let base = format!("{VOICE_ID_PREFIX}-{}", slugify(name));
    if !exists(&base) {
        return base;
    }
    (2..1_000_000_u32)
        .map(|n| format!("{base}-{n}"))
        .find(|candidate| !exists(candidate))
        .unwrap_or(base)
}

#[cfg(test)]
mod tests {
    use super::{allocate_voice_id, slugify};
    use std::collections::HashSet;

    #[test]
    fn slugify_normalizes() {
        assert_eq!(slugify("Tommy"), "tommy");
        assert_eq!(slugify("  Uncle Fu!! "), "uncle-fu");
        assert_eq!(slugify("東京"), "voice");
        assert_eq!(slugify(""), "voice");
    }

    #[test]
    fn allocate_avoids_collisions() {
        let mut taken: HashSet<String> = HashSet::new();
        let a = allocate_voice_id("Tommy", |id| taken.contains(id));
        assert_eq!(a, "vox-tommy");
        taken.insert(a);
        let b = allocate_voice_id("Tommy", |id| taken.contains(id));
        assert_eq!(b, "vox-tommy-2");
    }
}
