//! Wall-clock helper.

use std::time::{SystemTime, UNIX_EPOCH};

/// Current unix time in seconds, or `0` if the clock is before the epoch.
#[must_use]
pub fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}
