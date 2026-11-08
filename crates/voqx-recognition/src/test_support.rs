//! Fallible test assertions: preserve diagnostics without panicking in Result tests.

/// Compare complete values and propagate a normal test error on mismatch.
///
/// # Errors
/// Returns the debug representations of both unequal values.
pub fn check_eq<L, R>(actual: &L, expected: &R) -> anyhow::Result<()>
where
    L: std::fmt::Debug + PartialEq<R> + ?Sized,
    R: std::fmt::Debug + ?Sized,
{
    anyhow::ensure!(actual == expected, "got {actual:?}, expected {expected:?}");
    Ok(())
}
