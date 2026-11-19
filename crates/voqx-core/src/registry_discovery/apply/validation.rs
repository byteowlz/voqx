//! Treat external plans and recovery metadata as data, never executable authority.

use std::collections::BTreeMap;
use std::path::Path;

use super::{MigrationFailure as Failure, Result};
use crate::registry_discovery::inventory::{Inventory, MAX_ENTRIES, portable_path, scan_held};
use crate::registry_discovery::migration::{location, validate_references};
use crate::registry_discovery::{
    EntryKind, MigrationPlan, PlanState, RegistryState, TransferState, check_unlocked, digest,
    read_registry_inner,
};

pub(super) struct Reviewed<'a> {
    plan: &'a MigrationPlan,
    pub(super) source: Inventory,
    baseline: Inventory,
}

impl<'a> Reviewed<'a> {
    pub(super) fn new(plan: &'a MigrationPlan, expected: &str) -> Result<Self> {
        if plan.protocol != "voqx-registry-migration-plan/1"
            || !plan.apply_supported
            || plan.state == PlanState::Blocked
            || !plan.blockers.is_empty()
            || plan.revision.as_deref() != Some(expected)
            || plan.content_revision().ok().as_deref() != Some(expected)
            || !plan.source.root.is_absolute()
            || !plan.destination.is_absolute()
            || location(&plan.source.root).as_ref() != Some(&plan.source.root)
            || location(&plan.destination).as_ref() != Some(&plan.destination)
            || plan.source.root.starts_with(&plan.destination)
            || plan.destination.starts_with(&plan.source.root)
            || plan.source.state != RegistryState::Ready
            || plan.entries.len() > MAX_ENTRIES
        {
            return Err(Failure::InvalidPlan);
        }
        let mut source = BTreeMap::new();
        let mut baseline = BTreeMap::new();
        for entry in &plan.entries {
            let file = entry.source.as_ref().ok_or(Failure::InvalidPlan)?;
            if portable_path(Path::new(&file.path)).as_ref() != Some(&file.path)
                || file.path == super::RECOVERY_DIR
                || file.path.starts_with(&format!("{}/", super::RECOVERY_DIR))
                || file.path == ".lock"
                || source.insert(file.path.clone(), file.clone()).is_some()
            {
                return Err(Failure::InvalidPlan);
            }
            match (&entry.destination, entry.state) {
                (Some(destination), TransferState::AlreadyPresent) if destination == file => {
                    baseline.insert(file.path.clone(), file.clone());
                }
                (None, TransferState::Copy) => {}
                _ => return Err(Failure::InvalidPlan),
            }
        }
        if !source.get("registry.json").is_some_and(|entry| {
            entry.kind == EntryKind::File && entry.sha256 == plan.source.revision
        }) || (baseline.contains_key("registry.json") && baseline != source)
        {
            return Err(Failure::InvalidPlan);
        }
        Ok(Self {
            plan,
            source,
            baseline,
        })
    }

    pub(super) fn inspect(&self, held: bool, recovering: bool) -> Result<Inventory> {
        if !held {
            self.check_unlocked()?;
        }
        let source =
            scan_held(&self.plan.source.root, false).map_err(|_| Failure::SourceChanged)?;
        let destination =
            scan_held(&self.plan.destination, true).map_err(|_| Failure::DestinationChanged)?;
        if source != self.source {
            return Err(Failure::SourceChanged);
        }
        let (registry, bytes) = read_registry_inner(&self.plan.source.root, held)
            .map_err(|_| Failure::SourceChanged)?;
        if Some(digest(&bytes)) != self.plan.source.revision {
            return Err(Failure::SourceChanged);
        }
        validate_references(&registry, &source).map_err(|_| Failure::SourceChanged)?;
        if self
            .baseline
            .iter()
            .any(|(key, entry)| destination.get(key) != Some(entry))
            || destination
                .iter()
                .any(|(key, entry)| source.get(key) != Some(entry))
            || (!recovering && destination != self.baseline && destination != source)
        {
            return Err(Failure::DestinationChanged);
        }
        if !held {
            self.check_unlocked()?;
        }
        Ok(destination)
    }

    pub(super) fn verify_before_registry(&self) -> Result<()> {
        let actual = self.inspect(true, true)?;
        if self
            .source
            .iter()
            .any(|(path, entry)| path != "registry.json" && actual.get(path) != Some(entry))
        {
            return Err(Failure::VerificationFailed);
        }
        Ok(())
    }

    fn check_unlocked(&self) -> Result<()> {
        check_unlocked(&self.plan.source.root).map_err(|_| Failure::Busy)?;
        check_unlocked(&self.plan.destination).map_err(|_| Failure::Busy)?;
        Ok(())
    }
}
