# Change 1: actual private-data execution

Start with [the audit report](../../../SPLIT_AUDIT_REPORT.md) and [updated project memory](../../PROJECT_BLUEPRINT.md).

- `raw_identity_audit.json`: actual-file census, verified-near-duplicate rule, missing source coverage and historical overlap.
- `before_after_counts.csv` / `.json`: split × organ × manipulation type × generator counts.
- `identity_coverage.json`: available identity fields, including separate KiTS volume/case coverage.
- `materialized_provenance.json`: independent final hashes, decoded-content components, raw identities and exact-source twins.
- `training_exposure.json`: 88 historical/new candidate stage manifests; 40 regenerated inventories pass. Confirmed historical CF replay consumption is distinguished from filename-only candidate classification.
- `mfclean-split-lock.json` / `extension-split-lock.json`: exact partition manifest hashes and audit results.
- `pool-lock-summary.json`: aggregate pool hash inventory; full private-path lock remains in the run directory.
- `commands.json`, `command_status.json`, `environment.json`, `attempt_history.json`: exact commands, outcomes, versions and preserved failures/recoveries.
- `preservation_checks.json`: historical partition manifest bytes remained unchanged.

No raw images, per-image paths, patient/volume/video identities, duplicate decisions or full private manifests are included. Split/pool separation is verified for available reconstructed identities. Unknown identities and historical test exposure remain limitations. No model was trained or newly evaluated in this task.
