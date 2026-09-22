## ADDED Requirements

### Requirement: Immutable private artifact releases
The system SHALL publish each production model as an immutable release in private Supabase Storage and SHALL identify the release by an explicit version or release identifier rather than a mutable latest pointer.

#### Scenario: Artifact release published
- **WHEN** an approved transaction model package is promoted
- **THEN** the model files and release manifest SHALL be uploaded under an immutable private release identifier without making raw training data public

### Requirement: Artifact manifest and integrity verification
Each release SHALL include a manifest declaring its artifact schema version, model version, required filenames, byte sizes, and SHA-256 checksums, and the runtime SHALL verify the manifest before loading the model.

#### Scenario: Valid release downloaded
- **WHEN** every downloaded file matches the manifest size and checksum and the package schema is compatible
- **THEN** the runtime SHALL allow the release to proceed to model validation and activation

#### Scenario: Artifact integrity check fails
- **WHEN** a required file is missing or its size, checksum, path, or schema is invalid
- **THEN** the runtime SHALL reject the release, keep readiness false, and emit a sanitized failure category

### Requirement: Atomic release activation and verified cache
The runtime SHALL stage downloads outside the active artifact directory and SHALL activate a release atomically only after integrity and model-package validation succeed. A cached release SHALL be reused only when its identifier and manifest still match the configured release.

#### Scenario: New release activation succeeds
- **WHEN** a staged release passes all validation
- **THEN** the runtime SHALL atomically select it as the active release without exposing a partially downloaded package

#### Scenario: Cached release does not match
- **WHEN** cached artifacts do not match the configured release identifier or manifest
- **THEN** the runtime SHALL NOT load them as the selected production model

### Requirement: Release rollback
Operators SHALL be able to roll back by selecting a previously verified immutable release identifier without overwriting release objects.

#### Scenario: Previous release selected
- **WHEN** an operator changes the configured release identifier to a prior verified release and redeploys
- **THEN** the runtime SHALL validate and activate that release using the same readiness gates as a forward deployment

### Requirement: Artifact credential confidentiality
Storage credentials and signed access material SHALL remain server-side and SHALL NOT be committed, logged, returned by APIs, or embedded in monitoring reports.

#### Scenario: Artifact operation is logged
- **WHEN** publishing or downloading an artifact release emits diagnostics
- **THEN** logs SHALL include release and sanitized outcome metadata without credential values or signed URLs
