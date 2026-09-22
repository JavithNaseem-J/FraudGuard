CREATE TABLE IF NOT EXISTS artifact_releases (
    release_id TEXT PRIMARY KEY,
    model_version TEXT NOT NULL,
    artifact_schema_version INTEGER NOT NULL,
    storage_bucket TEXT NOT NULL,
    storage_prefix TEXT NOT NULL,
    manifest JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'published'
        CHECK (status IN ('published', 'verified', 'failed', 'retired')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    verified_at TIMESTAMPTZ
);

ALTER TABLE model_releases
    ADD COLUMN IF NOT EXISTS release_id TEXT;

CREATE INDEX IF NOT EXISTS idx_model_releases_release_id
    ON model_releases (release_id);

CREATE TABLE IF NOT EXISTS monitoring_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_type TEXT NOT NULL CHECK (run_type IN ('drift', 'delayed_label_performance')),
    status TEXT NOT NULL CHECK (status IN ('success', 'no_data', 'insufficient_data', 'failed')),
    release_id TEXT,
    model_version TEXT,
    reference_version TEXT,
    window_start TIMESTAMPTZ,
    window_end TIMESTAMPTZ,
    row_count INTEGER NOT NULL DEFAULT 0,
    label_count INTEGER NOT NULL DEFAULT 0,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    report_path TEXT,
    error_category TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_monitoring_runs_type_created
    ON monitoring_runs (run_type, created_at DESC);

ALTER TABLE artifact_releases ENABLE ROW LEVEL SECURITY;
ALTER TABLE monitoring_runs ENABLE ROW LEVEL SECURITY;

-- Create the private bucket in the Supabase dashboard or CLI if it does not
-- exist. Keep it private and grant object access only to server-side service
-- credentials. Recommended bucket name: fraudguard-model-releases.
