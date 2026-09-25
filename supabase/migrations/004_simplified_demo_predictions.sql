-- Minimal server-only persistence for the public production-style demo.
ALTER TABLE prediction_requests
    ADD COLUMN IF NOT EXISTS transaction_amount DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS release_id TEXT;

ALTER TABLE prediction_requests
    DROP COLUMN IF EXISTS model_mode,
    DROP COLUMN IF EXISTS metadata;

ALTER TABLE model_releases
    DROP COLUMN IF EXISTS model_mode;

DROP TABLE IF EXISTS prediction_feedback;
DROP TABLE IF EXISTS audit_events;
DROP TABLE IF EXISTS monitoring_runs;
DROP TABLE IF EXISTS artifact_releases;

CREATE INDEX IF NOT EXISTS idx_prediction_requests_created_at
    ON prediction_requests (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_prediction_requests_flagged_created_at
    ON prediction_requests (created_at DESC)
    WHERE decision = 'Yes';

CREATE INDEX IF NOT EXISTS idx_prediction_requests_release_created_at
    ON prediction_requests (release_id, created_at DESC);

CREATE UNIQUE INDEX IF NOT EXISTS idx_model_releases_unique_release_id
    ON model_releases (release_id)
    WHERE release_id IS NOT NULL;

-- RLS remains enabled. Browser clients receive no table policy; all access is
-- through FastAPI with the server-only service-role credential.
ALTER TABLE prediction_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_releases ENABLE ROW LEVEL SECURITY;
