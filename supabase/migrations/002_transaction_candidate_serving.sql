ALTER TABLE prediction_requests
    ADD COLUMN IF NOT EXISTS model_mode TEXT NOT NULL DEFAULT 'baseline';

ALTER TABLE model_releases
    ADD COLUMN IF NOT EXISTS model_mode TEXT NOT NULL DEFAULT 'baseline';

ALTER TABLE model_releases
    ADD COLUMN IF NOT EXISTS feature_schema_summary JSONB NOT NULL DEFAULT '{}'::jsonb;

CREATE INDEX IF NOT EXISTS idx_prediction_requests_mode_created
    ON prediction_requests (model_mode, created_at DESC);
