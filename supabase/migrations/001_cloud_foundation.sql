CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS model_releases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version TEXT NOT NULL,
    model_name TEXT,
    artifact_schema_version INTEGER,
    artifact_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    threshold DOUBLE PRECISION NOT NULL CHECK (threshold >= 0 AND threshold <= 1),
    false_positive_cost DOUBLE PRECISION,
    false_negative_cost DOUBLE PRECISION,
    score_is_calibrated BOOLEAN NOT NULL DEFAULT FALSE,
    released_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (model_version, released_at)
);

CREATE TABLE IF NOT EXISTS prediction_requests (
    prediction_id UUID PRIMARY KEY,
    request_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    score DOUBLE PRECISION NOT NULL CHECK (score >= 0 AND score <= 1),
    threshold DOUBLE PRECISION NOT NULL CHECK (threshold >= 0 AND threshold <= 1),
    decision TEXT NOT NULL CHECK (decision IN ('Yes', 'No')),
    score_is_calibrated BOOLEAN NOT NULL DEFAULT FALSE,
    latency_ms DOUBLE PRECISION,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS prediction_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_id UUID NOT NULL REFERENCES prediction_requests(prediction_id) ON DELETE CASCADE,
    confirmed_label INTEGER CHECK (confirmed_label IN (0, 1)),
    reviewer_decision TEXT,
    feedback_source TEXT NOT NULL DEFAULT 'manual',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS audit_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type TEXT NOT NULL,
    entity_id TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_prediction_requests_model_created
    ON prediction_requests (model_version, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_prediction_requests_request_id
    ON prediction_requests (request_id);

CREATE INDEX IF NOT EXISTS idx_prediction_feedback_prediction
    ON prediction_feedback (prediction_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_model_releases_version
    ON model_releases (model_version, released_at DESC);

CREATE INDEX IF NOT EXISTS idx_audit_events_type_created
    ON audit_events (event_type, created_at DESC);

ALTER TABLE model_releases ENABLE ROW LEVEL SECURITY;
ALTER TABLE prediction_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE prediction_feedback ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_events ENABLE ROW LEVEL SECURITY;
