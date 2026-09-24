// ─── Frontend display types ───────────────────────────────────────────────────

export type RiskLevel = 'Low' | 'Medium' | 'High' | 'Critical';

/** Normalized frontend decision (mapped from backend fraud_status "Yes"/"No") */
export type Decision = 'Fraud' | 'Legit';

// ─── Backend API response types ───────────────────────────────────────────────
// These match the exact FastAPI response shapes — do not modify field names.

/**
 * GET /schema/transactions
 */
export interface ApiSchemaResponse {
  model_mode: string;
  model_version: string;
  model_name: string;
  threshold: number;
  score_is_calibrated: boolean;
  max_batch_rows: number;
  feature_count: number;
  feature_names: string[];
  numeric_features: string[];
  categorical_features: string[];
  schema_risks: string[];
}

/**
 * GET /ready
 */
export interface ApiReadyResponse {
  status: 'ready' | 'not_ready';
  service: string;
  model_mode: string;
  model_loaded: boolean;
  candidate_model_loaded: boolean;
  threshold: number;
  model_version: string;
  persistence: string;
  rate_limit: string;
}

/**
 * GET /version
 */
export interface ApiVersionResponse {
  commit_sha: string;
  build_time: string;
}

/**
 * Raw per-row prediction result from POST /predict/transactions.
 * Note: fraud_status is "Yes" for fraud and "No" for legitimate.
 */
export interface ApiPredictionRow {
  prediction_id: string;
  row_index: number;
  fraud_status: 'Yes' | 'No';
  fraud_score: number;
  fraud_probability: number;
  threshold_used: number;
  score_is_calibrated: boolean;
}

/**
 * Full response from POST /predict/transactions.
 */
export interface ApiBatchPredictionResponse {
  request_id: string;
  model_mode: string;
  model_version: string;
  model_name: string;
  row_count: number;
  feature_count: number;
  ignored_features: string[];
  results: ApiPredictionRow[];
  latency_ms: number;
  persistence: string;
  persisted_count: number;
  rate_limit: string;
}

// ─── Normalized frontend prediction row ──────────────────────────────────────
// Maps raw backend fields into UI-friendly names in one place.

export interface PredictionResultRow {
  /** Auto-generated display ID: TX-000001, TX-000002, … */
  transactionId: string;
  /** Original row index from the backend response */
  rowIndex: number;
  /** Mapped from the source CSV/JSON row */
  transactionDt: string | number;
  transactionAmt: number;
  /** Mapped from ApiPredictionRow.fraud_score */
  fraudScore: number;
  /** Mapped from ApiPredictionRow.threshold_used */
  threshold: number;
  /** Mapped: "Yes" → "Fraud", "No" → "Legit" */
  decision: Decision;
  /** Derived from fraudScore via getRiskLevel() */
  riskLevel: RiskLevel;
}

// ─── Scoring batch summary ────────────────────────────────────────────────────

export interface ScoringBatchSummary {
  totalTransactions: number;
  predictedLegit: number;
  predictedFraud: number;
  fraudRate: number;
  averageFraudScore: number;
  processingDurationSeconds: number;
  modelVersion: string;
  modelName: string;
}

// ─── Legacy types (kept for Overview dashboard mock data) ────────────────────

export interface Transaction {
  id: string;
  timestamp: string;
  amount: number;
  fraudScore: number;
  riskLevel: RiskLevel;
  decision: Decision;
  merchant?: string;
}

export interface MetricSummary {
  label: string;
  value: string | number;
  changePercent?: number;
  trend?: 'up' | 'down' | 'neutral';
  timeframeText?: string;
}

export interface OverviewMetrics {
  transactionsScored: number;
  predictedFraud: number;
  fraudRate: number;
  amountFlagged: number;
  averageFraudScore: number;
}

export interface SystemStatus {
  allOperational: boolean;
  lastChecked: string;
  api: 'Operational' | 'Degraded' | 'Down';
  predictionService: 'Operational' | 'Degraded' | 'Down';
  database: 'Connected' | 'Disconnected';
  redis: 'Connected' | 'Disconnected';
  modelReady: boolean;
  modelVersion: string;
  modelDeployedDate: string;
}
