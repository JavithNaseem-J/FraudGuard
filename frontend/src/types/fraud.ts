export type RiskLevel = 'Low' | 'Medium' | 'High' | 'Critical';
export type Decision = 'Fraud' | 'Legit';

export interface ApiSchemaResponse {
  release_id: string;
  model_version: string;
  model_name: string;
  threshold: number;
  score_is_calibrated: boolean;
  max_batch_rows: number;
  feature_count: number;
  feature_names: string[];
  numeric_features: string[];
  categorical_features: string[];
}

export interface ApiReadyResponse {
  status: 'ready';
  service: string;
  model_loaded: true;
  release_id: string;
  model: {
    model_version: string;
    model_name: string;
    threshold: number;
    score_is_calibrated: boolean;
    feature_count: number;
  };
  persistence: string;
  rate_limit: string;
  rate_limit_policy: {
    requests: number;
    window_seconds: number;
  };
}

export interface ApiVersionResponse {
  commit_sha: string;
  build_time: string;
}

export interface ApiPredictionRow {
  prediction_id: string;
  release_id: string;
  row_index: number;
  fraud_status: 'Yes' | 'No';
  fraud_score: number;
  fraud_probability: number;
  threshold_used: number;
  score_is_calibrated: boolean;
  validation_status: 'valid';
}

export interface ApiBatchPredictionResponse {
  request_id: string;
  release_id: string;
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

export interface ApiDashboardResponse {
  persistence: 'supabase' | 'local_noop';
  window: { start: string; end: string; retention_days: number };
  truncated: boolean;
  transaction_count: number;
  flagged_count: number;
  fraud_rate: number;
  flagged_amount: number;
  average_score: number;
  threshold: number;
  last_updated: string | null;
  model_version: string | null;
  release_id: string | null;
  daily_volume: Array<{
    date: string;
    transactions: number;
    flagged: number;
    fraud_rate: number;
  }>;
  score_distribution: Array<{ score: string; count: number }>;
  recent_flagged: Array<{
    prediction_id: string;
    created_at: string;
    amount: number | null;
    score: number;
    threshold: number;
    decision: 'Yes';
    release_id: string;
  }>;
}

export interface PredictionResultRow {
  transactionId: string;
  rowIndex: number;
  transactionDt: string | number;
  transactionAmt: number;
  fraudScore: number;
  threshold: number;
  decision: Decision;
  riskLevel: RiskLevel;
}

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
