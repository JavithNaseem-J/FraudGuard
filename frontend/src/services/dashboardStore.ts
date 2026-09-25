import type {
  DailyVolumeRecord,
  FlaggedTransaction,
  KpiMetric,
  ScoreDistributionBucket,
} from '@/types/dashboard';
import type { ApiDashboardResponse } from '@/types/fraud';
import { getRiskLevel } from '@/utils/riskLevel';

export interface DashboardSnapshot {
  metrics: KpiMetric[];
  dailyVolume: DailyVolumeRecord[];
  scoreDistribution: ScoreDistributionBucket[];
  recentFlagged: FlaggedTransaction[];
  totalTransactions: number;
  predictedFraud: number;
  fraudRate: number;
  amountFlagged: number;
  averageFraudScore: number;
  threshold: number;
  lastUpdated: string | null;
  modelVersion: string | null;
  releaseId: string | null;
  persistence: string;
  truncated: boolean;
  isEmpty: boolean;
}

export function emptyDashboardSnapshot(): DashboardSnapshot {
  const empty: ApiDashboardResponse = {
    persistence: 'local_noop',
    window: { start: '', end: '', retention_days: 30 },
    truncated: false,
    transaction_count: 0,
    flagged_count: 0,
    fraud_rate: 0,
    flagged_amount: 0,
    average_score: 0,
    threshold: 0,
    last_updated: null,
    model_version: null,
    release_id: null,
    daily_volume: [],
    score_distribution: Array.from({ length: 10 }, (_, index) => ({
      score: (index / 10).toFixed(1),
      count: 0,
    })),
    recent_flagged: [],
  };
  return buildDashboardSnapshot(empty);
}

const formatNumber = (value: number): string => value.toLocaleString('en-US');
const formatCurrency = (value: number): string =>
  value.toLocaleString('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 0,
  });

export function buildDashboardSnapshot(source: ApiDashboardResponse): DashboardSnapshot {
  const metrics: KpiMetric[] = [
    {
      id: 'transactions-scored',
      label: 'Transactions Scored',
      value: formatNumber(source.transaction_count),
      change: '30 days',
      changeDirection: 'up',
      changeSentiment: 'neutral',
      secondaryText: source.truncated ? 'bounded snapshot' : 'persisted predictions',
      iconName: 'file-text',
    },
    {
      id: 'predicted-fraud',
      label: 'Predicted Fraud',
      value: formatNumber(source.flagged_count),
      change: `${source.fraud_rate.toFixed(2)}%`,
      changeDirection: source.flagged_count > 0 ? 'up' : 'down',
      changeSentiment: source.flagged_count > 0 ? 'danger' : 'positive',
      secondaryText: 'flagged by the model',
      iconName: 'shield-alert',
    },
    {
      id: 'fraud-rate',
      label: 'Flag Rate',
      value: `${source.fraud_rate.toFixed(2)}%`,
      change: 'Model',
      changeDirection: 'down',
      changeSentiment: 'neutral',
      secondaryText: 'not confirmed fraud',
      iconName: 'bar-chart',
    },
    {
      id: 'amount-flagged',
      label: 'Amount Flagged',
      value: formatCurrency(source.flagged_amount),
      change: source.flagged_count > 0 ? 'Review' : 'Clear',
      changeDirection: source.flagged_count > 0 ? 'up' : 'down',
      changeSentiment: source.flagged_count > 0 ? 'danger' : 'positive',
      secondaryText: 'transaction amount',
      iconName: 'dollar-sign',
    },
    {
      id: 'avg-fraud-score',
      label: 'Average Fraud Score',
      value: source.average_score.toFixed(4),
      change: source.threshold > 0 ? `T ${source.threshold.toFixed(3)}` : 'No data',
      changeDirection: 'down',
      changeSentiment: 'neutral',
      secondaryText: 'model score, not probability',
      iconName: 'activity',
    },
  ];

  return {
    metrics,
    dailyVolume: source.daily_volume.map((row) => ({
      date: new Intl.DateTimeFormat('en-US', {
        month: 'short',
        day: 'numeric',
      }).format(new Date(`${row.date}T00:00:00Z`)),
      transactions: row.transactions,
      fraudRate: row.fraud_rate,
    })),
    scoreDistribution: source.score_distribution,
    recentFlagged: source.recent_flagged.map((row) => ({
      transactionId: `PRED-${row.prediction_id}`,
      date: new Intl.DateTimeFormat('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      }).format(new Date(row.created_at)),
      amount: row.amount ?? 0,
      fraudScore: row.score,
      riskLevel: getRiskLevel(row.score),
      decision: 'Fraud',
    })),
    totalTransactions: source.transaction_count,
    predictedFraud: source.flagged_count,
    fraudRate: source.fraud_rate,
    amountFlagged: source.flagged_amount,
    averageFraudScore: source.average_score,
    threshold: source.threshold,
    lastUpdated: source.last_updated,
    modelVersion: source.model_version,
    releaseId: source.release_id,
    persistence: source.persistence,
    truncated: source.truncated,
    isEmpty: source.transaction_count === 0,
  };
}
