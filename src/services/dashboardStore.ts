import type {
  ApiBatchPredictionResponse,
  PredictionResultRow,
  ScoringBatchSummary,
} from '@/types/fraud';
import type {
  DailyVolumeRecord,
  FlaggedTransaction,
  KpiMetric,
  ScoreDistributionBucket,
} from '@/types/dashboard';

const STORAGE_KEY = 'fraudguard.scoringHistory.v1';
const MAX_STORED_BATCHES = 50;
const MAX_STORED_TRANSACTIONS = 1000;

export const DASHBOARD_HISTORY_EVENT = 'fraudguard:dashboard-history-updated';

export interface StoredScoringBatch {
  id: string;
  scoredAt: string;
  summary: ScoringBatchSummary;
  results: PredictionResultRow[];
  modelVersion: string;
  modelName: string;
  requestId: string;
  featureCount: number;
}

export interface DashboardSnapshot {
  batches: StoredScoringBatch[];
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
  isEmpty: boolean;
}

function hasBrowserStorage(): boolean {
  return typeof window !== 'undefined' && typeof window.localStorage !== 'undefined';
}

function emitHistoryChanged(): void {
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new Event(DASHBOARD_HISTORY_EVENT));
  }
}

function safeNumber(value: unknown, fallback = 0): number {
  const parsed = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function formatCompactNumber(value: number): string {
  return value.toLocaleString('en-US');
}

function formatCurrency(value: number): string {
  return value.toLocaleString('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 0,
  });
}

function formatScore(value: number): string {
  return value.toFixed(4);
}

function formatShortDate(isoDate: string): string {
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
  }).format(new Date(isoDate));
}

function formatDateTime(isoDate: string): string {
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(isoDate));
}

function trimHistory(batches: StoredScoringBatch[]): StoredScoringBatch[] {
  const newestFirst = [...batches].sort(
    (a, b) => new Date(b.scoredAt).getTime() - new Date(a.scoredAt).getTime()
  );

  const kept: StoredScoringBatch[] = [];
  let transactionCount = 0;

  for (const batch of newestFirst) {
    if (kept.length >= MAX_STORED_BATCHES) break;
    if (transactionCount >= MAX_STORED_TRANSACTIONS) break;

    const remaining = MAX_STORED_TRANSACTIONS - transactionCount;
    const results = batch.results.slice(0, remaining);
    kept.push({ ...batch, results });
    transactionCount += results.length;
  }

  return kept.sort(
    (a, b) => new Date(a.scoredAt).getTime() - new Date(b.scoredAt).getTime()
  );
}

export function loadScoringHistory(): StoredScoringBatch[] {
  if (!hasBrowserStorage()) return [];

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return trimHistory(parsed as StoredScoringBatch[]);
  } catch {
    return [];
  }
}

function writeScoringHistory(batches: StoredScoringBatch[]): void {
  if (!hasBrowserStorage()) return;
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(trimHistory(batches)));
  emitHistoryChanged();
}

export function saveScoredBatch(
  summary: ScoringBatchSummary,
  results: PredictionResultRow[],
  apiResponse: ApiBatchPredictionResponse
): StoredScoringBatch {
  const scoredAt = new Date().toISOString();
  const batch: StoredScoringBatch = {
    id: `${scoredAt}-${apiResponse.request_id}`,
    scoredAt,
    summary,
    results,
    modelVersion: apiResponse.model_version,
    modelName: apiResponse.model_name,
    requestId: apiResponse.request_id,
    featureCount: apiResponse.feature_count,
  };

  writeScoringHistory([...loadScoringHistory(), batch]);
  return batch;
}

export function clearScoringHistory(): void {
  if (!hasBrowserStorage()) return;
  window.localStorage.removeItem(STORAGE_KEY);
  emitHistoryChanged();
}

export function subscribeToScoringHistory(listener: () => void): () => void {
  if (typeof window === 'undefined') return () => undefined;

  const handleStorage = (event: StorageEvent) => {
    if (event.key === STORAGE_KEY) listener();
  };

  window.addEventListener(DASHBOARD_HISTORY_EVENT, listener);
  window.addEventListener('storage', handleStorage);

  return () => {
    window.removeEventListener(DASHBOARD_HISTORY_EVENT, listener);
    window.removeEventListener('storage', handleStorage);
  };
}

export function buildDashboardSnapshot(
  batches: StoredScoringBatch[] = loadScoringHistory()
): DashboardSnapshot {
  const flattened = batches.flatMap((batch) =>
    batch.results.map((result) => ({ ...result, batch }))
  );

  const totalTransactions = flattened.length;
  const predictedFraud = flattened.filter((row) => row.decision === 'Fraud').length;
  const fraudRate =
    totalTransactions > 0 ? +((predictedFraud / totalTransactions) * 100).toFixed(2) : 0;
  const amountFlagged = flattened
    .filter((row) => row.decision === 'Fraud')
    .reduce((sum, row) => sum + safeNumber(row.transactionAmt), 0);
  const scoreTotal = flattened.reduce((sum, row) => sum + safeNumber(row.fraudScore), 0);
  const averageFraudScore =
    totalTransactions > 0 ? +(scoreTotal / totalTransactions).toFixed(4) : 0;
  const latestBatch = batches.at(-1);
  const threshold = latestBatch?.results[0]?.threshold ?? 0;

  const metrics: KpiMetric[] = [
    {
      id: 'transactions-scored',
      label: 'Transactions Scored',
      value: formatCompactNumber(totalTransactions),
      change: 'Live',
      changeDirection: 'up',
      changeSentiment: 'neutral',
      secondaryText: 'stored locally',
      iconName: 'file-text',
    },
    {
      id: 'predicted-fraud',
      label: 'Predicted Fraud',
      value: formatCompactNumber(predictedFraud),
      change: totalTransactions > 0 ? `${fraudRate}%` : '0%',
      changeDirection: predictedFraud > 0 ? 'up' : 'down',
      changeSentiment: predictedFraud > 0 ? 'danger' : 'positive',
      secondaryText: 'of scored transactions',
      iconName: 'shield-alert',
    },
    {
      id: 'fraud-rate',
      label: 'Fraud Rate',
      value: `${fraudRate.toFixed(2)}%`,
      change: 'Model',
      changeDirection: 'down',
      changeSentiment: fraudRate > 10 ? 'danger' : 'positive',
      secondaryText: 'current local history',
      iconName: 'bar-chart',
    },
    {
      id: 'amount-flagged',
      label: 'Amount Flagged',
      value: formatCurrency(amountFlagged),
      change: predictedFraud > 0 ? 'Review' : 'Clear',
      changeDirection: predictedFraud > 0 ? 'up' : 'down',
      changeSentiment: predictedFraud > 0 ? 'danger' : 'positive',
      secondaryText: 'predicted fraud amount',
      iconName: 'dollar-sign',
    },
    {
      id: 'avg-fraud-score',
      label: 'Average Fraud Score',
      value: formatScore(averageFraudScore),
      change: threshold > 0 ? `T ${threshold.toFixed(3)}` : 'No threshold',
      changeDirection: averageFraudScore >= threshold && threshold > 0 ? 'up' : 'down',
      changeSentiment:
        averageFraudScore >= threshold && threshold > 0 ? 'danger' : 'positive',
      secondaryText: 'mean model score',
      iconName: 'activity',
    },
  ];

  const dailyMap = new Map<string, { iso: string; transactions: number; fraud: number }>();
  for (const row of flattened) {
    const iso = row.batch.scoredAt.slice(0, 10);
    const current = dailyMap.get(iso) ?? { iso, transactions: 0, fraud: 0 };
    current.transactions += 1;
    if (row.decision === 'Fraud') current.fraud += 1;
    dailyMap.set(iso, current);
  }

  const dailyVolume = [...dailyMap.values()]
    .sort((a, b) => a.iso.localeCompare(b.iso))
    .slice(-30)
    .map((record) => ({
      date: formatShortDate(record.iso),
      transactions: record.transactions,
      fraudRate:
        record.transactions > 0
          ? +((record.fraud / record.transactions) * 100).toFixed(2)
          : 0,
    }));

  const scoreDistribution = Array.from({ length: 21 }, (_, index) => {
    const bucketStart = +(index * 0.05).toFixed(2);
    const bucketEnd = +(bucketStart + 0.05).toFixed(2);
    const count = flattened.filter((row) => {
      const score = safeNumber(row.fraudScore);
      return index === 20
        ? score >= 1
        : score >= bucketStart && score < bucketEnd;
    }).length;

    return {
      score: bucketStart.toFixed(index === 0 || index === 20 ? 1 : 2),
      count,
    };
  });

  const recentFlagged = flattened
    .filter((row) => row.decision === 'Fraud' || row.riskLevel === 'High' || row.riskLevel === 'Critical')
    .sort(
      (a, b) =>
        new Date(b.batch.scoredAt).getTime() - new Date(a.batch.scoredAt).getTime()
    )
    .slice(0, 5)
    .map((row, index) => ({
      transactionId: `${row.transactionId}-${String(index + 1).padStart(2, '0')}`,
      date: formatDateTime(row.batch.scoredAt),
      amount: safeNumber(row.transactionAmt),
      fraudScore: safeNumber(row.fraudScore),
      riskLevel: row.riskLevel,
      decision: row.decision,
    }));

  return {
    batches,
    metrics,
    dailyVolume,
    scoreDistribution,
    recentFlagged,
    totalTransactions,
    predictedFraud,
    fraudRate,
    amountFlagged,
    averageFraudScore,
    threshold,
    lastUpdated: latestBatch?.scoredAt ?? null,
    modelVersion: latestBatch?.modelVersion ?? null,
    isEmpty: totalTransactions === 0,
  };
}
