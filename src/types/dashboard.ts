import type { Decision, RiskLevel } from './fraud';

export interface KpiMetric {
  id: string;
  label: string;
  value: string;
  change: string;
  changeDirection: 'up' | 'down';
  changeSentiment: 'positive' | 'danger' | 'neutral';
  secondaryText: string;
  iconName: 'file-text' | 'shield-alert' | 'bar-chart' | 'dollar-sign' | 'activity';
}

export interface DailyVolumeRecord {
  date: string;
  transactions: number;
  fraudRate: number;
}

export interface ScoreDistributionBucket {
  score: string;
  count: number;
}

export interface FlaggedTransaction {
  transactionId: string;
  date: string;
  amount: number;
  fraudScore: number;
  riskLevel: RiskLevel;
  decision: Decision;
}

export interface ServiceStatusItem {
  id: 'api' | 'prediction-service' | 'database' | 'redis';
  name: string;
  status: 'Operational' | 'Connected' | 'Degraded' | 'Down';
}
