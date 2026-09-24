import { type RiskLevel, type Decision } from '@/types/fraud';

export const DEFAULT_FRAUD_THRESHOLD = 0.72;

/**
 * Determine decision strictly based on the model decision threshold.
 */
export function getDecision(fraudScore: number, threshold: number = DEFAULT_FRAUD_THRESHOLD): Decision {
  return fraudScore >= threshold ? 'Fraud' : 'Legit';
}

/**
 * Categorize visual risk tier based on fraud score ranges.
 * 0.00–0.29 → Low
 * 0.30–0.59 → Medium
 * 0.60–0.79 → High
 * 0.80–1.00 → Critical
 */
export function getRiskLevel(fraudScore: number): RiskLevel {
  if (fraudScore >= 0.80) {
    return 'Critical';
  }
  if (fraudScore >= 0.60) {
    return 'High';
  }
  if (fraudScore >= 0.30) {
    return 'Medium';
  }
  return 'Low';
}
