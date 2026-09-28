import { type RiskLevel, type Decision } from '@/types/fraud';

export const DEFAULT_FRAUD_THRESHOLD = 0.028;

/**
 * Determine decision strictly based on the model decision threshold.
 */
export function getDecision(
  fraudScore: number,
  threshold: number = DEFAULT_FRAUD_THRESHOLD
): Decision {
  return fraudScore >= threshold ? 'Fraud' : 'Legit';
}

/**
 * Categorize visual risk tier based on fraud score ranges and model threshold.
 * Score >= 0.80 or >= 10x threshold → Critical
 * Score >= 0.40 or >= 4x threshold → High
 * Score >= threshold → Medium (Flagged)
 * Score < threshold → Low (Legit)
 */
export function getRiskLevel(
  fraudScore: number,
  threshold: number = DEFAULT_FRAUD_THRESHOLD
): RiskLevel {
  const t = threshold > 0 ? threshold : DEFAULT_FRAUD_THRESHOLD;
  if (fraudScore >= 0.80 || fraudScore >= t * 10) {
    return 'Critical';
  }
  if (fraudScore >= 0.40 || fraudScore >= t * 4) {
    return 'High';
  }
  if (fraudScore >= t) {
    return 'Medium';
  }
  return 'Low';
}
