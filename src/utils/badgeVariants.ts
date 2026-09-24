import type { BadgeVariant } from '@/components/ui';
import type { Decision, RiskLevel } from '@/types/fraud';

export function riskBadgeVariant(risk: RiskLevel): BadgeVariant {
  switch (risk) {
    case 'Critical':
      return 'danger';
    case 'High':
    case 'Medium':
      return 'warning';
    case 'Low':
    default:
      return 'success';
  }
}

export function decisionBadgeVariant(decision: Decision): BadgeVariant {
  return decision === 'Fraud' ? 'danger' : 'success';
}
