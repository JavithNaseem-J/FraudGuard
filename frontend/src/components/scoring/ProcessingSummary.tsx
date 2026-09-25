import React from 'react';
import { ShieldAlert, ShieldCheck } from 'lucide-react';
import { Card, Badge } from '@/components/ui';
import type { ScoringBatchSummary } from '@/types/fraud';

export interface ProcessingSummaryProps {
  summary: ScoringBatchSummary;
  isSingleTransaction?: boolean;
  singleScore?: number;
  threshold?: number;
}

export const ProcessingSummary: React.FC<ProcessingSummaryProps> = ({
  summary,
  isSingleTransaction = false,
  singleScore = 0,
  threshold = 0.72,
}) => {
  const isSingleFraud = singleScore >= threshold;


  return (
    <div className="space-y-4">

      {isSingleTransaction ? (
        /* ── Single-transaction decision view ── */
        <Card className="p-6">
          <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4 pb-4 border-b border-slate-border">
            <div>
              <span className="text-xs text-slate-muted block">Single Transaction Decision</span>
              <div className="flex items-center gap-3 mt-1.5">
                {isSingleFraud ? (
                  <>
                    <ShieldAlert className="w-6 h-6 text-status-danger" />
                    <span className="text-2xl font-bold text-status-danger tracking-tight">FRAUD</span>
                    <Badge variant="danger" size="md">Above Threshold</Badge>
                  </>
                ) : (
                  <>
                    <ShieldCheck className="w-6 h-6 text-status-success" />
                    <span className="text-2xl font-bold text-status-success tracking-tight">LEGIT</span>
                    <Badge variant="success" size="md">Below Threshold</Badge>
                  </>
                )}
              </div>
            </div>

            <div className="p-3 bg-slate-50 border border-slate-border rounded-lg text-xs space-y-1.5 font-mono min-w-[200px]">
              <div className="flex justify-between gap-6">
                <span className="text-slate-secondary">Fraud Score:</span>
                <strong className="text-slate-main">{singleScore.toFixed(4)}</strong>
              </div>
              <div className="flex justify-between gap-6">
                <span className="text-slate-secondary">Threshold:</span>
                <strong className="text-slate-main">{threshold.toFixed(2)}</strong>
              </div>
              <div className="pt-1.5 border-t border-slate-200 text-slate-700 font-sans text-[11px]">
                {singleScore.toFixed(4)} {isSingleFraud ? '\u2265' : '<'} {threshold.toFixed(2)}{' '}
                &rarr; <strong>{isSingleFraud ? 'FRAUD' : 'LEGIT'}</strong>
              </div>
            </div>
          </div>
        </Card>
      ) : (
        /* ── Batch summary metric cards ── */
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <Card className="p-4">
            <span className="text-xs text-slate-muted block font-medium">Total Transactions</span>
            <div className="text-xl sm:text-2xl font-bold text-slate-main mt-1 font-mono">
              {summary.totalTransactions.toLocaleString()}
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center justify-between">
              <span className="text-xs text-slate-muted font-medium">Predicted Legit</span>
              <span className="text-[11px] font-semibold text-emerald-600 bg-emerald-50 px-1.5 py-0.5 rounded">
                {summary.totalTransactions > 0
                  ? ((summary.predictedLegit / summary.totalTransactions) * 100).toFixed(1)
                  : '0.0'}%
              </span>
            </div>
            <div className="text-xl sm:text-2xl font-bold text-emerald-700 mt-1 font-mono">
              {summary.predictedLegit.toLocaleString()}
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center justify-between">
              <span className="text-xs text-slate-muted font-medium">Predicted Fraud</span>
              <span className="text-[11px] font-semibold text-red-600 bg-red-50 px-1.5 py-0.5 rounded">
                {summary.fraudRate}%
              </span>
            </div>
            <div className="text-xl sm:text-2xl font-bold text-red-600 mt-1 font-mono">
              {summary.predictedFraud.toLocaleString()}
            </div>
          </Card>

          <Card className="p-4">
            <span className="text-xs text-slate-muted block font-medium">Average Fraud Score</span>
            <div className="text-xl sm:text-2xl font-bold text-slate-main mt-1 font-mono">
              {summary.averageFraudScore.toFixed(4)}
            </div>
          </Card>
        </div>
      )}
    </div>
  );
};

export default ProcessingSummary;
