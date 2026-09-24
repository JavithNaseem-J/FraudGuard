import React from 'react';
import { Card, Badge } from '@/components/ui';
import type { FlaggedTransaction } from '@/types/dashboard';
import { decisionBadgeVariant, riskBadgeVariant } from '@/utils/badgeVariants';

export interface RecentFlaggedTransactionsProps {
  transactions?: FlaggedTransaction[];
}

export const RecentFlaggedTransactions: React.FC<RecentFlaggedTransactionsProps> = ({
  transactions = [],
}) => {
  return (
    <Card className="h-full flex flex-col justify-between" noPadding>
      {/* Header */}
      <div className="px-5 py-4 border-b border-slate-border flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-slate-main">Recent Flagged Transactions</h3>
          <p className="text-xs text-slate-muted">High-risk transactions flagged by the model</p>
        </div>
        <span className="text-xs text-slate-secondary font-mono">
          Showing {transactions.length} latest
        </span>
      </div>

      {/* Table Container */}
      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-slate-border bg-slate-50/75 text-[11px] font-semibold text-slate-secondary uppercase tracking-wider">
              <th className="py-2.5 px-5">Transaction ID</th>
              <th className="py-2.5 px-4">Date &amp; Time</th>
              <th className="py-2.5 px-4 text-right">Amount</th>
              <th className="py-2.5 px-4 text-center">Fraud Score</th>
              <th className="py-2.5 px-4 text-center">Risk Level</th>
              <th className="py-2.5 px-5 text-right">Decision</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 text-xs text-slate-main">
            {transactions.map((tx) => (
              <tr
                key={tx.transactionId}
                className="hover:bg-slate-50/80 transition-colors"
              >
                <td className="py-3 px-5 font-mono font-medium text-primary cursor-pointer hover:underline">
                  {tx.transactionId}
                </td>
                <td className="py-3 px-4 text-slate-secondary whitespace-nowrap">
                  {tx.date}
                </td>
                <td className="py-3 px-4 text-right font-medium font-mono">
                  ${tx.amount.toFixed(2)}
                </td>
                <td className="py-3 px-4 text-center font-mono font-semibold text-slate-main">
                  {tx.fraudScore.toFixed(2)}
                </td>
                <td className="py-3 px-4 text-center">
                  <Badge variant={riskBadgeVariant(tx.riskLevel)} size="sm">
                    {tx.riskLevel}
                  </Badge>
                </td>
                <td className="py-3 px-5 text-right">
                  <Badge variant={decisionBadgeVariant(tx.decision)} size="sm">
                    {tx.decision}
                  </Badge>
                </td>
              </tr>
            ))}
            {transactions.length === 0 && (
              <tr>
                <td
                  colSpan={6}
                  className="py-10 px-5 text-center text-xs text-slate-muted"
                >
                  No high-risk transactions in the current dashboard history.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Footer info */}
      <div className="px-5 py-2.5 border-t border-slate-border bg-slate-50/40 text-[11px] text-slate-muted flex items-center justify-between">
        <span>Transactions with score &ge; 0.72 require analyst review</span>
        <span className="text-slate-secondary font-medium">Real-time Feed</span>
      </div>
    </Card>
  );
};

export default RecentFlaggedTransactions;
