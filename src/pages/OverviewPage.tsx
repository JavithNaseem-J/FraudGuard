import React, { useEffect, useState } from 'react';
import { RefreshCw, Trash2 } from 'lucide-react';
import { Button, Card, PageHeader } from '@/components/ui';
import {
  MetricsGrid,
  TransactionVolumeChart,
  FraudScoreDistribution,
  RecentFlaggedTransactions,
  SystemStatus,
} from '@/components/dashboard';
import {
  buildDashboardSnapshot,
  clearScoringHistory,
  subscribeToScoringHistory,
  type DashboardSnapshot,
} from '@/services/dashboardStore';

export const OverviewPage: React.FC = () => {
  const [dashboard, setDashboard] = useState<DashboardSnapshot>(() => buildDashboardSnapshot());

  const refreshDashboard = () => {
    setDashboard(buildDashboardSnapshot());
  };

  const clearDashboardData = () => {
    clearScoringHistory();
    setDashboard(buildDashboardSnapshot([]));
  };

  useEffect(() => subscribeToScoringHistory(refreshDashboard), []);

  return (
    <div className="space-y-6">
      {/* 1. Page Header */}
      <PageHeader
        title="Overview"
        description="Real-time insights from your transaction fraud detection system."
        actions={
          <>
            <Button
              type="button"
              variant="secondary"
              size="sm"
              onClick={refreshDashboard}
              icon={<RefreshCw className="w-3.5 h-3.5" />}
            >
              Refresh
            </Button>
            <Button
              type="button"
              variant="danger"
              size="sm"
              onDoubleClick={clearDashboardData}
              disabled={dashboard.isEmpty}
              title="Double-click to clear all locally stored dashboard and scoring history"
              aria-label="Clear all locally stored dashboard and scoring history. Double-click to confirm."
              icon={<Trash2 className="w-3.5 h-3.5" />}
            >
              Clear
            </Button>
          </>
        }
      />

      {dashboard.isEmpty && (
        <Card className="p-4 border-dashed bg-slate-50/60">
          <div className="text-sm font-semibold text-slate-main mb-1">
            No scored transaction data yet
          </div>
          <p className="text-xs text-slate-secondary">
            Score transactions from the scoring page. Successful predictions are stored locally
            in this browser and will appear in the dashboard after scoring or after Refresh.
          </p>
        </Card>
      )}

      {/* 2. Top 5 KPI Metrics */}
      <section aria-label="Key Performance Indicators">
        <MetricsGrid metrics={dashboard.metrics} />
      </section>

      {/* 3. Composed Charts Grid (60% / 40%) */}
      <section
        aria-label="Fraud Detection Analytics"
        className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch"
      >
        <div className="lg:col-span-7">
          <TransactionVolumeChart data={dashboard.dailyVolume} />
        </div>
        <div className="lg:col-span-5">
          <FraudScoreDistribution
            data={dashboard.scoreDistribution}
            threshold={dashboard.threshold}
          />
        </div>
      </section>

      {/* 4. Bottom Grid: Flagged Transactions & System Status (70% / 30%) */}
      <section
        aria-label="Recent Transactions and System Status"
        className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch"
      >
        <div className="lg:col-span-8">
          <RecentFlaggedTransactions transactions={dashboard.recentFlagged} />
        </div>
        <div className="lg:col-span-4">
          <SystemStatus
            lastUpdated={dashboard.lastUpdated}
            modelVersion={dashboard.modelVersion}
            isEmpty={dashboard.isEmpty}
          />
        </div>
      </section>
    </div>
  );
};

export default OverviewPage;
