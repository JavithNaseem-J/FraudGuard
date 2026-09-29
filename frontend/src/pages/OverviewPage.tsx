import React, { useCallback, useEffect, useState } from 'react';
import { Loader2, RefreshCw, Trash2 } from 'lucide-react';
import { Button, Card, PageHeader } from '@/components/ui';
import { FraudScoreDistribution } from '@/components/dashboard/FraudScoreDistribution';
import { MetricsGrid } from '@/components/dashboard/MetricsGrid';
import { RecentFlaggedTransactions } from '@/components/dashboard/RecentFlaggedTransactions';
import { SystemStatus } from '@/components/dashboard/SystemStatus';
import { TransactionVolumeChart } from '@/components/dashboard/TransactionVolumeChart';
import { clearDashboard, getDashboard, getReadyStatus } from '@/services/api';
import {
  buildDashboardSnapshot,
  emptyDashboardSnapshot,
  type DashboardSnapshot,
} from '@/services/dashboardStore';
import { parseApiError } from '@/utils/apiError';
import type { ApiReadyResponse } from '@/types/fraud';

export const OverviewPage: React.FC = () => {
  const [dashboard, setDashboard] = useState<DashboardSnapshot>(
    emptyDashboardSnapshot
  );
  const [readiness, setReadiness] = useState<ApiReadyResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [clearing, setClearing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refreshDashboard = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [snapshot, ready] = await Promise.all([
        getDashboard(),
        getReadyStatus(),
      ]);
      setDashboard(buildDashboardSnapshot(snapshot));
      setReadiness(ready);
    } catch (requestError) {
      setError(parseApiError(requestError));
      setReadiness(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshDashboard();
  }, [refreshDashboard]);

  const clearBrowserView = async () => {
    setClearing(true);
    setError(null);
    try {
      await clearDashboard();
      setDashboard(emptyDashboardSnapshot());
    } catch (requestError) {
      setError(parseApiError(requestError));
    } finally {
      setClearing(false);
    }
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Overview"
        description="A bounded 30-day snapshot of sanitized model predictions."
        actions={
          <>
            <Button
              type="button"
              variant="secondary"
              size="sm"
              onClick={() => void refreshDashboard()}
              disabled={loading || clearing}
              icon={
                loading ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <RefreshCw className="w-3.5 h-3.5" />
                )
              }
            >
              Refresh
            </Button>
            <Button
              type="button"
              variant="danger"
              size="sm"
              onClick={() => void clearBrowserView()}
              onDoubleClick={clearBrowserView}
              disabled={loading || clearing || dashboard.isEmpty}
              title="Clear all predictions from database"
              aria-label="Clear all predictions from database"
              icon={
                clearing ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Trash2 className="w-3.5 h-3.5" />
                )
              }
            >
              Clear
            </Button>
          </>
        }
      />

      {error && (
        <Card className="p-4 border-red-200 bg-red-50">
          <div className="text-sm font-semibold text-red-800 mb-1">
            Dashboard unavailable
          </div>
          <p className="text-xs text-red-700">{error}</p>
        </Card>
      )}


      {dashboard.truncated && (
        <Card className="p-4 border-amber-200 bg-amber-50 text-xs text-amber-800">
          Metrics use the newest 10,000 records in the 30-day window.
        </Card>
      )}

      <section aria-label="Key Performance Indicators">
        <MetricsGrid metrics={dashboard.metrics} />
      </section>

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

      <section
        aria-label="Recent Transactions and System Status"
        className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch"
      >
        <div className="lg:col-span-8">
          <RecentFlaggedTransactions
            transactions={dashboard.recentFlagged}
            threshold={dashboard.threshold}
          />
        </div>
        <div className="lg:col-span-4">
          <SystemStatus
            readiness={readiness}
            lastUpdated={dashboard.lastUpdated}
            isLoading={loading}
            error={error}
          />
        </div>
      </section>
    </div>
  );
};

export default OverviewPage;
