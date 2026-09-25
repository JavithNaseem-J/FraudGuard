import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { Card } from '@/components/ui';
import type { ScoreDistributionBucket } from '@/types/dashboard';

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{
    value: number;
    dataKey: string;
  }>;
  label?: string;
}

const CustomTooltip: React.FC<CustomTooltipProps> = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    const count = payload[0].value;
    const isAboveThreshold = parseFloat(label || '0') >= 0.72;

    return (
      <div className="bg-white border border-slate-border p-2.5 rounded-lg shadow-card text-xs space-y-1">
        <p className="font-semibold text-slate-main border-b border-slate-100 pb-1">
          Score Bucket: <span className="font-mono">{label}</span>
        </p>
        <div className="flex items-center justify-between gap-4 text-slate-secondary">
          <span>Transaction Count:</span>
          <span className="font-semibold text-slate-main">{count.toLocaleString()}</span>
        </div>
        <div className="text-[11px] pt-0.5">
          {isAboveThreshold ? (
            <span className="text-status-danger font-medium">Above decision threshold (0.72)</span>
          ) : (
            <span className="text-status-success font-medium">Below decision threshold</span>
          )}
        </div>
      </div>
    );
  }
  return null;
};

export interface FraudScoreDistributionProps {
  data?: ScoreDistributionBucket[];
  threshold?: number;
}

export const FraudScoreDistribution: React.FC<FraudScoreDistributionProps> = ({
  data = [],
  threshold = 0.72,
}) => {
  const chartData = data.length > 0 ? data : [{ score: '0.0', count: 0 }];
  const thresholdLabel = threshold > 0 ? threshold.toFixed(2) : '0.72';
  const thresholdBucket = Math.max(0, Math.min(1, Math.round(threshold / 0.05) * 0.05));
  const thresholdBucketLabel =
    thresholdBucket === 0 || thresholdBucket === 1
      ? thresholdBucket.toFixed(1)
      : thresholdBucket.toFixed(2);

  return (
    <Card className="h-full flex flex-col justify-between">
      {/* Header */}
      <div className="pb-3 mb-2 border-b border-slate-border">
        <h3 className="text-sm font-semibold text-slate-main">Fraud Score Distribution</h3>
        <p className="text-xs text-slate-muted">Frequency histogram across score buckets (0.0 – 1.0)</p>
      </div>



      {/* Chart */}
      <div className="w-full h-56 sm:h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            margin={{ top: 20, right: 15, left: -20, bottom: 5 }}
            barCategoryGap={1}
          >
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F1F5F9" />
            <XAxis
              dataKey="score"
              tickLine={false}
              axisLine={{ stroke: '#E2E8F0' }}
              tick={{ fontSize: 10, fill: '#94A3B8' }}
              ticks={['0.0', '0.20', '0.40', '0.60', '0.80', '1.0']}
            />
            <YAxis
              tickLine={false}
              axisLine={false}
              tick={{ fontSize: 10, fill: '#94A3B8' }}
              domain={[0, 1000]}
              ticks={[0, 200, 400, 600, 800, 1000]}
              tickFormatter={(val: number) => val.toLocaleString()}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine
              x={thresholdBucketLabel}
              stroke="#DC2626"
              strokeDasharray="3 3"
              strokeWidth={1.5}
              label={{
                value: `Threshold ${thresholdLabel}`,
                position: 'top',
                fill: '#DC2626',
                fontSize: 10,
                fontWeight: 600,
              }}
            />
            <Bar
              dataKey="count"
              fill="#93C5FD"
              radius={[2, 2, 0, 0]}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="pt-2 text-[11px] text-slate-muted border-t border-slate-100 flex items-center justify-between">
        <span>0.0 (Legitimate)</span>
        <span>1.0 (High Fraud Risk)</span>
      </div>
    </Card>
  );
};

export default FraudScoreDistribution;
