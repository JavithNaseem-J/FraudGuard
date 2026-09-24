import React from 'react';
import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { Card } from '@/components/ui';
import type { DailyVolumeRecord } from '@/types/dashboard';

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{
    value: number;
    dataKey: string;
    name: string;
    color: string;
  }>;
  label?: string;
}

const CustomTooltip: React.FC<CustomTooltipProps> = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    const transactions = payload.find((p) => p.dataKey === 'transactions')?.value;
    const fraudRate = payload.find((p) => p.dataKey === 'fraudRate')?.value;

    return (
      <div className="bg-white border border-slate-border p-2.5 rounded-lg shadow-card text-xs space-y-1">
        <p className="font-semibold text-slate-main border-b border-slate-100 pb-1">{label}</p>
        <div className="flex items-center justify-between gap-4 text-slate-secondary">
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-sm bg-blue-400" />
            Transactions:
          </span>
          <span className="font-medium text-slate-main">{transactions?.toLocaleString()}</span>
        </div>
        <div className="flex items-center justify-between gap-4 text-slate-secondary">
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-red-500" />
            Fraud Rate:
          </span>
          <span className="font-medium text-red-600">{fraudRate}%</span>
        </div>
      </div>
    );
  }
  return null;
};

export interface TransactionVolumeChartProps {
  data?: DailyVolumeRecord[];
}

export const TransactionVolumeChart: React.FC<TransactionVolumeChartProps> = ({
  data = [],
}) => {
  const chartData = data.length > 0 ? data : [{ date: 'No data', transactions: 0, fraudRate: 0 }];

  return (
    <Card className="h-full flex flex-col justify-between">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 mb-2 border-b border-slate-border">
        <div>
          <h3 className="text-sm font-semibold text-slate-main">
            Transaction Volume &amp; Fraud Rate
          </h3>
          <p className="text-xs text-slate-muted">Daily transaction counts and detected fraud rate</p>
        </div>
      </div>

      {/* Chart */}
      <div className="w-full h-64 sm:h-72">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={chartData}
            margin={{ top: 10, right: 10, left: -15, bottom: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F1F5F9" />
            <XAxis
              dataKey="date"
              tickLine={false}
              axisLine={{ stroke: '#E2E8F0' }}
              tick={{ fontSize: 11, fill: '#94A3B8' }}
              interval={4}
            />
            {/* Left Y Axis: Transactions */}
            <YAxis
              yAxisId="left"
              tickLine={false}
              axisLine={false}
              tick={{ fontSize: 11, fill: '#94A3B8' }}
              domain={[0, 1000]}
              ticks={[0, 200, 400, 600, 800, 1000]}
              tickFormatter={(val: number) => val.toLocaleString()}
            />
            {/* Right Y Axis: Fraud Rate */}
            <YAxis
              yAxisId="right"
              orientation="right"
              tickLine={false}
              axisLine={false}
              tick={{ fontSize: 11, fill: '#94A3B8' }}
              domain={[0, 8]}
              ticks={[0, 2, 4, 6, 8]}
              tickFormatter={(val: number) => `${val}%`}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              verticalAlign="top"
              align="center"
              iconType="circle"
              iconSize={7}
              wrapperStyle={{ paddingBottom: '12px', fontSize: '11px', color: '#64748B' }}
              formatter={(value) => (
                <span className="text-xs text-slate-secondary font-medium mr-3">
                  {value === 'transactions' ? 'Transactions' : 'Fraud Rate'}
                </span>
              )}
            />
            <Bar
              yAxisId="left"
              dataKey="transactions"
              name="transactions"
              fill="#93C5FD"
              radius={[2, 2, 0, 0]}
              maxBarSize={16}
            />
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="fraudRate"
              name="fraudRate"
              stroke="#EF4444"
              strokeWidth={2}
              dot={{ r: 2.5, fill: '#EF4444', strokeWidth: 0 }}
              activeDot={{ r: 4, stroke: '#FFFFFF', strokeWidth: 2 }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
};

export default TransactionVolumeChart;
