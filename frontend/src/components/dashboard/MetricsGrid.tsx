import React from 'react';
import { Card } from '@/components/ui';
import { FileText, ShieldAlert, BarChart3, DollarSign, Activity } from 'lucide-react';
import type { KpiMetric } from '@/types/dashboard';

export interface MetricsGridProps {
  metrics?: KpiMetric[];
}

export const MetricsGrid: React.FC<MetricsGridProps> = ({ metrics = [] }) => {
  const getIconProps = (iconName: string) => {
    switch (iconName) {
      case 'file-text':
        return {
          icon: <FileText className="w-4 h-4 text-blue-600" />,
          bgColor: 'bg-blue-50/80 text-blue-600 border-blue-100',
        };
      case 'shield-alert':
        return {
          icon: <ShieldAlert className="w-4 h-4 text-red-600" />,
          bgColor: 'bg-red-50/80 text-red-600 border-red-100',
        };
      case 'bar-chart':
        return {
          icon: <BarChart3 className="w-4 h-4 text-sky-600" />,
          bgColor: 'bg-sky-50/80 text-sky-600 border-sky-100',
        };
      case 'dollar-sign':
        return {
          icon: <DollarSign className="w-4 h-4 text-blue-600" />,
          bgColor: 'bg-blue-50/80 text-blue-600 border-blue-100',
        };
      case 'activity':
      default:
        return {
          icon: <Activity className="w-4 h-4 text-indigo-600" />,
          bgColor: 'bg-indigo-50/80 text-indigo-600 border-indigo-100',
        };
    }
  };

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
      {metrics.map((metric) => {
        const { icon, bgColor } = getIconProps(metric.iconName);
        return (
          <Card
            key={metric.id}
            className="p-4 sm:p-5 flex flex-col justify-between hover:border-slate-300 transition-colors"
          >
            <div className="flex items-start justify-between gap-3">
              <div className="space-y-1">
                <span className="text-xs font-medium text-slate-secondary tracking-normal block leading-tight">
                  {metric.label}
                </span>
                <div className="text-xl sm:text-2xl font-bold text-slate-main tracking-tight pt-0.5">
                  {metric.value}
                </div>
              </div>
              <div className={`w-9 h-9 rounded-lg border flex items-center justify-center shrink-0 ${bgColor}`}>
                {icon}
              </div>
            </div>
          </Card>
        );
      })}
    </div>
  );
};

export default MetricsGrid;
