import React from 'react';
import { FileText, ShieldAlert, BarChart3, DollarSign, Activity } from 'lucide-react';
import { MetricCard } from './MetricCard';
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
          <MetricCard
            key={metric.id}
            label={metric.label}
            value={metric.value}
            change={metric.change}
            changeDirection={metric.changeDirection}
            changeSentiment={metric.changeSentiment}
            secondaryText={metric.secondaryText}
            icon={icon}
            iconBgColor={bgColor}
          />
        );
      })}
    </div>
  );
};

export default MetricsGrid;
