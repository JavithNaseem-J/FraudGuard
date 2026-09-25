import React from 'react';
import { Card } from '@/components/ui';
import { cn } from '@/utils/cn';

export interface MetricCardProps {
  label: string;
  value: string;
  change?: string;
  changeDirection?: 'up' | 'down';
  changeSentiment?: 'positive' | 'danger' | 'neutral';
  secondaryText?: string;
  icon: React.ReactNode;
  iconBgColor?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  icon,
  iconBgColor = 'bg-blue-50 text-blue-600 border-blue-100',
}) => {
  return (
    <Card className="p-4 sm:p-5 flex flex-col justify-between hover:border-slate-300 transition-colors">
      <div className="flex items-start justify-between gap-3">
        <div className="space-y-1">
          <span className="text-xs font-medium text-slate-secondary tracking-normal block leading-tight">
            {label}
          </span>
          <div className="text-xl sm:text-2xl font-bold text-slate-main tracking-tight pt-0.5">
            {value}
          </div>
        </div>

        <div
          className={cn(
            'w-9 h-9 rounded-lg border flex items-center justify-center shrink-0',
            iconBgColor
          )}
        >
          {icon}
        </div>
      </div>
    </Card>
  );
};

export default MetricCard;
