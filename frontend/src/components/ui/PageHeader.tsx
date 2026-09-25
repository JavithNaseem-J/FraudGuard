import React from 'react';
import { cn } from '@/utils/cn';

export interface PageHeaderProps {
  title: string;
  description?: string;
  actions?: React.ReactNode;
  className?: string;
}

export const PageHeader: React.FC<PageHeaderProps> = ({
  title,
  description,
  actions,
  className,
}) => {
  return (
    <div className={cn('flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6 sm:mb-8', className)}>
      <div className="space-y-1">
        <h1 className="text-2xl sm:text-[30px] font-semibold tracking-tight text-slate-main leading-tight">
          {title}
        </h1>
        {description && (
          <p className="text-sm text-slate-secondary">
            {description}
          </p>
        )}
      </div>
      {actions && (
        <div className="flex items-center gap-3 shrink-0">
          {actions}
        </div>
      )}
    </div>
  );
};

export default PageHeader;
