import React from 'react';
import { cn } from '@/utils/cn';

export type BadgeVariant = 'success' | 'warning' | 'danger' | 'neutral';
export type BadgeSize = 'sm' | 'md';

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant;
  size?: BadgeSize;
  dot?: boolean;
}

export const Badge: React.FC<BadgeProps> = ({
  className,
  variant = 'neutral',
  size = 'md',
  dot = false,
  children,
  ...props
}) => {
  const baseStyles = 'inline-flex items-center font-medium rounded-badge select-none';

  const variantStyles: Record<BadgeVariant, string> = {
    success: 'bg-status-success-soft text-[#15803D] border border-status-success-border',
    warning: 'bg-status-warning-soft text-[#B45309] border border-status-warning-border',
    danger: 'bg-status-danger-soft text-[#B91C1C] border border-status-danger-border',
    neutral: 'bg-slate-100 text-slate-700 border border-slate-border',
  };

  const dotStyles: Record<BadgeVariant, string> = {
    success: 'bg-[#16A34A]',
    warning: 'bg-[#D97706]',
    danger: 'bg-[#DC2626]',
    neutral: 'bg-slate-400',
  };

  const sizeStyles: Record<BadgeSize, string> = {
    sm: 'px-1.5 py-0.5 text-[11px] gap-1',
    md: 'px-2 py-0.5 text-xs gap-1.5',
  };

  return (
    <span className={cn(baseStyles, variantStyles[variant], sizeStyles[size], className)} {...props}>
      {dot && <span className={cn('w-1.5 h-1.5 rounded-full shrink-0', dotStyles[variant])} />}
      {children}
    </span>
  );
};

export default Badge;
