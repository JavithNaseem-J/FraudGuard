import React from 'react';
import { Cpu, Database, Server, Zap } from 'lucide-react';
import { Card } from '@/components/ui';
import type { ApiReadyResponse } from '@/types/fraud';

export interface SystemStatusProps {
  readiness: ApiReadyResponse | null;
  lastUpdated: string | null;
  isLoading: boolean;
  error: string | null;
}

export const SystemStatus: React.FC<SystemStatusProps> = ({
  readiness,
  lastUpdated,
  isLoading,
  error,
}) => {
  const lastChecked = lastUpdated
    ? new Intl.DateTimeFormat('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      }).format(new Date(lastUpdated))
    : 'no persisted prediction yet';

  const services = [
    {
      name: 'API',
      icon: <Server className="w-3.5 h-3.5 text-slate-400" />,
      value: readiness ? 'Operational' : isLoading ? 'Waking up' : 'Unavailable',
    },
    {
      name: 'Prediction model',
      icon: <Cpu className="w-3.5 h-3.5 text-slate-400" />,
      value: readiness?.model_loaded ? 'Ready' : 'Not ready',
    },
    {
      name: 'Persistence',
      icon: <Database className="w-3.5 h-3.5 text-slate-400" />,
      value: readiness?.persistence === 'supabase' ? 'Supabase' : 'Local no-op',
    },
    {
      name: 'Rate limit',
      icon: <Zap className="w-3.5 h-3.5 text-slate-400" />,
      value: readiness
        ? `${readiness.rate_limit} · ${readiness.rate_limit_policy.requests}/${readiness.rate_limit_policy.window_seconds}s`
        : 'Unknown',
    },
  ];

  const healthy = Boolean(readiness) && !error;
  return (
    <Card className="h-full flex flex-col justify-between" noPadding>
      <div className="px-5 py-4 border-b border-slate-border">
        <h3 className="text-sm font-semibold text-slate-main">System Status</h3>
      </div>
      <div className="p-5 space-y-4 flex-1">
        <div
          className={`flex items-center justify-between p-2.5 border rounded-lg text-xs ${
            healthy
              ? 'bg-emerald-50/70 border-emerald-100 text-emerald-900'
              : 'bg-amber-50/70 border-amber-100 text-amber-900'
          }`}
        >
          <span className="font-semibold">
            {healthy ? 'Service ready' : isLoading ? 'Checking service' : 'Service degraded'}
          </span>
          <span className="text-[11px]">Last data: {lastChecked}</span>
        </div>

        {readiness && (
          <div className="rounded-lg border border-slate-border bg-slate-50/70 p-3 text-xs">
            <div className="text-slate-muted mb-1">Active model release</div>
            <div className="font-mono text-slate-main break-all">
              {readiness.release_id}
            </div>
          </div>
        )}

        <div className="space-y-2.5">
          {services.map((service) => (
            <div
              key={service.name}
              className="flex items-center justify-between py-1.5 px-1 border-b border-slate-100 last:border-0 text-xs"
            >
              <div className="flex items-center gap-2.5 text-slate-700">
                {service.icon}
                <span>{service.name}</span>
              </div>
              <span className="font-medium text-slate-600">{service.value}</span>
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
};

export default SystemStatus;
