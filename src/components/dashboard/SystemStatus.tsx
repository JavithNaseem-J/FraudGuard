import React from 'react';
import { Cpu, Database, Server, Zap } from 'lucide-react';
import { Card } from '@/components/ui';
import type { ServiceStatusItem } from '@/types/dashboard';

const SYSTEM_SERVICES: ServiceStatusItem[] = [
  { id: 'api', name: 'API', status: 'Operational' },
  { id: 'prediction-service', name: 'Prediction Service', status: 'Operational' },
  { id: 'database', name: 'Database (Supabase)', status: 'Connected' },
  { id: 'redis', name: 'Redis (Upstash)', status: 'Connected' },
];

export interface SystemStatusProps {
  lastUpdated?: string | null;
  modelVersion?: string | null;
  isEmpty?: boolean;
}

export const SystemStatus: React.FC<SystemStatusProps> = ({
  lastUpdated = null,
  modelVersion = null,
  isEmpty = false,
}) => {
  const lastChecked = lastUpdated
    ? new Intl.DateTimeFormat('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      }).format(new Date(lastUpdated))
    : 'not checked yet';

  const overallStatus = isEmpty
    ? 'Ready - waiting for scored transactions'
    : 'All Systems Operational';

  const getServiceIcon = (id: string) => {
    switch (id) {
      case 'api':
        return <Server className="w-3.5 h-3.5 text-slate-400" />;
      case 'prediction-service':
        return <Cpu className="w-3.5 h-3.5 text-slate-400" />;
      case 'database':
        return <Database className="w-3.5 h-3.5 text-slate-400" />;
      case 'redis':
      default:
        return <Zap className="w-3.5 h-3.5 text-slate-400" />;
    }
  };

  return (
    <Card className="h-full flex flex-col justify-between" noPadding>
      <div className="px-5 py-4 border-b border-slate-border">
        <h3 className="text-sm font-semibold text-slate-main">System Status</h3>
      </div>

      <div className="p-5 space-y-4 flex-1">
        <div className="flex items-center justify-between p-2.5 bg-emerald-50/70 border border-emerald-100 rounded-lg text-xs">
          <div className="flex items-center gap-2">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-600" />
            </span>
            <span className="font-semibold text-emerald-900">{overallStatus}</span>
          </div>
          <span className="text-[11px] text-emerald-700">
            Last checked: {lastChecked}
          </span>
        </div>

        {modelVersion && (
          <div className="rounded-lg border border-slate-border bg-slate-50/70 p-3 text-xs">
            <div className="text-slate-muted mb-1">Latest scored model version</div>
            <div className="font-mono text-slate-main break-all">{modelVersion}</div>
          </div>
        )}

        <div className="space-y-2.5">
          {SYSTEM_SERVICES.map((service) => (
            <div
              key={service.id}
              className="flex items-center justify-between py-1.5 px-1 border-b border-slate-100 last:border-0 text-xs"
            >
              <div className="flex items-center gap-2.5 text-slate-700">
                {getServiceIcon(service.id)}
                <span>{service.name}</span>
              </div>
              <span className="font-medium text-emerald-600 text-xs">
                {service.status}
              </span>
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
};

export default SystemStatus;
