import { NavLink } from 'react-router-dom';
import { LayoutDashboard, CheckSquare, Shield } from 'lucide-react';
import { cn } from '@/utils/cn';

interface NavItem {
  name: string;
  path: string;
  icon: React.ComponentType<{ className?: string }>;
}

const navItems: NavItem[] = [
  {
    name: 'Overview',
    path: '/',
    icon: LayoutDashboard,
  },
  {
    name: 'Score Transactions',
    path: '/score',
    icon: CheckSquare,
  },
];

export const Sidebar: React.FC = () => {
  return (
    <aside className="w-[240px] bg-sidebar text-sidebar-text flex flex-col justify-between border-r border-sidebar-border select-none shrink-0 h-screen sticky top-0 z-30">
      <div>
        {/* Brand Header */}
        <div className="flex items-center gap-3 px-5 py-5 border-b border-sidebar-border">
          <div className="w-8 h-8 rounded-md bg-primary/20 border border-primary/30 flex items-center justify-center text-primary-soft shrink-0">
            <Shield className="w-4 h-4 text-blue-400" />
          </div>
          <div className="overflow-hidden">
            <h1 className="text-sm font-semibold text-white tracking-wide leading-tight">
              FraudShield
            </h1>
            <p className="text-[11px] text-slate-400 truncate">
              Transaction Fraud Detection
            </p>
          </div>
        </div>

        {/* Navigation Section */}
        <nav className="p-3 space-y-1 mt-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.path}
                to={item.path}
                end={item.path === '/'}
                className={({ isActive }) =>
                  cn(
                    'flex items-center gap-3 px-3 py-2 rounded-btn text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary',
                    isActive
                      ? 'bg-blue-600/15 text-blue-400 border border-blue-500/20 font-semibold'
                      : 'text-slate-400 hover:text-slate-100 hover:bg-sidebar-hover'
                  )
                }
              >
                <Icon className="w-4 h-4 shrink-0" />
                <span className="truncate">{item.name}</span>
              </NavLink>
            );
          })}
        </nav>
      </div>
    </aside>
  );
};

export default Sidebar;
