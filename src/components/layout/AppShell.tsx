import React from 'react';
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';

export const AppShell: React.FC = () => {
  return (
    <div className="flex min-h-screen w-full bg-workspace antialiased">
      {/* Fixed Sticky Sidebar */}
      <Sidebar />

      {/* Main Workspace */}
      <div className="flex flex-col flex-1 min-w-0">
        {/* Independent Scrollable Page Container */}
        <main className="flex-1 w-full px-6 py-6 lg:px-8 lg:py-8 max-w-none">
          <Outlet />
        </main>
      </div>
    </div>
  );
};

export default AppShell;
