import React, { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AppShell } from '@/components/layout/AppShell';

const OverviewPage = lazy(() => import('@/pages/OverviewPage'));
const ScoreTransactionsPage = lazy(() => import('@/pages/ScoreTransactionsPage'));

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <Suspense fallback={<div className="p-6 text-sm text-slate-secondary">Loading…</div>}>
        <Routes>
          <Route path="/" element={<AppShell />}>
            <Route index element={<OverviewPage />} />
            <Route path="score" element={<ScoreTransactionsPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
};

export default App;
