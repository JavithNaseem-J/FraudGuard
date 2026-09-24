import React, { useState } from 'react';
import { Eye, X, ListFilter } from 'lucide-react';
import { Card, Button } from '@/components/ui';
import { PREVIEW_KEY_FIELDS } from '@/constants/transactionSchema';

export interface TransactionPreviewProps {
  firstRow: Record<string, unknown>;
  totalRows: number;
}

export const TransactionPreview: React.FC<TransactionPreviewProps> = ({ firstRow, totalRows }) => {
  const [isModalOpen, setIsModalOpen] = useState(false);

  const allKeys = Object.keys(firstRow);

  return (
    <>
      <Card className="p-5 space-y-3">
        <div className="flex items-center justify-between pb-2 border-b border-slate-border">
          <div>
            <h4 className="text-sm font-semibold text-slate-main">
              Preview — First Transaction Row
            </h4>
            <p className="text-xs text-slate-muted">
              Sample inspection of {totalRows > 1 ? `row 1 of ${totalRows.toLocaleString()}` : 'the single transaction'}
            </p>
          </div>

          <Button
            type="button"
            variant="secondary"
            size="sm"
            onClick={() => setIsModalOpen(true)}
            icon={<Eye className="w-3.5 h-3.5" />}
          >
            View all fields ({allKeys.length})
          </Button>
        </div>

        {/* Key Fields Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 pt-1">
          {PREVIEW_KEY_FIELDS.map(({ key, label }) => {
            const rawVal = firstRow[key];
            const displayVal =
              rawVal !== undefined && rawVal !== null && rawVal !== ''
                ? String(rawVal)
                : '—';

            return (
              <div key={key} className="p-2.5 bg-slate-50 border border-slate-border rounded-input">
                <span className="text-[11px] text-slate-muted block truncate font-medium">
                  {label}
                </span>
                <span className="text-xs font-semibold text-slate-main font-mono block truncate mt-0.5">
                  {key === 'TransactionAmt' && displayVal !== '—' ? `$${displayVal}` : displayVal}
                </span>
              </div>
            );
          })}
        </div>
      </Card>

      {/* Complete Row Modal */}
      {isModalOpen && (
        <div
          role="dialog"
          aria-modal="true"
          className="fixed inset-0 z-50 bg-slate-900/40 backdrop-blur-[1px] flex items-center justify-center p-4"
        >
          <div className="bg-white border border-slate-border rounded-xl shadow-xl max-w-3xl w-full max-h-[85vh] flex flex-col overflow-hidden animate-in fade-in zoom-in-95 duration-150">
            {/* Modal Header */}
            <div className="px-6 py-4 border-b border-slate-border flex items-center justify-between bg-slate-50/50">
              <div className="flex items-center gap-2">
                <ListFilter className="w-4 h-4 text-primary" />
                <h3 className="text-sm font-semibold text-slate-main">
                  Complete Transaction Payload — First Row ({allKeys.length} Fields)
                </h3>
              </div>
              <button
                type="button"
                onClick={() => setIsModalOpen(false)}
                className="p-1.5 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-lg transition-colors"
                aria-label="Close modal"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Modal Body: Scrollable Table of All Fields */}
            <div className="flex-1 overflow-y-auto p-6">
              <div className="border border-slate-border rounded-lg overflow-hidden">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="bg-slate-50 border-b border-slate-border font-semibold text-slate-secondary text-[11px]">
                      <th className="py-2 px-4 w-1/3">Feature Name</th>
                      <th className="py-2 px-4 w-2/3">Value</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 font-mono">
                    {allKeys.map((key) => (
                      <tr key={key} className="hover:bg-slate-50/60">
                        <td className="py-1.5 px-4 font-medium text-slate-700">{key}</td>
                        <td className="py-1.5 px-4 text-slate-main break-all">
                          {String(firstRow[key] ?? 'null')}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Modal Footer */}
            <div className="px-6 py-3 border-t border-slate-border bg-slate-50 flex justify-end">
              <Button variant="secondary" size="sm" onClick={() => setIsModalOpen(false)}>
                Close
              </Button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default TransactionPreview;
