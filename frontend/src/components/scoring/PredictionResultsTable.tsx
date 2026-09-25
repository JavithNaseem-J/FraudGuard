import React, { useMemo } from 'react';
import {
  useReactTable,
  getCoreRowModel,
  getPaginationRowModel,
  flexRender,
  createColumnHelper,
} from '@tanstack/react-table';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { Card, Button, Badge } from '@/components/ui';
import { type PredictionResultRow } from '@/types/fraud';
import { decisionBadgeVariant, riskBadgeVariant } from '@/utils/badgeVariants';

export interface PredictionResultsTableProps {
  results: PredictionResultRow[];
  onDownloadCsv: () => void;
}

const columnHelper = createColumnHelper<PredictionResultRow>();

export const PredictionResultsTable: React.FC<PredictionResultsTableProps> = ({
  results,
}) => {
  const columns = useMemo(
    () => [
      columnHelper.accessor('transactionId', {
        header: 'Transaction ID',
        cell: (info) => (
          <span className="font-mono font-medium text-primary cursor-pointer hover:underline">
            {info.getValue()}
          </span>
        ),
      }),
      columnHelper.accessor('transactionDt', {
        header: 'TransactionDT',
        cell: (info) => <span className="font-mono text-slate-secondary">{info.getValue()}</span>,
      }),
      columnHelper.accessor('transactionAmt', {
        header: 'Transaction Amount',
        cell: (info) => (
          <span className="font-mono font-semibold text-slate-main">
            ${info.getValue().toFixed(2)}
          </span>
        ),
      }),
      columnHelper.accessor('fraudScore', {
        header: 'Fraud Score',
        cell: (info) => (
          <span className="font-mono font-bold text-slate-main">
            {info.getValue().toFixed(2)}
          </span>
        ),
      }),
      columnHelper.accessor('threshold', {
        header: 'Threshold',
        cell: (info) => <span className="font-mono text-slate-muted">{info.getValue().toFixed(2)}</span>,
      }),
      columnHelper.accessor('decision', {
        header: 'Decision',
        cell: (info) => {
          const decision = info.getValue();
          return (
            <Badge variant={decisionBadgeVariant(decision)} size="sm">
              {decision}
            </Badge>
          );
        },
      }),
      columnHelper.accessor('riskLevel', {
        header: 'Risk Level',
        cell: (info) => (
          <Badge variant={riskBadgeVariant(info.getValue())} size="sm">
            {info.getValue()}
          </Badge>
        ),
      }),
    ],
    []
  );

  const table = useReactTable({
    data: results,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    initialState: {
      pagination: {
        pageSize: 10,
      },
    },
  });

  return (
    <Card className="flex flex-col justify-between" noPadding>
      {/* Table Toolbar */}
      <div className="px-5 py-4 border-b border-slate-border bg-white">
        <h4 className="text-sm font-semibold text-slate-main">Scored Prediction Results</h4>
        <p className="text-xs text-slate-muted">
          Threshold: {results[0]?.threshold ?? '—'} &bull; Showing {results.length.toLocaleString()} scored transactions
        </p>
      </div>

      {/* TanStack Table Container */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id} className="border-b border-slate-border bg-slate-50/75">
                {headerGroup.headers.map((header) => (
                  <th
                    key={header.id}
                    className="py-2.5 px-4 font-semibold text-slate-secondary uppercase text-[11px] tracking-wider"
                  >
                    {header.isPlaceholder
                      ? null
                      : flexRender(header.column.columnDef.header, header.getContext())}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody className="divide-y divide-slate-100">
            {table.getRowModel().rows.map((row) => (
              <tr key={row.id} className="hover:bg-slate-50/70 transition-colors">
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id} className="py-2.5 px-4 whitespace-nowrap">
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination Controls */}
      {table.getPageCount() > 1 && (
        <div className="px-5 py-3 border-t border-slate-border bg-slate-50/40 flex items-center justify-between text-xs text-slate-secondary">
          <div>
            Page <span className="font-semibold">{table.getState().pagination.pageIndex + 1}</span> of{' '}
            <span className="font-semibold">{table.getPageCount()}</span>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => table.previousPage()}
              disabled={!table.getCanPreviousPage()}
              icon={<ChevronLeft className="w-3.5 h-3.5" />}
            >
              Previous
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => table.nextPage()}
              disabled={!table.getCanNextPage()}
              icon={<ChevronRight className="w-3.5 h-3.5" />}
            >
              Next
            </Button>
          </div>
        </div>
      )}
    </Card>
  );
};

export default PredictionResultsTable;
