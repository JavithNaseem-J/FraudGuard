import React from 'react';
import { FileSpreadsheet, X, Loader2, Play, CheckCircle2, AlertCircle } from 'lucide-react';
import { Card, Button, Badge } from '@/components/ui';
import { type SchemaValidationResult } from './SchemaValidation';

export interface FileMetadata {
  fileName: string;
  fileSizeBytes: number;
  rowCount: number;
  columnCount: number;
  fileType: 'CSV' | 'JSON';
}

export interface FileInformationProps {
  metadata: FileMetadata;
  validationResult: SchemaValidationResult;
  isProcessing: boolean;
  onRunScoring: () => void;
  onClear: () => void;
}

export const FileInformation: React.FC<FileInformationProps> = ({
  metadata,
  validationResult,
  isProcessing,
  onRunScoring,
  onClear,
}) => {
  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <Card className="p-5 flex flex-col justify-between space-y-4">
      {/* Top row: File info and remove button */}
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-center gap-3 min-w-0">
          <div className="w-10 h-10 rounded-lg bg-blue-50 border border-blue-200 text-primary flex items-center justify-center shrink-0">
            <FileSpreadsheet className="w-5 h-5" />
          </div>
          <div className="truncate">
            <div className="text-xs text-slate-muted">File Information</div>
            <h4 className="text-sm font-semibold text-slate-main truncate" title={metadata.fileName}>
              {metadata.fileName}
            </h4>
            <span className="text-[11px] text-slate-secondary">
              {formatFileSize(metadata.fileSizeBytes)}
            </span>
          </div>
        </div>

        <button
          type="button"
          onClick={onClear}
          disabled={isProcessing}
          className="p-1 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded transition-colors disabled:opacity-50"
          title="Remove file"
          aria-label="Remove file"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Middle: Metrics grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 py-3 border-y border-slate-border text-xs">
        <div>
          <span className="text-slate-muted block text-[11px]">Rows</span>
          <span className="font-semibold text-slate-main text-sm font-mono">
            {metadata.rowCount.toLocaleString()}
          </span>
        </div>
        <div>
          <span className="text-slate-muted block text-[11px]">Columns</span>
          <span className="font-semibold text-slate-main text-sm font-mono">
            {metadata.columnCount.toLocaleString()}
          </span>
        </div>
        <div>
          <span className="text-slate-muted block text-[11px]">File Type</span>
          <span className="font-semibold text-slate-main text-sm font-mono">
            {metadata.fileType}
          </span>
        </div>
        <div>
          <span className="text-slate-muted block text-[11px]">Schema Validation</span>
          <div className="mt-0.5">
            {validationResult.isValid ? (
              <Badge variant="success" size="sm" className="gap-1">
                <CheckCircle2 className="w-3 h-3 text-status-success" />
                Valid
              </Badge>
            ) : (
              <Badge variant="danger" size="sm" className="gap-1">
                <AlertCircle className="w-3 h-3 text-status-danger" />
                Invalid
              </Badge>
            )}
          </div>
        </div>
      </div>

      {/* Action button */}
      <div className="pt-1 flex items-center justify-between gap-3">
        <span className="text-xs text-slate-secondary">
          {metadata.rowCount === 1
            ? '1 transaction ready for scoring'
            : `${metadata.rowCount.toLocaleString()} transactions ready for scoring`}
        </span>

        <Button
          type="button"
          variant="primary"
          size="md"
          disabled={!validationResult.isValid || isProcessing}
          onClick={onRunScoring}
          icon={
            isProcessing ? (
              <Loader2 className="w-4 h-4 animate-spin text-white" />
            ) : (
              <Play className="w-3.5 h-3.5 fill-current" />
            )
          }
        >
          {isProcessing ? 'Scoring...' : 'Run Fraud Detection'}
        </Button>
      </div>
    </Card>
  );
};

export default FileInformation;
