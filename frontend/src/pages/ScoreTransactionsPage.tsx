import React, { useState } from 'react';
import {
  Download,
  Upload,
  ClipboardList,
  AlertCircle,
  Loader2,
  RefreshCw,
  WifiOff,
} from 'lucide-react';
import { PageHeader, Button, Card } from '@/components/ui';
import {
  FileUpload,
  PasteData,
  FileInformation,
  SchemaValidation,
  TransactionPreview,
  ProcessingSummary,
  PredictionResultsTable,
  type FileMetadata,
  type SchemaValidationResult,
} from '@/components/scoring';
import { parseTransactionCsv } from '@/constants/transactionSchema';
import { predictTransactions } from '@/services/api';
import { parseApiError } from '@/utils/apiError';
import { getRiskLevel } from '@/utils/riskLevel';
import { useTransactionSchema } from '@/hooks/useTransactionSchema';
import type {
  PredictionResultRow,
  ScoringBatchSummary,
  ApiBatchPredictionResponse,
} from '@/types/fraud';

type InputMode = 'upload' | 'paste';

interface ScoringState {
  summary: ScoringBatchSummary;
  results: PredictionResultRow[];
  apiResponse: ApiBatchPredictionResponse;
}

export const ScoreTransactionsPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<InputMode>('upload');

  // Schema from backend
  const schemaState = useTransactionSchema();
  const maxBatchRows =
    schemaState.status === 'ready' ? schemaState.schema.max_batch_rows : null;

  // Parsed dataset state
  const [fileMetadata, setFileMetadata] = useState<FileMetadata | null>(null);
  const [parsedRows, setParsedRows] = useState<Record<string, unknown>[]>([]);
  const [validationResult, setValidationResult] = useState<SchemaValidationResult | null>(null);
  const [parseError, setParseError] = useState<string | null>(null);

  // Scoring execution state
  const [isProcessing, setIsProcessing] = useState(false);
  const [scoringState, setScoringState] = useState<ScoringState | null>(null);
  const [scoringError, setScoringError] = useState<string | null>(null);

  // ── Schema-aware column validation ──────────────────────────────────────────
  const validateHeaders = (columns: string[]): SchemaValidationResult => {
    // Use real backend schema when available; fallback to empty (blocks scoring)
    const requiredFields =
      schemaState.status === 'ready' ? schemaState.schema.feature_names : [];

    const presentFields = columns.filter((col) => requiredFields.includes(col));
    const missingFields = requiredFields.filter((req) => !columns.includes(req));
    const unexpectedFields = columns.filter((col) => !requiredFields.includes(col));

    return {
      isValid: requiredFields.length > 0 && missingFields.length === 0,
      presentFields,
      missingFields,
      unexpectedFields,
    };
  };

  const readCsvPreviewText = async (
    file: File,
    rowLimit: number
  ): Promise<string> => {
    const requiredLineCount = rowLimit + 1; // header + data rows
    let chunkSize = Math.min(file.size, 512 * 1024);
    let text = '';

    while (chunkSize <= file.size) {
      text = await file.slice(0, chunkSize).text();
      const lineCount = text.split(/\r\n|\n|\r/).filter(Boolean).length;
      if (lineCount >= requiredLineCount || chunkSize === file.size) {
        break;
      }
      chunkSize = Math.min(file.size, chunkSize * 2);
    }

    const lines = text.split(/\r\n|\n|\r/).filter(Boolean);
    return lines.slice(0, requiredLineCount).join('\n');
  };

  // ── File parsing helpers ────────────────────────────────────────────────────
  const applyParsedData = (
    rows: Record<string, unknown>[],
    fileName: string,
    fileSizeBytes: number,
    fileType: 'CSV' | 'JSON'
  ) => {
    const columns = Object.keys(rows[0] || {});
    const vResult = validateHeaders(columns);
    setFileMetadata({
      fileName,
      fileSizeBytes,
      rowCount: rows.length,
      columnCount: columns.length,
      fileType,
    });
    setParsedRows(rows);
    setValidationResult(vResult);
    setScoringState(null);
    setScoringError(null);
  };

  // ── Handle File Upload ──────────────────────────────────────────────────────
  const handleFileSelected = async (file: File) => {
    setParseError(null);
    setScoringState(null);

    const ext = file.name.split('.').pop()?.toLowerCase();

    if (ext === 'csv') {
      try {
        const csvPreview = await readCsvPreviewText(file, maxBatchRows ?? 100);
        const results = parseTransactionCsv(csvPreview);
        if (results.errors.length > 0 && results.rows.length === 0) {
          setParseError(`Failed to parse CSV: ${results.errors[0]}`);
          return;
        }
        if (results.rows.length === 0) {
          setParseError(
            'The uploaded CSV file has no readable data rows in the first batch.'
          );
          return;
        }
        applyParsedData(results.rows, file.name, file.size, 'CSV');
      } catch (err) {
        setParseError(`Error reading CSV: ${(err as Error).message}`);
      }
    } else if (ext === 'json') {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const parsed = JSON.parse(e.target?.result as string);
          const rows: Record<string, unknown>[] = Array.isArray(parsed) ? parsed : [parsed];
          if (rows.length === 0) {
            setParseError('The JSON file contains no transaction records.');
            return;
          }
          applyParsedData(rows, file.name, file.size, 'JSON');
        } catch (err) {
          setParseError(`Invalid JSON: ${(err as Error).message}`);
        }
      };
      reader.readAsText(file);
    }
  };

  // ── Handle Pasted Data ──────────────────────────────────────────────────────
  const handleDataPasted = (rawText: string, format: 'json' | 'csv') => {
    setParseError(null);
    setScoringState(null);
    const sizeBytes = new Blob([rawText]).size;

    if (format === 'json') {
      try {
        const parsed = JSON.parse(rawText);
        const rows: Record<string, unknown>[] = Array.isArray(parsed) ? parsed : [parsed];
        if (rows.length === 0) {
          setParseError('No transaction records found in the pasted JSON.');
          return;
        }
        applyParsedData(rows, 'pasted_transactions.json', sizeBytes, 'JSON');
      } catch (err) {
        setParseError(`JSON parse failed: ${(err as Error).message}`);
      }
    } else {
      const results = parseTransactionCsv(rawText);
      if (results.rows.length === 0) {
        setParseError('No rows could be parsed from the pasted CSV content.');
        return;
      }
      applyParsedData(results.rows, 'pasted_transactions.csv', sizeBytes, 'CSV');
    }
  };

  // ── Clear / Reset ───────────────────────────────────────────────────────────
  const handleClear = () => {
    setFileMetadata(null);
    setParsedRows([]);
    setValidationResult(null);
    setParseError(null);
    setScoringState(null);
    setScoringError(null);
  };

  // ── Map raw API rows to normalized frontend rows ────────────────────────────
  const normaliseResults = (
    apiResp: ApiBatchPredictionResponse,
    sourceRows: Record<string, unknown>[]
  ): PredictionResultRow[] => {
    return apiResp.results.map((apiRow) => {
      const sourceRow = sourceRows[apiRow.row_index] ?? {};
      const rawAmt = sourceRow['TransactionAmt'] ?? sourceRow['amount'] ?? 0;
      const transactionAmt =
        typeof rawAmt === 'number' ? rawAmt : parseFloat(String(rawAmt)) || 0;

      const rawDt = sourceRow['TransactionDT'] ?? sourceRow['timestamp'] ?? '';

      return {
        transactionId: `TX-${String(apiRow.row_index + 1).padStart(6, '0')}`,
        rowIndex: apiRow.row_index,
        transactionDt: rawDt as string | number,
        transactionAmt: +transactionAmt.toFixed(2),
        fraudScore: apiRow.fraud_score,
        threshold: apiRow.threshold_used,
        // Map "Yes" → "Fraud", "No" → "Legit"
        decision: apiRow.fraud_status === 'Yes' ? 'Fraud' : 'Legit',
        riskLevel: getRiskLevel(apiRow.fraud_score),
      };
    });
  };

  // ── Build summary from real API response ────────────────────────────────────
  const buildSummary = (
    apiResp: ApiBatchPredictionResponse,
    results: PredictionResultRow[]
  ): ScoringBatchSummary => {
    const fraudCount = results.filter((r) => r.decision === 'Fraud').length;
    const legitCount = results.length - fraudCount;
    const totalScore = results.reduce((sum, r) => sum + r.fraudScore, 0);
    const avgScore = results.length > 0 ? totalScore / results.length : 0;

    return {
      totalTransactions: results.length,
      predictedLegit: legitCount,
      predictedFraud: fraudCount,
      fraudRate: results.length > 0
        ? +((fraudCount / results.length) * 100).toFixed(1)
        : 0,
      averageFraudScore: +avgScore.toFixed(4),
      processingDurationSeconds: +(apiResp.latency_ms / 1000).toFixed(3),
      modelVersion: apiResp.model_version,
      modelName: apiResp.model_name,
    };
  };

  // ── Run real fraud detection ────────────────────────────────────────────────
  const handleRunFraudDetection = async () => {
    if (!validationResult?.isValid || parsedRows.length === 0) return;
    if (schemaState.status !== 'ready') return;

    setIsProcessing(true);
    setScoringError(null);
    setScoringState(null);

    // Warn if over the batch limit (backend will reject >100 rows by default)
    const maxRows = schemaState.schema.max_batch_rows;
    const rowsToScore = parsedRows.slice(0, maxRows);

    try {
      const apiResp = await predictTransactions(rowsToScore);
      const results = normaliseResults(apiResp, rowsToScore);
      const summary = buildSummary(apiResp, results);
      setScoringState({ summary, results, apiResponse: apiResp });
    } catch (err) {
      setScoringError(parseApiError(err));
    } finally {
      setIsProcessing(false);
    }
  };

  // ── Download Sample CSV ─────────────────────────────────────────────────────
  const handleDownloadSampleCsv = () => {
    const a = document.createElement('a');
    a.href = '/sample_transactions.csv';
    a.download = 'transaction_sample.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  // ── Download Real Results CSV ───────────────────────────────────────────────
  const handleDownloadResultsCsv = () => {
    if (!scoringState || scoringState.results.length === 0) return;

    const headers = [
      'Transaction ID',
      'TransactionDT',
      'TransactionAmt',
      'Fraud Score',
      'Threshold',
      'Decision',
      'Risk Level',
    ];
    const rows = scoringState.results.map((r) => [
      r.transactionId,
      r.transactionDt,
      r.transactionAmt,
      r.fraudScore,
      r.threshold,
      r.decision,
      r.riskLevel,
    ]);

    const csv = [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'fraudguard_prediction_results.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // ── Max batch row warning ───────────────────────────────────────────────────
  const exceedsBatchLimit = maxBatchRows !== null && parsedRows.length > maxBatchRows;

  return (
    <div className="space-y-6">
      {/* 1. Page Header */}
      <PageHeader
        title="Score Transactions"
        description="Upload a transaction file or paste data to run fraud detection using the trained model."
        actions={
          <Button
            type="button"
            variant="secondary"
            size="sm"
            onClick={handleDownloadSampleCsv}
            icon={<Download className="w-3.5 h-3.5" />}
          >
            Download Sample CSV
          </Button>
        }
      />

      {/* 2. Schema Loading Status */}
      {schemaState.status === 'loading' && (
        <Card className="flex items-center gap-3 p-4 text-xs text-slate-secondary">
          <Loader2 className="w-4 h-4 animate-spin text-primary shrink-0" />
          <span>Loading model schema from prediction service…</span>
        </Card>
      )}

      {schemaState.status === 'error' && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg flex items-start justify-between gap-4 text-xs">
          <div className="flex items-start gap-2 text-status-danger">
            <WifiOff className="w-4 h-4 shrink-0 mt-0.5" />
            <div>
              <div className="font-semibold mb-0.5">Unable to Load Transaction Schema</div>
              <div className="text-red-700">{schemaState.message}</div>
            </div>
          </div>
          <Button
            type="button"
            variant="secondary"
            size="sm"
            onClick={schemaState.refetch}
            icon={<RefreshCw className="w-3.5 h-3.5" />}
          >
            Retry
          </Button>
        </div>
      )}



      {/* 3. Input Mode Tabs */}
      <div className="flex items-center gap-1 border-b border-slate-border pb-3">
        <button
          type="button"
          onClick={() => setActiveTab('upload')}
          className={`flex items-center gap-2 px-3.5 py-1.5 rounded-btn text-xs font-medium transition-colors ${
            activeTab === 'upload'
              ? 'bg-blue-600 text-white shadow-subtle'
              : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
          }`}
        >
          <Upload className="w-3.5 h-3.5" />
          Upload File
        </button>

        <button
          type="button"
          onClick={() => setActiveTab('paste')}
          className={`flex items-center gap-2 px-3.5 py-1.5 rounded-btn text-xs font-medium transition-colors ${
            activeTab === 'paste'
              ? 'bg-blue-600 text-white shadow-subtle'
              : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
          }`}
        >
          <ClipboardList className="w-3.5 h-3.5" />
          Paste Data
        </button>
      </div>

      {/* 4. Input & Information Area */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        <div className={fileMetadata ? 'lg:col-span-7' : 'lg:col-span-12'}>
          {activeTab === 'upload' ? (
            <FileUpload
              onFileSelected={handleFileSelected}
              disabled={isProcessing || schemaState.status !== 'ready'}
            />
          ) : (
            <PasteData
              onDataPasted={handleDataPasted}
              disabled={isProcessing || schemaState.status !== 'ready'}
            />
          )}

          {parseError && (
            <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg flex items-center gap-2 text-xs text-status-danger">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{parseError}</span>
            </div>
          )}
        </div>

        {fileMetadata && validationResult && (
          <div className="lg:col-span-5 space-y-4">
            {/* Batch limit warning */}
            {exceedsBatchLimit && (
              <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg text-xs text-amber-800 flex items-center gap-2">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>
                  File has {parsedRows.length.toLocaleString()} rows but the backend allows a maximum
                  of {maxBatchRows?.toLocaleString()} per request. Only the first{' '}
                  {maxBatchRows?.toLocaleString()} rows will be scored.
                </span>
              </div>
            )}

            <FileInformation
              metadata={{
                ...fileMetadata,
                rowCount: exceedsBatchLimit ? (maxBatchRows ?? fileMetadata.rowCount) : fileMetadata.rowCount,
              }}
              validationResult={validationResult}
              isProcessing={isProcessing}
              onRunScoring={handleRunFraudDetection}
              onClear={handleClear}
            />

            {!validationResult.isValid && (
              <SchemaValidation validationResult={validationResult} />
            )}
          </div>
        )}
      </section>

      {/* 5. Transaction Preview (shown when valid data loaded) */}
      {parsedRows.length > 0 && validationResult?.isValid && (
        <section aria-label="Transaction Preview">
          <TransactionPreview
            firstRow={parsedRows[0]}
            totalRows={Math.min(parsedRows.length, maxBatchRows ?? parsedRows.length)}
          />
        </section>
      )}

      {/* 6. Scoring Error */}
      {scoringError && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3 text-xs">
          <AlertCircle className="w-4 h-4 text-status-danger shrink-0 mt-0.5" />
          <div>
            <div className="font-semibold text-status-danger mb-0.5">Prediction Failed</div>
            <div className="text-red-700">{scoringError}</div>
          </div>
        </div>
      )}

      {/* 7. Real Scoring Results */}
      {scoringState && (
        <section aria-label="Scoring Results" className="space-y-6 pt-2">
          <ProcessingSummary
            summary={scoringState.summary}
            isSingleTransaction={scoringState.summary.totalTransactions === 1}
            singleScore={scoringState.results[0]?.fraudScore}
            threshold={scoringState.results[0]?.threshold}
          />

          <PredictionResultsTable
            results={scoringState.results}
            onDownloadCsv={handleDownloadResultsCsv}
          />
        </section>
      )}
    </div>
  );
};

export default ScoreTransactionsPage;
