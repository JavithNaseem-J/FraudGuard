import Papa from 'papaparse';
import sampleCsvRaw from './sample_transactions.csv?raw';

/**
 * Key preview fields shown in the transaction preview card
 */
export const PREVIEW_KEY_FIELDS: { key: string; label: string }[] = [
  { key: 'TransactionAmt', label: 'Transaction Amount' },
  { key: 'ProductCD', label: 'Product Code' },
  { key: 'card4', label: 'Card Network' },
  { key: 'card6', label: 'Card Type' },
  { key: 'P_emaildomain', label: 'Payer Email Domain' },
];

/**
 * Valid sample transactions CSV containing all transaction-only features expected by the model.
 */
export const SAMPLE_TRANSACTION_CSV: string = sampleCsvRaw.trim();

const DEFAULT_RANDOM_SAMPLE_ROWS = 10;

export interface ParsedTransactionCsv {
  rows: Record<string, unknown>[];
  errors: string[];
}

export function cleanCsvHeader(header: string): string {
  return header.trim().replace(/^\uFEFF/, '');
}

export function parseTransactionCsv(rawText: string): ParsedTransactionCsv {
  const results = Papa.parse<Record<string, unknown>>(rawText, {
    delimiter: ',',
    header: true,
    skipEmptyLines: true,
    dynamicTyping: true,
    transformHeader: cleanCsvHeader,
  });

  return {
    rows: results.data.filter((row) => Object.keys(row).length > 0),
    errors: results.errors.map((error) => error.message),
  };
}

function getSamplePoolLines(): { header: string; rows: string[] } {
  const lines = SAMPLE_TRANSACTION_CSV.split(/\r?\n/).filter(Boolean);
  return {
    header: lines[0] ?? '',
    rows: lines.slice(1),
  };
}

function getRandomRows(rowCount = DEFAULT_RANDOM_SAMPLE_ROWS): string[] {
  const { rows } = getSamplePoolLines();
  const count = Math.max(1, Math.min(rowCount, rows.length));
  const shuffled = [...rows];

  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    [shuffled[index], shuffled[swapIndex]] = [shuffled[swapIndex], shuffled[index]];
  }

  return shuffled.slice(0, count);
}

export function createRandomSampleCsv(rowCount = DEFAULT_RANDOM_SAMPLE_ROWS): string {
  const { header } = getSamplePoolLines();
  return [header, ...getRandomRows(rowCount)].join('\n');
}

export function createRandomSampleJson(rowCount = DEFAULT_RANDOM_SAMPLE_ROWS): string {
  const { header } = getSamplePoolLines();
  const csv = [header, ...getRandomRows(rowCount)].join('\n');
  const { rows } = parseTransactionCsv(csv);

  return JSON.stringify(rows, null, 2);
}
