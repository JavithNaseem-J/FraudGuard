import Papa from 'papaparse';

const SAMPLE_TRANSACTION_URL = '/sample_transactions.csv';
let sampleTransactionCsvPromise: Promise<string> | null = null;

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

export function loadSampleTransactionCsv(): Promise<string> {
  if (!sampleTransactionCsvPromise) {
    sampleTransactionCsvPromise = fetch(SAMPLE_TRANSACTION_URL).then(async (response) => {
      if (!response.ok) {
        throw new Error(`Unable to load sample transaction data: HTTP ${response.status}`);
      }
      return (await response.text()).trim();
    });
  }
  return sampleTransactionCsvPromise;
}

function getSamplePoolLines(sampleCsv: string): { header: string; rows: string[] } {
  const lines = sampleCsv.split(/\r?\n/).filter(Boolean);
  return {
    header: lines[0] ?? '',
    rows: lines.slice(1),
  };
}

function getRandomRows(sampleCsv: string, rowCount = DEFAULT_RANDOM_SAMPLE_ROWS): string[] {
  const { rows } = getSamplePoolLines(sampleCsv);
  const count = Math.max(1, Math.min(rowCount, rows.length));
  const shuffled = [...rows];

  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    [shuffled[index], shuffled[swapIndex]] = [shuffled[swapIndex], shuffled[index]];
  }

  return shuffled.slice(0, count);
}

export async function createRandomSampleCsv(
  rowCount = DEFAULT_RANDOM_SAMPLE_ROWS
): Promise<string> {
  const sampleCsv = await loadSampleTransactionCsv();
  const { header } = getSamplePoolLines(sampleCsv);
  return [header, ...getRandomRows(sampleCsv, rowCount)].join('\n');
}

export async function createRandomSampleJson(
  rowCount = DEFAULT_RANDOM_SAMPLE_ROWS
): Promise<string> {
  const sampleCsv = await loadSampleTransactionCsv();
  const { header } = getSamplePoolLines(sampleCsv);
  const csv = [header, ...getRandomRows(sampleCsv, rowCount)].join('\n');
  const { rows } = parseTransactionCsv(csv);

  return JSON.stringify(rows, null, 2);
}
