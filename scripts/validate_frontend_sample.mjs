import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import Papa from 'papaparse';

const samplePath = resolve('src/constants/sample_transactions.csv');
const sampleCsv = readFileSync(samplePath, 'utf8');

const parsed = Papa.parse(sampleCsv, {
  header: true,
  skipEmptyLines: true,
  dynamicTyping: true,
  transformHeader: (header) => header.trim().replace(/^\uFEFF/, ''),
});

if (parsed.errors.length > 0) {
  throw new Error(`Sample CSV parse failed: ${parsed.errors[0].message}`);
}

if (parsed.data.length < 1000) {
  throw new Error(`Expected at least 1000 sample rows, found ${parsed.data.length}`);
}

const requiredHeaders = [
  'TransactionDT',
  'TransactionAmt',
  'ProductCD',
  'card1',
  'card2',
  'card3',
];

const firstRow = parsed.data[0] ?? {};
const missing = requiredHeaders.filter((header) => !(header in firstRow));

if (missing.length > 0) {
  throw new Error(`Sample CSV missing required headers: ${missing.join(', ')}`);
}

console.log(`Sample CSV parsed successfully: ${parsed.data.length} rows`);
