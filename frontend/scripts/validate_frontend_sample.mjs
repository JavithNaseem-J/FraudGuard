import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import Papa from 'papaparse';

const samplePath = resolve('public/sample_transactions.csv');
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

const apiSource = readFileSync(resolve('src/services/api.ts'), 'utf8');
const dashboardSource = readFileSync(resolve('src/services/dashboardStore.ts'), 'utf8');
const overviewSource = readFileSync(resolve('src/pages/OverviewPage.tsx'), 'utf8');
const statusSource = readFileSync(
  resolve('src/components/dashboard/SystemStatus.tsx'),
  'utf8'
);

for (const forbidden of ['axios', 'FRAUDGUARD_API_KEY', 'localStorage']) {
  const combined = [apiSource, dashboardSource, overviewSource, statusSource].join('\n');
  if (combined.includes(forbidden)) {
    throw new Error(`Frontend must not contain ${forbidden}`);
  }
}

for (const requiredEndpoint of [
  '/predict/transactions',
  '/schema/transactions',
  '/dashboard',
  '/ready',
]) {
  if (!apiSource.includes(requiredEndpoint)) {
    throw new Error(`Frontend API client is missing ${requiredEndpoint}`);
  }
}

if (!overviewSource.includes('onDoubleClick={clearBrowserView}')) {
  throw new Error('Dashboard Clear must require the confirmation gesture');
}
if (!overviewSource.includes('getDashboard()')) {
  throw new Error('Dashboard Refresh must load the server-backed snapshot');
}
if (statusSource.includes('All Systems Operational')) {
  throw new Error('System status must not be hard-coded');
}

console.log('Frontend API, refresh, clear, and status contracts validated');
