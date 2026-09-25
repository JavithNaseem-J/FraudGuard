import React, { useState } from 'react';
import { Button } from '@/components/ui';
import {
  createRandomSampleCsv,
  createRandomSampleJson,
} from '@/constants/transactionSchema';
import { Play, RotateCcw } from 'lucide-react';

export interface PasteDataProps {
  onDataPasted: (rawText: string, detectedFormat: 'json' | 'csv') => void;
  disabled?: boolean;
}

export const PasteData: React.FC<PasteDataProps> = ({ onDataPasted, disabled = false }) => {
  const [text, setText] = useState('');
  const [error, setError] = useState<string | null>(null);

  const handleValidate = () => {
    const trimmed = text.trim();
    if (!trimmed) {
      setError('Please paste CSV or JSON transaction records before validating.');
      return;
    }

    // Detect format
    if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
      try {
        JSON.parse(trimmed);
        setError(null);
        onDataPasted(trimmed, 'json');
      } catch (err) {
        setError(`Invalid JSON format: ${(err as Error).message}`);
      }
    } else {
      // Treat as CSV if it has comma-separated lines
      if (!trimmed.includes(',')) {
        setError('Pasted content does not appear to be valid comma-separated values (CSV) or JSON.');
        return;
      }
      setError(null);
      onDataPasted(trimmed, 'csv');
    }
  };

  const handleFillSample = async (format: 'json' | 'csv') => {
    try {
      const sample =
        format === 'json'
          ? await createRandomSampleJson()
          : await createRandomSampleCsv();
      setText(sample);
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  return (
    <div className="w-full space-y-3">
      <div className="flex items-center justify-between text-xs">
        <span className="font-medium text-slate-secondary">
          Paste transaction payload (CSV with headers or JSON array/object):
        </span>
        <div className="flex items-center gap-2">
          <button
            type="button"
            disabled={disabled}
            onClick={() => handleFillSample('json')}
            className="text-primary hover:underline font-medium text-xs disabled:opacity-50"
          >
            Load Sample JSON
          </button>
          <span className="text-slate-border">&bull;</span>
          <button
            type="button"
            disabled={disabled}
            onClick={() => handleFillSample('csv')}
            className="text-primary hover:underline font-medium text-xs disabled:opacity-50"
          >
            Load Sample CSV
          </button>
        </div>
      </div>

      <textarea
        value={text}
        onChange={(e) => {
          setText(e.target.value);
          if (error) setError(null);
        }}
        disabled={disabled}
        placeholder={`{\n  "TransactionDT": 86400,\n  "TransactionAmt": 57.25,\n  "card1": 13926,\n  "card2": 111,\n  "card3": 150,\n  "card4": "discover",\n  "card5": 142,\n  "card6": "credit",\n  "addr1": 315,\n  "addr2": 87,\n  "ProductCD": "W",\n  "P_emaildomain": "gmail.com",\n  "DeviceType": "desktop"\n}`}
        rows={8}
        className="w-full font-mono text-xs p-3.5 bg-white border border-slate-border rounded-input text-slate-main focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary disabled:bg-slate-50 transition-colors placeholder:text-slate-muted"
      />

      {error && (
        <div className="p-3 bg-red-50 border border-red-200 rounded-input text-xs text-status-danger font-medium">
          {error}
        </div>
      )}

      <div className="flex items-center justify-between pt-1">
        <Button
          type="button"
          variant="ghost"
          size="sm"
          disabled={disabled || !text}
          onClick={() => {
            setText('');
            setError(null);
          }}
          icon={<RotateCcw className="w-3.5 h-3.5" />}
        >
          Clear
        </Button>

        <Button
          type="button"
          variant="primary"
          size="sm"
          disabled={disabled || !text.trim()}
          onClick={handleValidate}
          icon={<Play className="w-3.5 h-3.5" />}
        >
          Validate Data
        </Button>
      </div>
    </div>
  );
};

export default PasteData;
