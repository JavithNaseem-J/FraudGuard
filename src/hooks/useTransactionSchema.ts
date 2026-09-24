import { useState, useEffect, useCallback } from 'react';
import { type AxiosError } from 'axios';
import { getTransactionSchema } from '@/services/api';
import type { ApiSchemaResponse } from '@/types/fraud';

export type SchemaLoadState =
  | { status: 'loading' }
  | { status: 'ready'; schema: ApiSchemaResponse }
  | { status: 'error'; message: string };

export function useTransactionSchema(): SchemaLoadState & { refetch: () => void } {
  const [state, setState] = useState<SchemaLoadState>({ status: 'loading' });

  const load = useCallback(async () => {
    setState({ status: 'loading' });
    try {
      const schema = await getTransactionSchema();
      setState({ status: 'ready', schema });
    } catch (err) {
      const axiosErr = err as AxiosError<{ detail?: string }>;
      let message = 'Unable to load transaction schema from the prediction service.';

      if (!axiosErr.response) {
        message =
          'Prediction service is unreachable. Make sure the FastAPI backend is running at the configured URL.';
      } else if (axiosErr.response.status === 503) {
        const detail = axiosErr.response.data?.detail;
        message =
          typeof detail === 'string'
            ? `Prediction service not ready: ${detail}`
            : 'Prediction service is not ready. The model may still be loading.';
      } else {
        message = `Schema load failed (HTTP ${axiosErr.response.status}).`;
      }

      setState({ status: 'error', message });
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return { ...state, refetch: load };
}
