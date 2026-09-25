import { useState, useEffect, useCallback } from 'react';
import { getTransactionSchema } from '@/services/api';
import { parseApiError } from '@/utils/apiError';
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
    } catch (error) {
      setState({ status: 'error', message: parseApiError(error) });
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return { ...state, refetch: load };
}
