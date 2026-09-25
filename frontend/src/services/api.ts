import type {
  ApiBatchPredictionResponse,
  ApiDashboardResponse,
  ApiReadyResponse,
  ApiSchemaResponse,
  ApiVersionResponse,
} from '@/types/fraud';

const configuredBase = import.meta.env.VITE_API_URL?.replace(/\/$/, '') || '';
const baseUrl = import.meta.env.DEV ? '/api' : configuredBase;

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly detail: unknown,
    message: string
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function requestJson<T>(
  path: string,
  init: RequestInit = {},
  timeoutMs = 60_000
): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${baseUrl}${path}`, {
      ...init,
      signal: controller.signal,
      headers: {
        Accept: 'application/json',
        ...(init.body ? { 'Content-Type': 'application/json' } : {}),
        ...init.headers,
      },
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new ApiError(response.status, payload?.detail, `HTTP ${response.status}`);
    }
    return payload as T;
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new ApiError(408, null, 'Request timed out');
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}

export const getTransactionSchema = (): Promise<ApiSchemaResponse> =>
  requestJson('/schema/transactions');

export const predictTransactions = (
  rows: Record<string, unknown>[]
): Promise<ApiBatchPredictionResponse> =>
  requestJson('/predict/transactions', {
    method: 'POST',
    body: JSON.stringify({ rows }),
  });

export const getReadyStatus = (): Promise<ApiReadyResponse> =>
  requestJson('/ready', {}, 90_000);

export const getDashboard = (): Promise<ApiDashboardResponse> =>
  requestJson('/dashboard', {}, 90_000);

export const getVersion = (): Promise<ApiVersionResponse> =>
  requestJson('/version');
