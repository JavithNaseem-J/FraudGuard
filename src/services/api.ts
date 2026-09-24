import axios, { type AxiosInstance, type AxiosError } from 'axios';
import type {
  ApiSchemaResponse,
  ApiReadyResponse,
  ApiVersionResponse,
  ApiBatchPredictionResponse,
} from '@/types/fraud';

// ─── Shared Axios instance ────────────────────────────────────────────────────
// In development, requests go to /api/* which Vite proxies to localhost:8000,
// solving CORS without touching the backend.
// In production, leave VITE_API_URL empty when the frontend and API share an
// origin, or set it to the deployed API origin when they are hosted separately.

const configuredBase = import.meta.env.VITE_API_URL?.replace(/\/$/, '') || '';

// In dev we use the Vite proxy at /api; in production use the configured API
// origin, or same-origin relative URLs when VITE_API_URL is not set.
const resolvedBase = import.meta.env.DEV ? '/api' : configuredBase;

export const apiClient: AxiosInstance = axios.create({
  baseURL: resolvedBase,
  timeout: 60_000,
  headers: {
    'Content-Type': 'application/json',
    Accept: 'application/json',
  },
});

// ─── Request interceptor ──────────────────────────────────────────────────────
apiClient.interceptors.request.use(
  (config) => config,
  (error) => Promise.reject(error)
);

// ─── Response interceptor — unified error logging ─────────────────────────────
apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    if (import.meta.env.DEV) {
      console.error('[FraudShield API Error]', {
        url: error.config?.url,
        status: error.response?.status,
        data: error.response?.data,
        message: error.message,
      });
    }
    return Promise.reject(error);
  }
);

// ─── Typed API functions ───────────────────────────────────────────────────────

/**
 * GET /schema/transactions
 * Returns the full model schema including feature names, threshold, and batch limits.
 */
export async function getTransactionSchema(): Promise<ApiSchemaResponse> {
  const response = await apiClient.get<ApiSchemaResponse>('/schema/transactions');
  return response.data;
}

/**
 * POST /predict/transactions
 * Submits a batch of transaction rows for fraud scoring.
 *
 * @param rows - Array of raw transaction row objects (field names must match feature_names).
 */
export async function predictTransactions(
  rows: Record<string, unknown>[]
): Promise<ApiBatchPredictionResponse> {
  const response = await apiClient.post<ApiBatchPredictionResponse>(
    '/predict/transactions',
    { rows }
  );
  return response.data;
}

/**
 * GET /ready
 * Check whether the prediction service is ready to accept requests.
 */
export async function getReadyStatus(): Promise<ApiReadyResponse> {
  const response = await apiClient.get<ApiReadyResponse>('/ready');
  return response.data;
}

/**
 * GET /version
 * Returns the API build commit SHA and build timestamp.
 */
export async function getVersion(): Promise<ApiVersionResponse> {
  const response = await apiClient.get<ApiVersionResponse>('/version');
  return response.data;
}

export default apiClient;
