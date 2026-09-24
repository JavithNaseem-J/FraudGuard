import { type AxiosError } from 'axios';

/**
 * Convert an AxiosError into a user-friendly error message.
 * All API error mapping is centralised here — never display raw AxiosError to users.
 */
export function parseApiError(error: unknown): string {
  const axiosErr = error as AxiosError<{ detail?: string | Record<string, unknown> }>;

  if (!axiosErr.response) {
    // Network error or timeout — backend not reachable
    if (axiosErr.code === 'ECONNABORTED') {
      return 'Request timed out. The prediction service may be overloaded. Please try again.';
    }
    return 'Prediction service unavailable. Check that the FastAPI backend is running at the configured URL.';
  }

  const status = axiosErr.response.status;
  const detail = axiosErr.response.data?.detail;
  const detailMessage = typeof detail === 'string' ? detail : undefined;

  switch (status) {
    case 400:
      return detailMessage
        ? `Invalid request: ${detailMessage}`
        : 'The request was rejected by the server. Check the transaction payload format.';
    case 401:
      return 'Authentication is required by the backend. For local frontend testing, run the FastAPI service with AUTH_REQUIRED=false or add a proper server-side auth/session layer.';
    case 403:
      return 'Access denied by the backend. Do not put server API keys in browser environment variables.';
    case 413:
      return 'Request payload is too large. Reduce the number of rows and try again.';
    case 422:
      return detailMessage
        ? `Prediction request rejected: ${detailMessage}`
        : 'The transaction payload does not match the model schema. Verify that all required features are present.';
    case 429:
      return 'Rate limit exceeded. Please wait before submitting another request.';
    case 500:
      return 'Internal server error. The prediction service encountered an unexpected error.';
    case 503:
      return detailMessage
        ? `Prediction service not ready: ${detailMessage}`
        : 'Prediction service is not ready. The model may still be initialising.';
    default:
      return `Unexpected error (HTTP ${status}). Please try again.`;
  }
}
