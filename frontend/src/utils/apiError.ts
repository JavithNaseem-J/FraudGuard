import { ApiError } from '@/services/api';

export function parseApiError(error: unknown): string {
  if (!(error instanceof ApiError)) {
    return 'Prediction service is unavailable. A free-tier instance may be waking up; wait a moment and retry.';
  }
  const detail = typeof error.detail === 'string' ? error.detail : undefined;
  switch (error.status) {
    case 408:
      return 'The request timed out. The free-tier service may still be starting; retry shortly.';
    case 413:
      return 'The request is larger than the 1 MiB demo limit.';
    case 422:
      return detail
        ? `Prediction request rejected: ${detail}`
        : 'The payload does not match the deployed model schema.';
    case 429:
      return 'The public demo rate limit was reached. Wait one minute and retry.';
    case 503:
      return detail
        ? `Prediction service not ready: ${detail}`
        : 'The model is still loading. Retry after the service wakes up.';
    default:
      return `The request failed (HTTP ${error.status}). Please retry.`;
  }
}
