/**
 * Base API client for WeaverTools web-ui.
 * Provides common utilities for making HTTP requests to the Weaver backend.
 */

/** Default API base URL - can be overridden with environment variable */
const API_BASE_URL = import.meta.env.VITE_API_URL || '';

/** API error class with status code and body */
export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly statusText: string,
    public readonly body?: unknown
  ) {
    super(message);
    this.name = 'ApiError';
  }

  /** Check if error is a client error (4xx) */
  isClientError(): boolean {
    return this.status >= 400 && this.status < 500;
  }

  /** Check if error is a server error (5xx) */
  isServerError(): boolean {
    return this.status >= 500;
  }

  /** Check if error is a not found error (404) */
  isNotFound(): boolean {
    return this.status === 404;
  }

  /** Check if error is an unauthorized error (401) */
  isUnauthorized(): boolean {
    return this.status === 401;
  }

  /** Check if error is a validation error (400 or 422) */
  isValidationError(): boolean {
    return this.status === 400 || this.status === 422;
  }
}

/** Request options for API calls */
export interface ApiRequestOptions {
  /** Request headers */
  headers?: Record<string, string>;
  /** Request timeout in milliseconds */
  timeout?: number;
  /** AbortSignal for cancellation */
  signal?: AbortSignal;
}

/** Response type with typed body */
export interface ApiResponse<T> {
  data: T;
  status: number;
  headers: Headers;
}

/**
 * Create a timeout-aware AbortController.
 * Combines user-provided signal with timeout.
 */
function createTimeoutController(
  timeout?: number,
  userSignal?: AbortSignal
): { controller: AbortController; cleanup: () => void } {
  const controller = new AbortController();
  let timeoutId: ReturnType<typeof setTimeout> | null = null;

  // Abort on timeout
  if (timeout) {
    timeoutId = setTimeout(() => {
      controller.abort(new Error(`Request timeout after ${timeout}ms`));
    }, timeout);
  }

  // Abort on user signal
  if (userSignal) {
    if (userSignal.aborted) {
      controller.abort(userSignal.reason);
    } else {
      userSignal.addEventListener('abort', () => {
        controller.abort(userSignal.reason);
      });
    }
  }

  const cleanup = () => {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
  };

  return { controller, cleanup };
}

/**
 * Make an HTTP request to the API.
 * @param method - HTTP method
 * @param path - API path (without base URL)
 * @param body - Request body (will be JSON serialized)
 * @param options - Additional request options
 * @returns Promise resolving to typed response
 */
async function request<T>(
  method: string,
  path: string,
  body?: unknown,
  options: ApiRequestOptions = {}
): Promise<ApiResponse<T>> {
  const url = `${API_BASE_URL}${path}`;

  const { controller, cleanup } = createTimeoutController(
    options.timeout ?? 30000,
    options.signal
  );

  try {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      Accept: 'application/json',
      ...options.headers,
    };

    const fetchOptions: RequestInit = {
      method,
      headers,
      signal: controller.signal,
    };

    if (body !== undefined) {
      fetchOptions.body = JSON.stringify(body);
    }

    const response = await fetch(url, fetchOptions);

    // Parse response body
    let data: T;
    const contentType = response.headers.get('content-type');

    if (contentType?.includes('application/json')) {
      data = (await response.json()) as T;
    } else if (contentType?.includes('text/')) {
      data = (await response.text()) as unknown as T;
    } else {
      // For binary responses, return blob as data
      data = (await response.blob()) as unknown as T;
    }

    // Handle error responses
    if (!response.ok) {
      throw new ApiError(
        `API request failed: ${response.status} ${response.statusText}`,
        response.status,
        response.statusText,
        data
      );
    }

    return {
      data,
      status: response.status,
      headers: response.headers,
    };
  } finally {
    cleanup();
  }
}

/** HTTP GET request */
export async function get<T>(
  path: string,
  options?: ApiRequestOptions
): Promise<ApiResponse<T>> {
  return request<T>('GET', path, undefined, options);
}

/** HTTP POST request */
export async function post<T>(
  path: string,
  body?: unknown,
  options?: ApiRequestOptions
): Promise<ApiResponse<T>> {
  return request<T>('POST', path, body, options);
}

/** HTTP PUT request */
export async function put<T>(
  path: string,
  body?: unknown,
  options?: ApiRequestOptions
): Promise<ApiResponse<T>> {
  return request<T>('PUT', path, body, options);
}

/** HTTP PATCH request */
export async function patch<T>(
  path: string,
  body?: unknown,
  options?: ApiRequestOptions
): Promise<ApiResponse<T>> {
  return request<T>('PATCH', path, body, options);
}

/** HTTP DELETE request */
export async function del<T>(
  path: string,
  options?: ApiRequestOptions
): Promise<ApiResponse<T>> {
  return request<T>('DELETE', path, undefined, options);
}

/**
 * Build a query string from an object of parameters.
 * Handles arrays, null/undefined, and special characters.
 */
export function buildQueryString(
  params: Record<string, string | number | boolean | string[] | undefined | null>
): string {
  const searchParams = new URLSearchParams();

  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null) {
      continue;
    }

    if (Array.isArray(value)) {
      for (const item of value) {
        searchParams.append(key, item);
      }
    } else {
      searchParams.append(key, String(value));
    }
  }

  const queryString = searchParams.toString();
  return queryString ? `?${queryString}` : '';
}

/**
 * Streaming response handler for Server-Sent Events.
 * @param path - API path for SSE endpoint
 * @param onMessage - Callback for each message
 * @param onError - Callback for errors
 * @param options - Request options
 * @returns Cleanup function to close the connection
 */
export function streamEvents<T>(
  path: string,
  onMessage: (data: T) => void,
  onError?: (error: Error) => void,
  options?: ApiRequestOptions
): () => void {
  const url = `${API_BASE_URL}${path}`;
  const eventSource = new EventSource(url);

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data) as T;
      onMessage(data);
    } catch (error) {
      onError?.(error instanceof Error ? error : new Error(String(error)));
    }
  };

  eventSource.onerror = () => {
    onError?.(new Error('EventSource connection failed'));
  };

  // Abort on user signal
  if (options?.signal) {
    options.signal.addEventListener('abort', () => {
      eventSource.close();
    });
  }

  // Return cleanup function
  return () => {
    eventSource.close();
  };
}

/**
 * Fetch with retry logic.
 * Retries on network errors and 5xx responses.
 * @param fn - Function to retry
 * @param retries - Maximum number of retries
 * @param delay - Delay between retries in ms (doubles each retry)
 */
export async function withRetry<T>(
  fn: () => Promise<T>,
  retries = 3,
  delay = 1000
): Promise<T> {
  let lastError: Error | undefined;

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      // Don't retry on client errors (4xx)
      if (error instanceof ApiError && error.isClientError()) {
        throw error;
      }

      // Don't retry on last attempt
      if (attempt === retries) {
        break;
      }

      // Exponential backoff
      await new Promise((resolve) => setTimeout(resolve, delay * Math.pow(2, attempt)));
    }
  }

  throw lastError ?? new Error('Request failed after retries');
}

/**
 * Download a file from an API response.
 * @param response - API response containing blob data
 * @param filename - Name for the downloaded file
 */
export function downloadFile(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/** API client instance for convenience */
export const api = {
  get,
  post,
  put,
  patch,
  del,
  buildQueryString,
  streamEvents,
  withRetry,
  downloadFile,
};

export default api;
