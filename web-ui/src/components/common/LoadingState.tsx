/**
 * LoadingState component - comprehensive loading state displays.
 *
 * Provides various loading state components for different UI contexts,
 * including full-page loading, data fetching states, empty states,
 * and retry mechanisms.
 */
import type { ReactNode } from 'react';
import { Spinner } from './Spinner';
import { Button } from './Button';
import { Alert } from './Alert';

/**
 * Data loading state type.
 */
export type DataState = 'idle' | 'loading' | 'success' | 'error';

/**
 * Props for LoadingState component.
 */
export interface LoadingStateProps {
  /** Current loading state */
  loading: boolean;
  /** Error message if any */
  error?: string | null;
  /** Whether data is empty/not found */
  empty?: boolean;
  /** Content to render when loaded */
  children: ReactNode;
  /** Loading message */
  loadingMessage?: string;
  /** Empty state message */
  emptyMessage?: string;
  /** Empty state title */
  emptyTitle?: string;
  /** Empty state icon */
  emptyIcon?: ReactNode;
  /** Empty state action */
  emptyAction?: ReactNode;
  /** Error retry callback */
  onRetry?: () => void;
  /** Additional CSS classes */
  className?: string;
  /** Minimum height when loading/empty */
  minHeight?: string | number;
}

/**
 * Default empty state icon.
 */
const DefaultEmptyIcon: React.FC = () => (
  <svg
    className="h-12 w-12 text-gray-400"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.5"
    aria-hidden="true"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M20.25 7.5l-.625 10.632a2.25 2.25 0 01-2.247 2.118H6.622a2.25 2.25 0 01-2.247-2.118L3.75 7.5M10 11.25h4M3.375 7.5h17.25c.621 0 1.125-.504 1.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125H3.375c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125z"
    />
  </svg>
);

/**
 * Unified loading/error/empty state handler.
 *
 * @example
 * ```tsx
 * // Basic usage
 * <LoadingState loading={isLoading} error={error}>
 *   <DataDisplay data={data} />
 * </LoadingState>
 *
 * // With empty state
 * <LoadingState
 *   loading={isLoading}
 *   error={error}
 *   empty={data.length === 0}
 *   emptyTitle="No sessions"
 *   emptyMessage="Create a new session to get started."
 *   emptyAction={<Button onClick={createSession}>New Session</Button>}
 * >
 *   <SessionList sessions={data} />
 * </LoadingState>
 *
 * // With retry
 * <LoadingState
 *   loading={isLoading}
 *   error={error}
 *   onRetry={refetch}
 * >
 *   <DataContent />
 * </LoadingState>
 * ```
 */
export const LoadingState: React.FC<LoadingStateProps> = ({
  loading,
  error,
  empty = false,
  children,
  loadingMessage = 'Loading...',
  emptyMessage = 'No data available.',
  emptyTitle = 'No data',
  emptyIcon,
  emptyAction,
  onRetry,
  className = '',
  minHeight = 200,
}) => {
  const minHeightStyle =
    typeof minHeight === 'number' ? `${minHeight}px` : minHeight;

  // Loading state
  if (loading) {
    return (
      <div
        className={`flex flex-col items-center justify-center ${className}`}
        style={{ minHeight: minHeightStyle }}
        role="status"
        aria-label="Loading"
      >
        <Spinner size="lg" label={loadingMessage} />
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div
        className={`flex items-center justify-center p-4 ${className}`}
        style={{ minHeight: minHeightStyle }}
        role="alert"
      >
        <div className="max-w-md w-full">
          <Alert
            variant="error"
            title="Error loading data"
            actions={
              onRetry && (
                <Button variant="outline" size="sm" onClick={onRetry}>
                  Try Again
                </Button>
              )
            }
          >
            {error}
          </Alert>
        </div>
      </div>
    );
  }

  // Empty state
  if (empty) {
    return (
      <div
        className={`flex flex-col items-center justify-center text-center p-6 ${className}`}
        style={{ minHeight: minHeightStyle }}
      >
        {emptyIcon ?? <DefaultEmptyIcon />}
        <h3 className="mt-4 text-lg font-medium text-gray-900">{emptyTitle}</h3>
        <p className="mt-1 text-sm text-gray-500">{emptyMessage}</p>
        {emptyAction && <div className="mt-4">{emptyAction}</div>}
      </div>
    );
  }

  // Content state
  return <>{children}</>;
};

/**
 * Props for PageLoadingState component.
 */
export interface PageLoadingStateProps {
  /** Whether the page is loading */
  loading: boolean;
  /** Error message if any */
  error?: string | null;
  /** Page content */
  children: ReactNode;
  /** Loading message */
  loadingMessage?: string;
  /** Page title for errors */
  pageTitle?: string;
  /** Error retry callback */
  onRetry?: () => void;
  /** Navigate home callback */
  onNavigateHome?: () => void;
}

/**
 * Full-page loading state for route transitions.
 *
 * @example
 * ```tsx
 * function DashboardPage() {
 *   const { data, loading, error, refetch } = useDashboardData();
 *
 *   return (
 *     <PageLoadingState
 *       loading={loading}
 *       error={error}
 *       onRetry={refetch}
 *       pageTitle="Dashboard"
 *     >
 *       <Dashboard data={data} />
 *     </PageLoadingState>
 *   );
 * }
 * ```
 */
export const PageLoadingState: React.FC<PageLoadingStateProps> = ({
  loading,
  error,
  children,
  loadingMessage = 'Loading page...',
  pageTitle = 'Page',
  onRetry,
  onNavigateHome,
}) => {
  // Loading state
  if (loading) {
    return (
      <div
        className="min-h-[60vh] flex flex-col items-center justify-center"
        role="status"
        aria-label="Loading page"
      >
        <Spinner size="xl" />
        <p className="mt-4 text-gray-600">{loadingMessage}</p>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="min-h-[60vh] flex items-center justify-center p-8">
        <div className="max-w-md w-full">
          <Alert
            variant="error"
            title={`Failed to load ${pageTitle}`}
            actions={
              <div className="flex gap-2">
                {onRetry && (
                  <Button variant="outline" size="sm" onClick={onRetry}>
                    Try Again
                  </Button>
                )}
                {onNavigateHome && (
                  <Button variant="secondary" size="sm" onClick={onNavigateHome}>
                    Go Home
                  </Button>
                )}
              </div>
            }
          >
            {error}
          </Alert>
        </div>
      </div>
    );
  }

  return <>{children}</>;
};

/**
 * Props for DataFetchState component.
 */
export interface DataFetchStateProps<T> {
  /** Data from fetch */
  data: T | undefined | null;
  /** Loading state */
  loading: boolean;
  /** Error message */
  error?: string | null;
  /** Render function for data */
  children: (data: T) => ReactNode;
  /** Check if data is empty (default: checks array length) */
  isEmpty?: (data: T) => boolean;
  /** Loading placeholder */
  loadingPlaceholder?: ReactNode;
  /** Empty state props */
  emptyProps?: {
    title?: string;
    message?: string;
    icon?: ReactNode;
    action?: ReactNode;
  };
  /** Retry callback */
  onRetry?: () => void;
}

/**
 * Type-safe data fetch state handler.
 *
 * @example
 * ```tsx
 * const { data, loading, error, refetch } = useQuery('sessions', fetchSessions);
 *
 * return (
 *   <DataFetchState
 *     data={data}
 *     loading={loading}
 *     error={error}
 *     onRetry={refetch}
 *     isEmpty={(sessions) => sessions.length === 0}
 *     emptyProps={{
 *       title: 'No sessions',
 *       message: 'Create your first session.',
 *       action: <Button onClick={create}>Create Session</Button>,
 *     }}
 *   >
 *     {(sessions) => <SessionList sessions={sessions} />}
 *   </DataFetchState>
 * );
 * ```
 */
export function DataFetchState<T>({
  data,
  loading,
  error,
  children,
  isEmpty,
  loadingPlaceholder,
  emptyProps,
  onRetry,
}: DataFetchStateProps<T>): React.ReactElement | null {
  // Loading
  if (loading) {
    if (loadingPlaceholder) {
      return <>{loadingPlaceholder}</>;
    }
    return (
      <div className="flex items-center justify-center min-h-[200px]">
        <Spinner size="lg" label="Loading..." />
      </div>
    );
  }

  // Error
  if (error) {
    return (
      <div className="flex items-center justify-center p-4 min-h-[200px]">
        <Alert
          variant="error"
          title="Failed to load data"
          actions={
            onRetry && (
              <Button variant="outline" size="sm" onClick={onRetry}>
                Retry
              </Button>
            )
          }
        >
          {error}
        </Alert>
      </div>
    );
  }

  // No data
  if (data === undefined || data === null) {
    return (
      <div className="flex flex-col items-center justify-center p-6 min-h-[200px] text-center">
        <DefaultEmptyIcon />
        <h3 className="mt-4 text-lg font-medium text-gray-900">No data</h3>
        <p className="mt-1 text-sm text-gray-500">Data not available.</p>
      </div>
    );
  }

  // Check if empty
  const dataIsEmpty = isEmpty
    ? isEmpty(data)
    : Array.isArray(data) && data.length === 0;

  if (dataIsEmpty) {
    return (
      <div className="flex flex-col items-center justify-center p-6 min-h-[200px] text-center">
        {emptyProps?.icon ?? <DefaultEmptyIcon />}
        <h3 className="mt-4 text-lg font-medium text-gray-900">
          {emptyProps?.title ?? 'No data'}
        </h3>
        <p className="mt-1 text-sm text-gray-500">
          {emptyProps?.message ?? 'No items found.'}
        </p>
        {emptyProps?.action && <div className="mt-4">{emptyProps.action}</div>}
      </div>
    );
  }

  // Render data
  return <>{children(data)}</>;
}

/**
 * Props for AsyncBoundary component.
 */
export interface AsyncBoundaryProps<T> {
  /** Promise or async operation state */
  state: {
    data: T | undefined | null;
    loading: boolean;
    error: Error | string | null;
  };
  /** Render content with data */
  children: (data: T) => ReactNode;
  /** Retry callback */
  onRetry?: () => void;
}

/**
 * Simplified async boundary for promise states.
 *
 * @example
 * ```tsx
 * function UserProfile({ userId }: { userId: string }) {
 *   const userState = useAsyncUser(userId);
 *
 *   return (
 *     <AsyncBoundary state={userState} onRetry={userState.refetch}>
 *       {(user) => <ProfileCard user={user} />}
 *     </AsyncBoundary>
 *   );
 * }
 * ```
 */
export function AsyncBoundary<T>({
  state,
  children,
  onRetry,
}: AsyncBoundaryProps<T>): React.ReactElement | null {
  const errorMessage =
    state.error instanceof Error ? state.error.message : state.error;

  return (
    <DataFetchState
      data={state.data}
      loading={state.loading}
      error={errorMessage}
      onRetry={onRetry}
    >
      {children}
    </DataFetchState>
  );
}

/**
 * Props for RetryableLoadingState component.
 */
export interface RetryableLoadingStateProps {
  /** Maximum retry attempts */
  maxRetries?: number;
  /** Current retry count */
  retryCount?: number;
  /** Whether currently loading */
  loading: boolean;
  /** Error message */
  error?: string | null;
  /** Retry callback */
  onRetry: () => void;
  /** Content when loaded successfully */
  children: ReactNode;
}

/**
 * Loading state with retry progress tracking.
 *
 * @example
 * ```tsx
 * const [retryCount, setRetryCount] = useState(0);
 * const { data, loading, error, refetch } = useFetch('/api/data');
 *
 * const handleRetry = () => {
 *   setRetryCount((c) => c + 1);
 *   refetch();
 * };
 *
 * return (
 *   <RetryableLoadingState
 *     loading={loading}
 *     error={error}
 *     retryCount={retryCount}
 *     maxRetries={3}
 *     onRetry={handleRetry}
 *   >
 *     <DataDisplay data={data} />
 *   </RetryableLoadingState>
 * );
 * ```
 */
export const RetryableLoadingState: React.FC<RetryableLoadingStateProps> = ({
  maxRetries = 3,
  retryCount = 0,
  loading,
  error,
  onRetry,
  children,
}) => {
  const canRetry = retryCount < maxRetries;

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[200px]">
        <Spinner size="lg" />
        {retryCount > 0 && (
          <p className="mt-2 text-sm text-gray-500">
            Retry attempt {retryCount}/{maxRetries}
          </p>
        )}
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center p-4 min-h-[200px]">
        <Alert
          variant="error"
          title="Error"
          actions={
            canRetry ? (
              <Button variant="outline" size="sm" onClick={onRetry}>
                Retry ({maxRetries - retryCount} left)
              </Button>
            ) : (
              <span className="text-sm text-red-600">Max retries exceeded</span>
            )
          }
        >
          {error}
        </Alert>
      </div>
    );
  }

  return <>{children}</>;
};

/**
 * Props for SkeletonState component.
 */
export interface SkeletonStateProps {
  /** Whether loading */
  loading: boolean;
  /** Skeleton placeholder */
  skeleton: ReactNode;
  /** Content when loaded */
  children: ReactNode;
}

/**
 * Loading state with skeleton placeholder.
 *
 * @example
 * ```tsx
 * <SkeletonState
 *   loading={loading}
 *   skeleton={<ContentSkeleton variant="card" count={3} />}
 * >
 *   <CardGrid items={items} />
 * </SkeletonState>
 * ```
 */
export const SkeletonState: React.FC<SkeletonStateProps> = ({
  loading,
  skeleton,
  children,
}) => {
  if (loading) {
    return <>{skeleton}</>;
  }
  return <>{children}</>;
};

export default LoadingState;
