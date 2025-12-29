/**
 * ErrorBoundary component - catches and displays React component errors.
 *
 * A class-based error boundary that catches JavaScript errors in child
 * component trees, logs them, and displays a fallback UI. Supports
 * error recovery, retry mechanisms, and customizable error displays.
 */
import React, { Component, type ErrorInfo, type ReactNode } from 'react';
import { Alert } from './Alert';
import { Button } from './Button';

/**
 * Props for the ErrorBoundary component.
 */
export interface ErrorBoundaryProps {
  /** Child components to render */
  children: ReactNode;
  /** Custom fallback UI to display on error */
  fallback?: ReactNode | ((error: Error, reset: () => void) => ReactNode);
  /** Callback when an error is caught */
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  /** Whether to show the default error UI (default: true) */
  showDefaultUI?: boolean;
  /** Custom title for the error message */
  errorTitle?: string;
  /** Whether to allow retrying (resetting the boundary) */
  allowRetry?: boolean;
  /** Scope name for better error identification */
  scope?: string;
}

/**
 * State for the ErrorBoundary component.
 */
export interface ErrorBoundaryState {
  /** Whether an error has been caught */
  hasError: boolean;
  /** The caught error */
  error: Error | null;
  /** Error info with component stack */
  errorInfo: ErrorInfo | null;
  /** Number of times the component has been reset */
  resetCount: number;
}

/**
 * Error information for display.
 */
export interface ErrorDisplayInfo {
  /** Error message */
  message: string;
  /** Component stack trace */
  componentStack?: string;
  /** Scope where the error occurred */
  scope?: string;
  /** Number of retry attempts */
  retryCount: number;
}

/**
 * Parse error info for display.
 */
function parseErrorInfo(
  error: Error | null,
  errorInfo: ErrorInfo | null,
  scope?: string,
  resetCount?: number
): ErrorDisplayInfo {
  return {
    message: error?.message ?? 'An unexpected error occurred',
    componentStack: errorInfo?.componentStack ?? undefined,
    scope,
    retryCount: resetCount ?? 0,
  };
}

/**
 * ErrorBoundary class component for catching React errors.
 *
 * @example
 * ```tsx
 * // Basic usage
 * <ErrorBoundary>
 *   <YourComponent />
 * </ErrorBoundary>
 *
 * // With custom fallback
 * <ErrorBoundary
 *   fallback={(error, reset) => (
 *     <div>
 *       <p>Error: {error.message}</p>
 *       <button onClick={reset}>Try Again</button>
 *     </div>
 *   )}
 * >
 *   <YourComponent />
 * </ErrorBoundary>
 *
 * // With error callback and scope
 * <ErrorBoundary
 *   scope="MetricsChart"
 *   onError={(error, info) => logErrorToService(error, info)}
 * >
 *   <MetricsChart />
 * </ErrorBoundary>
 * ```
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  static displayName = 'ErrorBoundary';

  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      resetCount: 0,
    };
  }

  /**
   * Derive state from caught error.
   */
  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { hasError: true, error };
  }

  /**
   * Log error information.
   */
  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    // Update state with error info
    this.setState({ errorInfo });

    // Call error callback if provided
    this.props.onError?.(error, errorInfo);
  }

  /**
   * Reset the error boundary to retry rendering.
   */
  reset = (): void => {
    this.setState((prevState) => ({
      hasError: false,
      error: null,
      errorInfo: null,
      resetCount: prevState.resetCount + 1,
    }));
  };

  /**
   * Get the error display info.
   */
  getErrorInfo(): ErrorDisplayInfo {
    return parseErrorInfo(
      this.state.error,
      this.state.errorInfo,
      this.props.scope,
      this.state.resetCount
    );
  }

  render(): ReactNode {
    const { children, fallback, showDefaultUI = true, errorTitle, allowRetry = true } = this.props;
    const { hasError, error } = this.state;

    if (hasError && error) {
      // Custom fallback as function
      if (typeof fallback === 'function') {
        return fallback(error, this.reset);
      }

      // Custom fallback as element
      if (fallback) {
        return fallback;
      }

      // Default error UI
      if (showDefaultUI) {
        const errorInfo = this.getErrorInfo();

        return (
          <DefaultErrorUI
            title={errorTitle}
            errorInfo={errorInfo}
            onRetry={allowRetry ? this.reset : undefined}
          />
        );
      }

      // No UI
      return null;
    }

    return children;
  }
}

/**
 * Props for the DefaultErrorUI component.
 */
interface DefaultErrorUIProps {
  /** Custom title */
  title?: string;
  /** Error information */
  errorInfo: ErrorDisplayInfo;
  /** Retry callback */
  onRetry?: () => void;
}

/**
 * Default error UI component.
 */
const DefaultErrorUI: React.FC<DefaultErrorUIProps> = ({
  title = 'Something went wrong',
  errorInfo,
  onRetry,
}) => {
  const showRetryWarning = errorInfo.retryCount >= 2;

  return (
    <div className="p-4" role="alert">
      <Alert
        variant="error"
        title={title}
        actions={
          onRetry ? (
            <Button
              variant="outline"
              size="sm"
              onClick={onRetry}
              disabled={errorInfo.retryCount >= 5}
            >
              {errorInfo.retryCount >= 5 ? 'Too many retries' : 'Try Again'}
            </Button>
          ) : undefined
        }
      >
        <div className="space-y-2">
          <p>{errorInfo.message}</p>
          {errorInfo.scope && (
            <p className="text-sm opacity-80">Component: {errorInfo.scope}</p>
          )}
          {showRetryWarning && (
            <p className="text-sm opacity-80">
              Retry attempt {errorInfo.retryCount}/5
            </p>
          )}
          {errorInfo.componentStack && process.env.NODE_ENV === 'development' && (
            <details className="mt-2">
              <summary className="cursor-pointer text-sm font-medium">
                Component Stack
              </summary>
              <pre className="mt-2 text-xs whitespace-pre-wrap overflow-auto max-h-48 bg-red-100 p-2 rounded">
                {errorInfo.componentStack}
              </pre>
            </details>
          )}
        </div>
      </Alert>
    </div>
  );
};

/**
 * Props for PageErrorBoundary component.
 */
export interface PageErrorBoundaryProps extends Omit<ErrorBoundaryProps, 'showDefaultUI'> {
  /** Page title for error display */
  pageTitle?: string;
  /** Navigation callback */
  onNavigateHome?: () => void;
}

/**
 * Page-level error boundary with navigation options.
 *
 * @example
 * ```tsx
 * <PageErrorBoundary
 *   pageTitle="Dashboard"
 *   onNavigateHome={() => navigate('/')}
 * >
 *   <DashboardPage />
 * </PageErrorBoundary>
 * ```
 */
export const PageErrorBoundary: React.FC<PageErrorBoundaryProps> = ({
  children,
  pageTitle = 'Page',
  onNavigateHome,
  ...props
}) => {
  return (
    <ErrorBoundary
      {...props}
      errorTitle={`${pageTitle} Error`}
      fallback={(error, reset) => (
        <div className="min-h-[50vh] flex items-center justify-center p-8">
          <div className="max-w-md w-full">
            <Alert
              variant="error"
              title={`Unable to load ${pageTitle}`}
              actions={
                <div className="flex gap-2">
                  <Button variant="outline" size="sm" onClick={reset}>
                    Try Again
                  </Button>
                  {onNavigateHome && (
                    <Button variant="secondary" size="sm" onClick={onNavigateHome}>
                      Go Home
                    </Button>
                  )}
                </div>
              }
            >
              <p>{error.message}</p>
            </Alert>
          </div>
        </div>
      )}
    >
      {children}
    </ErrorBoundary>
  );
};

/**
 * Props for ComponentErrorBoundary component.
 */
export interface ComponentErrorBoundaryProps extends Omit<ErrorBoundaryProps, 'showDefaultUI'> {
  /** Component name for display */
  componentName?: string;
  /** Fallback to show inline (default: simple error message) */
  inlineFallback?: boolean;
}

/**
 * Lightweight error boundary for individual components.
 *
 * @example
 * ```tsx
 * <ComponentErrorBoundary componentName="MetricChart" inlineFallback>
 *   <MetricChart data={data} />
 * </ComponentErrorBoundary>
 * ```
 */
export const ComponentErrorBoundary: React.FC<ComponentErrorBoundaryProps> = ({
  children,
  componentName = 'Component',
  inlineFallback = false,
  ...props
}) => {
  if (inlineFallback) {
    return (
      <ErrorBoundary
        {...props}
        scope={componentName}
        fallback={(error, reset) => (
          <div className="inline-flex items-center gap-2 text-sm text-red-600">
            <svg
              className="h-4 w-4 flex-shrink-0"
              viewBox="0 0 20 20"
              fill="currentColor"
              aria-hidden="true"
            >
              <path
                fillRule="evenodd"
                d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z"
                clipRule="evenodd"
              />
            </svg>
            <span>Error loading {componentName}</span>
            <button
              onClick={reset}
              className="underline hover:no-underline"
              type="button"
            >
              Retry
            </button>
          </div>
        )}
      >
        {children}
      </ErrorBoundary>
    );
  }

  return (
    <ErrorBoundary
      {...props}
      scope={componentName}
      errorTitle={`${componentName} Error`}
    >
      {children}
    </ErrorBoundary>
  );
};

/**
 * withErrorBoundary HOC wraps a component with an error boundary.
 *
 * @param Component The component to wrap
 * @param props Error boundary props
 * @returns Wrapped component
 *
 * @example
 * ```tsx
 * const SafeMetricChart = withErrorBoundary(MetricChart, {
 *   scope: 'MetricChart',
 *   onError: (error) => console.error('Chart error:', error),
 * });
 * ```
 */
export function withErrorBoundary<P extends object>(
  Component: React.ComponentType<P>,
  props: Omit<ErrorBoundaryProps, 'children'> = {}
): React.FC<P> {
  const WrappedComponent: React.FC<P> = (componentProps) => (
    <ErrorBoundary {...props}>
      <Component {...componentProps} />
    </ErrorBoundary>
  );

  WrappedComponent.displayName = `withErrorBoundary(${
    Component.displayName ?? Component.name ?? 'Component'
  })`;

  return WrappedComponent;
}

/**
 * useErrorHandler hook for manual error triggering.
 *
 * @returns A function to throw errors to the nearest error boundary
 *
 * @example
 * ```tsx
 * function DataLoader() {
 *   const handleError = useErrorHandler();
 *
 *   useEffect(() => {
 *     fetchData().catch(handleError);
 *   }, [handleError]);
 *
 *   return <div>Loading...</div>;
 * }
 * ```
 */
export function useErrorHandler(): (error: Error) => void {
  const [, setError] = React.useState<Error | null>(null);

  return React.useCallback((error: Error) => {
    setError(() => {
      throw error;
    });
  }, []);
}

export default ErrorBoundary;
