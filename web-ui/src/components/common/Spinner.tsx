/**
 * Spinner component - loading indicators in various sizes and styles.
 *
 * A flexible spinner component for indicating loading states,
 * with support for different sizes, colors, and full-page overlays.
 */
import type { HTMLAttributes } from 'react';

/**
 * Spinner size options.
 */
export type SpinnerSize = 'xs' | 'sm' | 'md' | 'lg' | 'xl';

/**
 * Spinner color options.
 */
export type SpinnerColor = 'primary' | 'secondary' | 'white' | 'current';

/**
 * Spinner component props.
 */
export interface SpinnerProps extends HTMLAttributes<HTMLDivElement> {
  /** Size of the spinner */
  size?: SpinnerSize;
  /** Color of the spinner */
  color?: SpinnerColor;
  /** Optional label for accessibility */
  label?: string;
  /** Whether to center the spinner in its container */
  centered?: boolean;
}

/**
 * Get size classes for the spinner.
 */
function getSizeClasses(size: SpinnerSize): string {
  switch (size) {
    case 'xs':
      return 'h-3 w-3';
    case 'sm':
      return 'h-4 w-4';
    case 'md':
      return 'h-6 w-6';
    case 'lg':
      return 'h-8 w-8';
    case 'xl':
      return 'h-12 w-12';
  }
}

/**
 * Get color classes for the spinner.
 */
function getColorClasses(color: SpinnerColor): string {
  switch (color) {
    case 'primary':
      return 'text-weaver-600';
    case 'secondary':
      return 'text-gray-500';
    case 'white':
      return 'text-white';
    case 'current':
      return 'text-current';
  }
}

/**
 * Spinner component for loading states.
 *
 * @example
 * ```tsx
 * // Basic spinner
 * <Spinner />
 *
 * // Large spinner with label
 * <Spinner size="lg" label="Loading data..." />
 *
 * // Centered spinner
 * <Spinner centered size="md" />
 *
 * // White spinner for dark backgrounds
 * <Spinner color="white" />
 * ```
 */
export const Spinner: React.FC<SpinnerProps> = ({
  size = 'md',
  color = 'primary',
  label,
  centered = false,
  className = '',
  ...props
}) => {
  const sizeClasses = getSizeClasses(size);
  const colorClasses = getColorClasses(color);

  const spinner = (
    <svg
      className={`animate-spin ${sizeClasses} ${colorClasses}`}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  );

  const content = label ? (
    <div className="flex items-center gap-2">
      {spinner}
      <span className={`text-sm ${colorClasses}`}>{label}</span>
    </div>
  ) : (
    spinner
  );

  const wrapperClasses = centered
    ? `flex items-center justify-center ${className}`
    : className;

  return (
    <div
      role="status"
      aria-label={label ?? 'Loading'}
      className={wrapperClasses}
      {...props}
    >
      {content}
      {!label && <span className="sr-only">Loading</span>}
    </div>
  );
};

/**
 * Full page loading overlay props.
 */
export interface LoadingOverlayProps {
  /** Whether the overlay is visible */
  visible: boolean;
  /** Loading message to display */
  message?: string;
  /** Whether to use a blurred backdrop */
  blur?: boolean;
  /** Spinner size */
  spinnerSize?: SpinnerSize;
}

/**
 * Full page loading overlay component.
 *
 * @example
 * ```tsx
 * <LoadingOverlay visible={isLoading} message="Saving changes..." />
 * ```
 */
export const LoadingOverlay: React.FC<LoadingOverlayProps> = ({
  visible,
  message = 'Loading...',
  blur = true,
  spinnerSize = 'lg',
}) => {
  if (!visible) {
    return null;
  }

  return (
    <div
      className={`fixed inset-0 z-50 flex items-center justify-center ${
        blur ? 'backdrop-blur-sm' : ''
      } bg-white/80`}
      role="dialog"
      aria-modal="true"
      aria-label="Loading"
    >
      <div className="flex flex-col items-center gap-4">
        <Spinner size={spinnerSize} color="primary" />
        {message && (
          <p className="text-sm font-medium text-gray-600">{message}</p>
        )}
      </div>
    </div>
  );
};

/**
 * Inline loading indicator props.
 */
export interface InlineLoadingProps {
  /** Whether loading is in progress */
  loading: boolean;
  /** Content to show when not loading */
  children: React.ReactNode;
  /** Spinner size (defaults to 'sm') */
  spinnerSize?: SpinnerSize;
  /** Optional loading text */
  loadingText?: string;
}

/**
 * Inline loading indicator that replaces content while loading.
 *
 * @example
 * ```tsx
 * <InlineLoading loading={isSubmitting} loadingText="Saving...">
 *   <span>Save Changes</span>
 * </InlineLoading>
 * ```
 */
export const InlineLoading: React.FC<InlineLoadingProps> = ({
  loading,
  children,
  spinnerSize = 'sm',
  loadingText,
}) => {
  if (loading) {
    return (
      <span className="inline-flex items-center gap-2">
        <Spinner size={spinnerSize} color="current" />
        {loadingText && <span>{loadingText}</span>}
      </span>
    );
  }

  return <>{children}</>;
};

/**
 * Skeleton loading placeholder props.
 */
export interface SkeletonProps extends HTMLAttributes<HTMLDivElement> {
  /** Width of the skeleton */
  width?: string | number;
  /** Height of the skeleton */
  height?: string | number;
  /** Whether to use rounded corners */
  rounded?: boolean | 'full';
  /** Number of lines for text skeleton */
  lines?: number;
}

/**
 * Skeleton loading placeholder component.
 *
 * @example
 * ```tsx
 * // Single skeleton
 * <Skeleton width="100%" height={20} />
 *
 * // Multi-line text skeleton
 * <Skeleton lines={3} />
 *
 * // Circle skeleton (avatar)
 * <Skeleton width={48} height={48} rounded="full" />
 * ```
 */
export const Skeleton: React.FC<SkeletonProps> = ({
  width,
  height,
  rounded = true,
  lines,
  className = '',
  style,
  ...props
}) => {
  const roundedClasses =
    rounded === 'full' ? 'rounded-full' : rounded ? 'rounded' : '';

  if (lines && lines > 1) {
    return (
      <div className={`space-y-2 ${className}`} {...props}>
        {Array.from({ length: lines }).map((_, i) => (
          <div
            key={i}
            className={`bg-gray-200 animate-pulse ${roundedClasses}`}
            style={{
              width: i === lines - 1 ? '75%' : '100%',
              height: height ?? 16,
              ...style,
            }}
          />
        ))}
      </div>
    );
  }

  return (
    <div
      className={`bg-gray-200 animate-pulse ${roundedClasses} ${className}`}
      style={{
        width: width ?? '100%',
        height: height ?? 16,
        ...style,
      }}
      {...props}
    />
  );
};

/**
 * Content skeleton props for more complex loading states.
 */
export interface ContentSkeletonProps {
  /** Variant of the content skeleton */
  variant?: 'card' | 'list' | 'table';
  /** Number of items to show */
  count?: number;
}

/**
 * Pre-built content skeletons for common UI patterns.
 *
 * @example
 * ```tsx
 * // Card skeleton
 * <ContentSkeleton variant="card" count={3} />
 *
 * // List skeleton
 * <ContentSkeleton variant="list" count={5} />
 * ```
 */
export const ContentSkeleton: React.FC<ContentSkeletonProps> = ({
  variant = 'card',
  count = 3,
}) => {
  if (variant === 'card') {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {Array.from({ length: count }).map((_, i) => (
          <div
            key={i}
            className="bg-white rounded-lg border border-gray-200 p-4 animate-pulse"
          >
            <div className="flex items-center gap-3 mb-4">
              <Skeleton width={40} height={40} rounded="full" />
              <div className="flex-1">
                <Skeleton width="60%" height={16} />
                <Skeleton width="40%" height={12} className="mt-2" />
              </div>
            </div>
            <Skeleton lines={2} />
          </div>
        ))}
      </div>
    );
  }

  if (variant === 'list') {
    return (
      <div className="space-y-3">
        {Array.from({ length: count }).map((_, i) => (
          <div
            key={i}
            className="flex items-center gap-3 p-3 bg-white rounded-lg border border-gray-200 animate-pulse"
          >
            <Skeleton width={32} height={32} rounded="full" />
            <div className="flex-1">
              <Skeleton width="70%" height={14} />
              <Skeleton width="50%" height={12} className="mt-1" />
            </div>
            <Skeleton width={60} height={24} />
          </div>
        ))}
      </div>
    );
  }

  // Table variant
  return (
    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden animate-pulse">
      <div className="border-b border-gray-200 p-4 flex gap-4">
        <Skeleton width="20%" height={14} />
        <Skeleton width="30%" height={14} />
        <Skeleton width="25%" height={14} />
        <Skeleton width="15%" height={14} />
      </div>
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="border-b border-gray-100 p-4 flex gap-4">
          <Skeleton width="20%" height={12} />
          <Skeleton width="30%" height={12} />
          <Skeleton width="25%" height={12} />
          <Skeleton width="15%" height={12} />
        </div>
      ))}
    </div>
  );
};

export default Spinner;
