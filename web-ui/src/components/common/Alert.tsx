/**
 * Alert component - notification and status messages.
 *
 * A flexible alert component for displaying informational, success,
 * warning, and error messages with optional actions and dismissibility.
 */
import { useState, useEffect, useCallback, type ReactNode } from 'react';

/**
 * Alert variant types.
 */
export type AlertVariant = 'info' | 'success' | 'warning' | 'error';

/**
 * Alert component props.
 */
export interface AlertProps {
  /** Visual variant of the alert */
  variant?: AlertVariant;
  /** Title of the alert */
  title?: string;
  /** Content/message of the alert */
  children: ReactNode;
  /** Custom icon to override the default */
  icon?: ReactNode;
  /** Whether to hide the icon */
  hideIcon?: boolean;
  /** Whether the alert can be dismissed */
  dismissible?: boolean;
  /** Callback when the alert is dismissed */
  onDismiss?: () => void;
  /** Actions to display (buttons, links) */
  actions?: ReactNode;
  /** Auto-dismiss after milliseconds (0 = no auto-dismiss) */
  autoDismiss?: number;
  /** Additional CSS classes */
  className?: string;
}

/**
 * Get variant-specific styling configuration.
 */
function getVariantConfig(variant: AlertVariant): {
  containerClasses: string;
  iconClasses: string;
  titleClasses: string;
  textClasses: string;
  icon: ReactNode;
} {
  switch (variant) {
    case 'info':
      return {
        containerClasses: 'bg-blue-50 border-blue-200',
        iconClasses: 'text-blue-400',
        titleClasses: 'text-blue-800',
        textClasses: 'text-blue-700',
        icon: (
          <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
            <path
              fillRule="evenodd"
              d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
              clipRule="evenodd"
            />
          </svg>
        ),
      };
    case 'success':
      return {
        containerClasses: 'bg-green-50 border-green-200',
        iconClasses: 'text-green-400',
        titleClasses: 'text-green-800',
        textClasses: 'text-green-700',
        icon: (
          <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
            <path
              fillRule="evenodd"
              d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
              clipRule="evenodd"
            />
          </svg>
        ),
      };
    case 'warning':
      return {
        containerClasses: 'bg-yellow-50 border-yellow-200',
        iconClasses: 'text-yellow-400',
        titleClasses: 'text-yellow-800',
        textClasses: 'text-yellow-700',
        icon: (
          <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
            <path
              fillRule="evenodd"
              d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
              clipRule="evenodd"
            />
          </svg>
        ),
      };
    case 'error':
      return {
        containerClasses: 'bg-red-50 border-red-200',
        iconClasses: 'text-red-400',
        titleClasses: 'text-red-800',
        textClasses: 'text-red-700',
        icon: (
          <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
            <path
              fillRule="evenodd"
              d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
              clipRule="evenodd"
            />
          </svg>
        ),
      };
  }
}

/**
 * Alert component for displaying messages and notifications.
 *
 * @example
 * ```tsx
 * // Basic info alert
 * <Alert variant="info">
 *   Your session will expire in 5 minutes.
 * </Alert>
 *
 * // Success alert with title
 * <Alert variant="success" title="Success!">
 *   Your changes have been saved.
 * </Alert>
 *
 * // Dismissible warning alert
 * <Alert
 *   variant="warning"
 *   dismissible
 *   onDismiss={() => setShowAlert(false)}
 * >
 *   This action cannot be undone.
 * </Alert>
 *
 * // Error alert with actions
 * <Alert
 *   variant="error"
 *   title="Upload Failed"
 *   actions={<button onClick={retry}>Retry</button>}
 * >
 *   The file could not be uploaded. Please try again.
 * </Alert>
 *
 * // Auto-dismissing success alert
 * <Alert variant="success" autoDismiss={3000}>
 *   Settings saved successfully!
 * </Alert>
 * ```
 */
export const Alert: React.FC<AlertProps> = ({
  variant = 'info',
  title,
  children,
  icon,
  hideIcon = false,
  dismissible = false,
  onDismiss,
  actions,
  autoDismiss = 0,
  className = '',
}) => {
  const [visible, setVisible] = useState(true);

  const config = getVariantConfig(variant);

  // Handle dismiss
  const handleDismiss = useCallback(() => {
    setVisible(false);
    onDismiss?.();
  }, [onDismiss]);

  // Auto-dismiss timer
  useEffect(() => {
    if (autoDismiss > 0) {
      const timer = setTimeout(handleDismiss, autoDismiss);
      return () => clearTimeout(timer);
    }
  }, [autoDismiss, handleDismiss]);

  if (!visible) {
    return null;
  }

  return (
    <div
      className={`border rounded-lg p-4 ${config.containerClasses} ${className}`}
      role="alert"
      aria-live="polite"
    >
      <div className="flex">
        {/* Icon */}
        {!hideIcon && (
          <div className={`flex-shrink-0 ${config.iconClasses}`}>
            {icon ?? config.icon}
          </div>
        )}

        {/* Content */}
        <div className={`${hideIcon ? '' : 'ml-3'} flex-1`}>
          {/* Header row with title and dismiss button */}
          {(title || dismissible) && (
            <div className="flex items-center justify-between">
              {title && (
                <h3 className={`text-sm font-medium ${config.titleClasses}`}>
                  {title}
                </h3>
              )}
              {!title && <div />}
              {dismissible && (
                <button
                  type="button"
                  onClick={handleDismiss}
                  className={`-mr-1 -mt-1 p-1 rounded hover:bg-black/5 focus:outline-none focus:ring-2 focus:ring-offset-2 ${
                    variant === 'error'
                      ? 'focus:ring-red-500'
                      : variant === 'warning'
                        ? 'focus:ring-yellow-500'
                        : variant === 'success'
                          ? 'focus:ring-green-500'
                          : 'focus:ring-blue-500'
                  }`}
                  aria-label="Dismiss"
                >
                  <svg
                    className={`h-4 w-4 ${config.iconClasses}`}
                    viewBox="0 0 20 20"
                    fill="currentColor"
                    aria-hidden="true"
                  >
                    <path
                      fillRule="evenodd"
                      d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                      clipRule="evenodd"
                    />
                  </svg>
                </button>
              )}
            </div>
          )}

          {/* Message content */}
          <div className={`text-sm ${config.textClasses} ${title ? 'mt-1' : ''}`}>
            {children}
          </div>

          {/* Actions */}
          {actions && (
            <div className={`mt-3 flex gap-2 ${config.textClasses}`}>
              {actions}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

/**
 * Toast notification props.
 */
export interface ToastProps extends Omit<AlertProps, 'dismissible' | 'className'> {
  /** Position of the toast */
  position?: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left' | 'top-center' | 'bottom-center';
}

/**
 * Get position classes for toast.
 */
function getPositionClasses(position: ToastProps['position']): string {
  switch (position) {
    case 'top-left':
      return 'top-4 left-4';
    case 'top-center':
      return 'top-4 left-1/2 -translate-x-1/2';
    case 'top-right':
    default:
      return 'top-4 right-4';
    case 'bottom-left':
      return 'bottom-4 left-4';
    case 'bottom-center':
      return 'bottom-4 left-1/2 -translate-x-1/2';
    case 'bottom-right':
      return 'bottom-4 right-4';
  }
}

/**
 * Toast notification component (positioned alert).
 *
 * @example
 * ```tsx
 * <Toast
 *   variant="success"
 *   position="top-right"
 *   autoDismiss={5000}
 *   onDismiss={() => setShowToast(false)}
 * >
 *   File uploaded successfully!
 * </Toast>
 * ```
 */
export const Toast: React.FC<ToastProps> = ({
  position = 'top-right',
  autoDismiss = 5000,
  ...props
}) => {
  const positionClasses = getPositionClasses(position);

  return (
    <div className={`fixed z-50 ${positionClasses} max-w-sm w-full`}>
      <Alert
        {...props}
        dismissible
        autoDismiss={autoDismiss}
        className="shadow-lg"
      />
    </div>
  );
};

/**
 * Banner alert props (full-width alert).
 */
export interface BannerProps extends Omit<AlertProps, 'className'> {
  /** Whether the banner is sticky */
  sticky?: boolean;
}

/**
 * Banner alert component (full-width notification).
 *
 * @example
 * ```tsx
 * <Banner variant="warning" sticky>
 *   System maintenance scheduled for tonight at 11 PM.
 * </Banner>
 * ```
 */
export const Banner: React.FC<BannerProps> = ({ sticky = false, ...props }) => {
  return (
    <div className={sticky ? 'sticky top-0 z-40' : ''}>
      <Alert {...props} className="rounded-none border-x-0" />
    </div>
  );
};

/**
 * Inline message props (compact alert for forms).
 */
export interface InlineMessageProps {
  /** Variant of the message */
  variant?: AlertVariant;
  /** Message content */
  children: ReactNode;
  /** Additional CSS classes */
  className?: string;
}

/**
 * Inline message component for form feedback.
 *
 * @example
 * ```tsx
 * <InlineMessage variant="error">
 *   Email is required.
 * </InlineMessage>
 * ```
 */
export const InlineMessage: React.FC<InlineMessageProps> = ({
  variant = 'error',
  children,
  className = '',
}) => {
  const config = getVariantConfig(variant);

  return (
    <p
      className={`flex items-center gap-1.5 text-sm ${config.textClasses} ${className}`}
      role="alert"
    >
      <span className={config.iconClasses}>
        {variant === 'error' ? (
          <svg className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
            <path
              fillRule="evenodd"
              d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z"
              clipRule="evenodd"
            />
          </svg>
        ) : variant === 'success' ? (
          <svg className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
            <path
              fillRule="evenodd"
              d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
              clipRule="evenodd"
            />
          </svg>
        ) : (
          <svg className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
            <path
              fillRule="evenodd"
              d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
              clipRule="evenodd"
            />
          </svg>
        )}
      </span>
      <span>{children}</span>
    </p>
  );
};

export default Alert;
