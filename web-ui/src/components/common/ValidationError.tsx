/**
 * ValidationError component for displaying validation messages.
 *
 * A reusable component for showing validation errors, warnings, and info messages
 * in a consistent, accessible format. Supports different severity levels and
 * can display single or multiple messages.
 */

/**
 * Validation message severity levels.
 */
export type ValidationSeverity = 'error' | 'warning' | 'info';

/**
 * Single validation message structure.
 */
export interface ValidationMessage {
  /** The message text to display */
  message: string;
  /** Optional field path this error relates to (e.g., "agents.senior.model") */
  field?: string;
  /** Severity level - defaults to 'error' */
  severity?: ValidationSeverity;
}

/**
 * ValidationError component props.
 */
export interface ValidationErrorProps {
  /** Single error message or array of messages */
  errors: string | string[] | ValidationMessage | ValidationMessage[];
  /** Title for the error section - auto-generated if not provided */
  title?: string;
  /** Default severity when using string errors - defaults to 'error' */
  severity?: ValidationSeverity;
  /** Whether to show field paths with errors */
  showFieldPaths?: boolean;
  /** Callback when dismiss button is clicked (if dismissible) */
  onDismiss?: () => void;
  /** Additional CSS classes */
  className?: string;
}

/**
 * Get styling configuration based on severity level.
 */
function getSeverityConfig(severity: ValidationSeverity): {
  containerClasses: string;
  iconColor: string;
  titleColor: string;
  textColor: string;
  icon: React.ReactNode;
} {
  switch (severity) {
    case 'error':
      return {
        containerClasses: 'bg-red-50 border-red-200',
        iconColor: 'text-red-400',
        titleColor: 'text-red-800',
        textColor: 'text-red-700',
        icon: (
          <svg className="w-5 h-5" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
            <path
              fillRule="evenodd"
              d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
              clipRule="evenodd"
            />
          </svg>
        ),
      };
    case 'warning':
      return {
        containerClasses: 'bg-yellow-50 border-yellow-200',
        iconColor: 'text-yellow-400',
        titleColor: 'text-yellow-800',
        textColor: 'text-yellow-700',
        icon: (
          <svg className="w-5 h-5" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
            <path
              fillRule="evenodd"
              d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
              clipRule="evenodd"
            />
          </svg>
        ),
      };
    case 'info':
      return {
        containerClasses: 'bg-blue-50 border-blue-200',
        iconColor: 'text-blue-400',
        titleColor: 'text-blue-800',
        textColor: 'text-blue-700',
        icon: (
          <svg className="w-5 h-5" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
            <path
              fillRule="evenodd"
              d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
              clipRule="evenodd"
            />
          </svg>
        ),
      };
  }
}

/**
 * Get default title based on severity and error count.
 */
function getDefaultTitle(severity: ValidationSeverity, count: number): string {
  const plural = count > 1;
  switch (severity) {
    case 'error':
      return plural ? 'Validation Errors' : 'Validation Error';
    case 'warning':
      return plural ? 'Warnings' : 'Warning';
    case 'info':
      return plural ? 'Information' : 'Information';
  }
}

/**
 * Normalize errors input to array of ValidationMessage objects.
 */
function normalizeErrors(
  errors: ValidationErrorProps['errors'],
  defaultSeverity: ValidationSeverity
): ValidationMessage[] {
  if (typeof errors === 'string') {
    return [{ message: errors, severity: defaultSeverity }];
  }

  if (Array.isArray(errors)) {
    return errors.map((err) => {
      if (typeof err === 'string') {
        return { message: err, severity: defaultSeverity };
      }
      return { ...err, severity: err.severity ?? defaultSeverity };
    });
  }

  // Single ValidationMessage object
  return [{ ...errors, severity: errors.severity ?? defaultSeverity }];
}

/**
 * ValidationError component for displaying validation messages.
 *
 * Supports multiple severity levels (error, warning, info) and can display
 * single or multiple messages. Optionally shows field paths for context.
 */
export const ValidationError: React.FC<ValidationErrorProps> = ({
  errors,
  title,
  severity = 'error',
  showFieldPaths = true,
  onDismiss,
  className = '',
}) => {
  const normalizedErrors = normalizeErrors(errors, severity);

  // Don't render if no errors
  if (normalizedErrors.length === 0) {
    return null;
  }

  // Determine primary severity (use most severe if mixed)
  const primarySeverity = normalizedErrors.some((e) => e.severity === 'error')
    ? 'error'
    : normalizedErrors.some((e) => e.severity === 'warning')
      ? 'warning'
      : 'info';

  const config = getSeverityConfig(primarySeverity);
  const displayTitle = title ?? getDefaultTitle(primarySeverity, normalizedErrors.length);

  return (
    <div
      className={`border rounded-lg p-4 ${config.containerClasses} ${className}`}
      role="alert"
      aria-live="polite"
    >
      <div className="flex">
        <div className={`flex-shrink-0 ${config.iconColor}`}>{config.icon}</div>
        <div className="ml-3 flex-1">
          <div className="flex items-center justify-between">
            <h3 className={`text-sm font-medium ${config.titleColor}`}>{displayTitle}</h3>
            {onDismiss && (
              <button
                type="button"
                onClick={onDismiss}
                className={`-mr-1 -mt-1 p-1 rounded hover:bg-black/5 focus:outline-none focus:ring-2 focus:ring-offset-2 ${
                  primarySeverity === 'error'
                    ? 'focus:ring-red-500'
                    : primarySeverity === 'warning'
                      ? 'focus:ring-yellow-500'
                      : 'focus:ring-blue-500'
                }`}
                aria-label="Dismiss"
              >
                <svg
                  className={`w-4 h-4 ${config.iconColor}`}
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
          {normalizedErrors.length === 1 ? (
            <p className={`mt-1 text-sm ${config.textColor}`}>
              {showFieldPaths && normalizedErrors[0].field && (
                <code className="font-mono text-xs bg-black/5 px-1 py-0.5 rounded mr-2">
                  {normalizedErrors[0].field}
                </code>
              )}
              {normalizedErrors[0].message}
            </p>
          ) : (
            <ul className={`mt-2 text-sm ${config.textColor} list-disc list-inside space-y-1`}>
              {normalizedErrors.map((err, index) => (
                <li key={index}>
                  {showFieldPaths && err.field && (
                    <code className="font-mono text-xs bg-black/5 px-1 py-0.5 rounded mr-2">
                      {err.field}
                    </code>
                  )}
                  {err.message}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
};

/**
 * FieldError component for inline field-level validation errors.
 * Use this for displaying errors directly below form fields.
 */
export interface FieldErrorProps {
  /** Error message to display */
  error?: string | null;
  /** Additional CSS classes */
  className?: string;
}

/**
 * FieldError component for inline field-level validation.
 */
export const FieldError: React.FC<FieldErrorProps> = ({ error, className = '' }) => {
  if (!error) {
    return null;
  }

  return (
    <p
      className={`mt-1 text-sm text-red-600 flex items-center gap-1 ${className}`}
      role="alert"
      aria-live="polite"
    >
      <svg
        className="w-4 h-4 flex-shrink-0"
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
      <span>{error}</span>
    </p>
  );
};

export default ValidationError;
