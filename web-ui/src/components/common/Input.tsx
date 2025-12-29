/**
 * Input component - reusable form input with label, error handling, and variants.
 *
 * A flexible input component supporting different types, sizes, and states
 * with built-in label, help text, and error display.
 */
import { forwardRef, type InputHTMLAttributes, type ReactNode, useId } from 'react';

/**
 * Input size types.
 */
export type InputSize = 'sm' | 'md' | 'lg';

/**
 * Input component props.
 */
export interface InputProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'size'> {
  /** Label text for the input */
  label?: string;
  /** Error message to display */
  error?: string | null;
  /** Help text to display below the input */
  helpText?: string;
  /** Size of the input */
  size?: InputSize;
  /** Icon to display on the left side */
  leftIcon?: ReactNode;
  /** Icon to display on the right side */
  rightIcon?: ReactNode;
  /** Whether the input should take full width */
  fullWidth?: boolean;
  /** Whether the label should be visually hidden but accessible */
  srOnlyLabel?: boolean;
  /** Additional classes for the container */
  containerClassName?: string;
}

/**
 * Get size-specific classes for the input.
 */
function getSizeClasses(size: InputSize): string {
  switch (size) {
    case 'sm':
      return 'px-2.5 py-1.5 text-xs';
    case 'md':
      return 'px-3 py-2 text-sm';
    case 'lg':
      return 'px-4 py-2.5 text-base';
  }
}

/**
 * Get padding adjustment for icons.
 */
function getIconPadding(size: InputSize, position: 'left' | 'right'): string {
  const padding = {
    sm: { left: 'pl-8', right: 'pr-8' },
    md: { left: 'pl-10', right: 'pr-10' },
    lg: { left: 'pl-12', right: 'pr-12' },
  };
  return padding[size][position];
}

/**
 * Get icon container position and size.
 */
function getIconContainerClasses(size: InputSize, position: 'left' | 'right'): string {
  const baseClasses = 'absolute inset-y-0 flex items-center pointer-events-none';
  const positionClasses = position === 'left' ? 'left-0 pl-3' : 'right-0 pr-3';
  const sizeClasses = {
    sm: '[&>svg]:w-4 [&>svg]:h-4',
    md: '[&>svg]:w-5 [&>svg]:h-5',
    lg: '[&>svg]:w-6 [&>svg]:h-6',
  };
  return `${baseClasses} ${positionClasses} ${sizeClasses[size]}`;
}

/**
 * Input component with label, icons, and error handling.
 *
 * @example
 * ```tsx
 * // Basic input with label
 * <Input label="Email" type="email" placeholder="you@example.com" />
 *
 * // Input with error
 * <Input label="Password" type="password" error="Password is required" />
 *
 * // Input with left icon
 * <Input label="Search" leftIcon={<SearchIcon />} placeholder="Search..." />
 *
 * // Controlled input
 * <Input
 *   label="Name"
 *   value={name}
 *   onChange={(e) => setName(e.target.value)}
 * />
 * ```
 */
export const Input = forwardRef<HTMLInputElement, InputProps>(
  (
    {
      label,
      error,
      helpText,
      size = 'md',
      leftIcon,
      rightIcon,
      fullWidth = true,
      srOnlyLabel = false,
      containerClassName = '',
      className = '',
      id: providedId,
      disabled,
      required,
      ...props
    },
    ref
  ) => {
    const generatedId = useId();
    const inputId = providedId ?? generatedId;
    const errorId = `${inputId}-error`;
    const helpId = `${inputId}-help`;

    const hasError = !!error;

    // Build input classes
    const baseInputClasses =
      'block w-full rounded-md shadow-sm transition-colors focus:outline-none focus:ring-2 focus:ring-offset-0';
    const sizeClasses = getSizeClasses(size);
    const stateClasses = hasError
      ? 'border-red-300 text-red-900 placeholder-red-300 focus:border-red-500 focus:ring-red-500'
      : 'border-gray-300 text-gray-900 placeholder-gray-400 focus:border-weaver-500 focus:ring-weaver-500';
    const disabledClasses = disabled ? 'bg-gray-50 text-gray-500 cursor-not-allowed' : '';
    const iconPaddingLeft = leftIcon ? getIconPadding(size, 'left') : '';
    const iconPaddingRight = rightIcon ? getIconPadding(size, 'right') : '';

    const inputClasses = [
      baseInputClasses,
      sizeClasses,
      stateClasses,
      disabledClasses,
      iconPaddingLeft,
      iconPaddingRight,
      className,
    ]
      .filter(Boolean)
      .join(' ');

    const containerClasses = [fullWidth ? 'w-full' : '', containerClassName]
      .filter(Boolean)
      .join(' ');

    return (
      <div className={containerClasses}>
        {/* Label */}
        {label && (
          <label
            htmlFor={inputId}
            className={`block text-sm font-medium text-gray-700 mb-1 ${
              srOnlyLabel ? 'sr-only' : ''
            }`}
          >
            {label}
            {required && <span className="text-red-500 ml-0.5">*</span>}
          </label>
        )}

        {/* Input container */}
        <div className="relative">
          {/* Left icon */}
          {leftIcon && (
            <div className={`${getIconContainerClasses(size, 'left')} text-gray-400`}>
              {leftIcon}
            </div>
          )}

          {/* Input element */}
          <input
            ref={ref}
            id={inputId}
            disabled={disabled}
            required={required}
            aria-invalid={hasError}
            aria-describedby={
              [hasError ? errorId : null, helpText ? helpId : null]
                .filter(Boolean)
                .join(' ') || undefined
            }
            className={inputClasses}
            {...props}
          />

          {/* Right icon or error icon */}
          {(rightIcon || hasError) && (
            <div
              className={`${getIconContainerClasses(size, 'right')} ${
                hasError ? 'text-red-500' : 'text-gray-400'
              }`}
            >
              {hasError ? (
                <svg
                  className="h-5 w-5"
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
              ) : (
                rightIcon
              )}
            </div>
          )}
        </div>

        {/* Error message */}
        {hasError && (
          <p id={errorId} className="mt-1 text-sm text-red-600" role="alert">
            {error}
          </p>
        )}

        {/* Help text */}
        {helpText && !hasError && (
          <p id={helpId} className="mt-1 text-sm text-gray-500">
            {helpText}
          </p>
        )}
      </div>
    );
  }
);

Input.displayName = 'Input';

/**
 * Textarea component props.
 */
export interface TextareaProps
  extends Omit<React.TextareaHTMLAttributes<HTMLTextAreaElement>, 'size'> {
  /** Label text for the textarea */
  label?: string;
  /** Error message to display */
  error?: string | null;
  /** Help text to display below the textarea */
  helpText?: string;
  /** Size of the textarea */
  size?: InputSize;
  /** Whether the textarea should take full width */
  fullWidth?: boolean;
  /** Whether to allow resize */
  resize?: 'none' | 'vertical' | 'horizontal' | 'both';
  /** Additional classes for the container */
  containerClassName?: string;
}

/**
 * Textarea component with label and error handling.
 *
 * @example
 * ```tsx
 * // Basic textarea
 * <Textarea label="Description" rows={4} />
 *
 * // Textarea with error
 * <Textarea label="Bio" error="Bio is too long" maxLength={500} />
 * ```
 */
export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  (
    {
      label,
      error,
      helpText,
      size = 'md',
      fullWidth = true,
      resize = 'vertical',
      containerClassName = '',
      className = '',
      id: providedId,
      disabled,
      required,
      ...props
    },
    ref
  ) => {
    const generatedId = useId();
    const textareaId = providedId ?? generatedId;
    const errorId = `${textareaId}-error`;
    const helpId = `${textareaId}-help`;

    const hasError = !!error;

    // Build textarea classes
    const baseClasses =
      'block w-full rounded-md shadow-sm transition-colors focus:outline-none focus:ring-2 focus:ring-offset-0';
    const sizeClasses = getSizeClasses(size);
    const stateClasses = hasError
      ? 'border-red-300 text-red-900 placeholder-red-300 focus:border-red-500 focus:ring-red-500'
      : 'border-gray-300 text-gray-900 placeholder-gray-400 focus:border-weaver-500 focus:ring-weaver-500';
    const disabledClasses = disabled ? 'bg-gray-50 text-gray-500 cursor-not-allowed' : '';
    const resizeClasses = {
      none: 'resize-none',
      vertical: 'resize-y',
      horizontal: 'resize-x',
      both: 'resize',
    };

    const textareaClasses = [
      baseClasses,
      sizeClasses,
      stateClasses,
      disabledClasses,
      resizeClasses[resize],
      className,
    ]
      .filter(Boolean)
      .join(' ');

    const containerClasses = [fullWidth ? 'w-full' : '', containerClassName]
      .filter(Boolean)
      .join(' ');

    return (
      <div className={containerClasses}>
        {/* Label */}
        {label && (
          <label
            htmlFor={textareaId}
            className="block text-sm font-medium text-gray-700 mb-1"
          >
            {label}
            {required && <span className="text-red-500 ml-0.5">*</span>}
          </label>
        )}

        {/* Textarea element */}
        <textarea
          ref={ref}
          id={textareaId}
          disabled={disabled}
          required={required}
          aria-invalid={hasError}
          aria-describedby={
            [hasError ? errorId : null, helpText ? helpId : null]
              .filter(Boolean)
              .join(' ') || undefined
          }
          className={textareaClasses}
          {...props}
        />

        {/* Error message */}
        {hasError && (
          <p id={errorId} className="mt-1 text-sm text-red-600" role="alert">
            {error}
          </p>
        )}

        {/* Help text */}
        {helpText && !hasError && (
          <p id={helpId} className="mt-1 text-sm text-gray-500">
            {helpText}
          </p>
        )}
      </div>
    );
  }
);

Textarea.displayName = 'Textarea';

export default Input;
