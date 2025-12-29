/**
 * Button component - reusable button with multiple variants and states.
 *
 * A flexible button component supporting different variants (primary, secondary,
 * outline, ghost, danger), sizes, loading states, and icons.
 */
import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react';

/**
 * Button variant types.
 */
export type ButtonVariant = 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger';

/**
 * Button size types.
 */
export type ButtonSize = 'xs' | 'sm' | 'md' | 'lg';

/**
 * Button component props.
 */
export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  /** Visual variant of the button */
  variant?: ButtonVariant;
  /** Size of the button */
  size?: ButtonSize;
  /** Whether the button is in a loading state */
  loading?: boolean;
  /** Icon to display before the text */
  leftIcon?: ReactNode;
  /** Icon to display after the text */
  rightIcon?: ReactNode;
  /** Whether the button should take full width */
  fullWidth?: boolean;
}

/**
 * Get variant-specific classes.
 */
function getVariantClasses(variant: ButtonVariant): string {
  switch (variant) {
    case 'primary':
      return 'bg-weaver-600 text-white hover:bg-weaver-700 focus:ring-weaver-500 disabled:bg-weaver-400';
    case 'secondary':
      return 'bg-gray-100 text-gray-700 hover:bg-gray-200 focus:ring-gray-500 disabled:bg-gray-50 disabled:text-gray-400';
    case 'outline':
      return 'border-2 border-gray-300 text-gray-700 hover:bg-gray-50 focus:ring-gray-500 disabled:border-gray-200 disabled:text-gray-400';
    case 'ghost':
      return 'text-gray-700 hover:bg-gray-100 focus:ring-gray-500 disabled:text-gray-400';
    case 'danger':
      return 'bg-red-600 text-white hover:bg-red-700 focus:ring-red-500 disabled:bg-red-400';
  }
}

/**
 * Get size-specific classes.
 */
function getSizeClasses(size: ButtonSize): string {
  switch (size) {
    case 'xs':
      return 'px-2 py-1 text-xs';
    case 'sm':
      return 'px-3 py-1.5 text-sm';
    case 'md':
      return 'px-4 py-2 text-sm';
    case 'lg':
      return 'px-6 py-3 text-base';
  }
}

/**
 * Get spinner size based on button size.
 */
function getSpinnerSize(size: ButtonSize): string {
  switch (size) {
    case 'xs':
      return 'h-3 w-3';
    case 'sm':
      return 'h-3.5 w-3.5';
    case 'md':
      return 'h-4 w-4';
    case 'lg':
      return 'h-5 w-5';
  }
}

/**
 * Loading spinner SVG component.
 */
const LoadingSpinner: React.FC<{ size: ButtonSize }> = ({ size }) => (
  <svg
    className={`animate-spin ${getSpinnerSize(size)}`}
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

/**
 * Button component with variants, sizes, and loading states.
 *
 * @example
 * ```tsx
 * // Primary button
 * <Button variant="primary">Save</Button>
 *
 * // Secondary button with loading
 * <Button variant="secondary" loading>Processing...</Button>
 *
 * // Button with icons
 * <Button variant="outline" leftIcon={<PlusIcon />}>Add Item</Button>
 *
 * // Danger button
 * <Button variant="danger" size="sm">Delete</Button>
 * ```
 */
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      variant = 'primary',
      size = 'md',
      loading = false,
      leftIcon,
      rightIcon,
      fullWidth = false,
      disabled,
      className = '',
      children,
      ...props
    },
    ref
  ) => {
    const isDisabled = disabled || loading;

    const baseClasses =
      'inline-flex items-center justify-center font-medium rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:cursor-not-allowed';
    const variantClasses = getVariantClasses(variant);
    const sizeClasses = getSizeClasses(size);
    const widthClasses = fullWidth ? 'w-full' : '';

    const combinedClasses = [
      baseClasses,
      variantClasses,
      sizeClasses,
      widthClasses,
      className,
    ]
      .filter(Boolean)
      .join(' ');

    return (
      <button
        ref={ref}
        type="button"
        disabled={isDisabled}
        className={combinedClasses}
        {...props}
      >
        {loading && <LoadingSpinner size={size} />}
        {!loading && leftIcon && (
          <span className="flex-shrink-0 mr-2">{leftIcon}</span>
        )}
        {loading ? (
          <span className="ml-2">{children}</span>
        ) : (
          children
        )}
        {!loading && rightIcon && (
          <span className="flex-shrink-0 ml-2">{rightIcon}</span>
        )}
      </button>
    );
  }
);

Button.displayName = 'Button';

export default Button;
