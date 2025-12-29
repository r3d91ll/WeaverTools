/**
 * Card component - reusable container with header, content, and footer sections.
 *
 * A flexible card component for displaying content in a contained, styled box
 * with optional header, footer, and various padding options.
 */
import { forwardRef, type HTMLAttributes, type ReactNode } from 'react';

/**
 * Card padding options.
 */
export type CardPadding = 'none' | 'sm' | 'md' | 'lg';

/**
 * Card component props.
 */
export interface CardProps extends HTMLAttributes<HTMLDivElement> {
  /** Card title - rendered in the header */
  title?: ReactNode;
  /** Subtitle or description - rendered below the title */
  subtitle?: ReactNode;
  /** Actions for the header (buttons, icons, etc.) */
  headerActions?: ReactNode;
  /** Footer content */
  footer?: ReactNode;
  /** Content padding size */
  padding?: CardPadding;
  /** Whether to show a hover effect */
  hoverable?: boolean;
  /** Whether to show as selected/active */
  selected?: boolean;
  /** Whether the card is in a loading state */
  loading?: boolean;
  /** Whether to remove the border */
  noBorder?: boolean;
  /** Whether to remove the shadow */
  noShadow?: boolean;
}

/**
 * Get padding classes.
 */
function getPaddingClasses(padding: CardPadding): string {
  switch (padding) {
    case 'none':
      return '';
    case 'sm':
      return 'p-3';
    case 'md':
      return 'p-4';
    case 'lg':
      return 'p-6';
  }
}

/**
 * Loading skeleton component.
 */
const LoadingSkeleton: React.FC = () => (
  <div className="animate-pulse space-y-4">
    <div className="h-4 bg-gray-200 rounded w-3/4" />
    <div className="space-y-2">
      <div className="h-3 bg-gray-200 rounded" />
      <div className="h-3 bg-gray-200 rounded w-5/6" />
    </div>
  </div>
);

/**
 * Card component with header, content, and footer sections.
 *
 * @example
 * ```tsx
 * // Basic card
 * <Card title="Settings">
 *   <p>Card content here</p>
 * </Card>
 *
 * // Card with header actions
 * <Card
 *   title="Users"
 *   subtitle="Manage team members"
 *   headerActions={<Button size="sm">Add User</Button>}
 * >
 *   <UserList />
 * </Card>
 *
 * // Card with footer
 * <Card
 *   title="Profile"
 *   footer={
 *     <div className="flex justify-end gap-2">
 *       <Button variant="secondary">Cancel</Button>
 *       <Button variant="primary">Save</Button>
 *     </div>
 *   }
 * >
 *   <ProfileForm />
 * </Card>
 *
 * // Hoverable card
 * <Card hoverable onClick={() => navigate('/details')}>
 *   <p>Click me</p>
 * </Card>
 * ```
 */
export const Card = forwardRef<HTMLDivElement, CardProps>(
  (
    {
      title,
      subtitle,
      headerActions,
      footer,
      padding = 'md',
      hoverable = false,
      selected = false,
      loading = false,
      noBorder = false,
      noShadow = false,
      className = '',
      children,
      ...props
    },
    ref
  ) => {
    const hasHeader = title || subtitle || headerActions;

    // Build card classes
    const baseClasses = 'bg-white rounded-lg transition-all';
    const borderClasses = noBorder ? '' : 'border border-gray-200';
    const shadowClasses = noShadow ? '' : 'shadow-sm';
    const hoverClasses = hoverable
      ? 'hover:shadow-md hover:border-gray-300 cursor-pointer'
      : '';
    const selectedClasses = selected
      ? 'ring-2 ring-weaver-500 border-weaver-500'
      : '';
    const loadingClasses = loading ? 'opacity-75' : '';

    const cardClasses = [
      baseClasses,
      borderClasses,
      shadowClasses,
      hoverClasses,
      selectedClasses,
      loadingClasses,
      className,
    ]
      .filter(Boolean)
      .join(' ');

    const contentPaddingClasses = getPaddingClasses(padding);

    return (
      <div ref={ref} className={cardClasses} {...props}>
        {/* Header */}
        {hasHeader && (
          <div
            className={`flex items-start justify-between ${contentPaddingClasses} ${
              children || footer ? 'border-b border-gray-100' : ''
            }`}
          >
            <div className="flex-1 min-w-0">
              {title && (
                <h3 className="text-base font-semibold text-gray-900 truncate">
                  {title}
                </h3>
              )}
              {subtitle && (
                <p className="mt-0.5 text-sm text-gray-500 truncate">{subtitle}</p>
              )}
            </div>
            {headerActions && (
              <div className="flex-shrink-0 ml-4">{headerActions}</div>
            )}
          </div>
        )}

        {/* Content */}
        {children && (
          <div className={contentPaddingClasses}>
            {loading ? <LoadingSkeleton /> : children}
          </div>
        )}

        {/* Footer */}
        {footer && (
          <div
            className={`${contentPaddingClasses} border-t border-gray-100`}
          >
            {footer}
          </div>
        )}
      </div>
    );
  }
);

Card.displayName = 'Card';

/**
 * CardGrid component for displaying cards in a responsive grid.
 */
export interface CardGridProps extends HTMLAttributes<HTMLDivElement> {
  /** Number of columns (responsive: 1 on mobile, specified on larger screens) */
  columns?: 1 | 2 | 3 | 4;
  /** Gap between cards */
  gap?: 'sm' | 'md' | 'lg';
}

/**
 * Get grid column classes.
 */
function getGridColumnClasses(columns: 1 | 2 | 3 | 4): string {
  switch (columns) {
    case 1:
      return 'grid-cols-1';
    case 2:
      return 'grid-cols-1 md:grid-cols-2';
    case 3:
      return 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3';
    case 4:
      return 'grid-cols-1 md:grid-cols-2 lg:grid-cols-4';
  }
}

/**
 * Get gap classes.
 */
function getGapClasses(gap: 'sm' | 'md' | 'lg'): string {
  switch (gap) {
    case 'sm':
      return 'gap-3';
    case 'md':
      return 'gap-4';
    case 'lg':
      return 'gap-6';
  }
}

/**
 * CardGrid component for responsive card layouts.
 *
 * @example
 * ```tsx
 * <CardGrid columns={3} gap="md">
 *   <Card>Card 1</Card>
 *   <Card>Card 2</Card>
 *   <Card>Card 3</Card>
 * </CardGrid>
 * ```
 */
export const CardGrid = forwardRef<HTMLDivElement, CardGridProps>(
  ({ columns = 3, gap = 'md', className = '', children, ...props }, ref) => {
    const columnClasses = getGridColumnClasses(columns);
    const gapClasses = getGapClasses(gap);

    return (
      <div
        ref={ref}
        className={`grid ${columnClasses} ${gapClasses} ${className}`}
        {...props}
      >
        {children}
      </div>
    );
  }
);

CardGrid.displayName = 'CardGrid';

/**
 * Stat card for displaying key metrics.
 */
export interface StatCardProps {
  /** Stat label */
  label: string;
  /** Stat value */
  value: ReactNode;
  /** Change indicator (e.g., "+12%") */
  change?: string;
  /** Whether the change is positive */
  changePositive?: boolean;
  /** Icon to display */
  icon?: ReactNode;
  /** Additional class name */
  className?: string;
}

/**
 * StatCard component for displaying key metrics.
 *
 * @example
 * ```tsx
 * <StatCard
 *   label="Total Users"
 *   value="1,234"
 *   change="+12%"
 *   changePositive
 *   icon={<UsersIcon />}
 * />
 * ```
 */
export const StatCard: React.FC<StatCardProps> = ({
  label,
  value,
  change,
  changePositive,
  icon,
  className = '',
}) => {
  return (
    <div
      className={`bg-white rounded-lg border border-gray-200 shadow-sm p-4 ${className}`}
    >
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium text-gray-500">{label}</p>
        {icon && <div className="text-gray-400">{icon}</div>}
      </div>
      <div className="mt-2 flex items-baseline gap-2">
        <p className="text-2xl font-semibold text-gray-900">{value}</p>
        {change && (
          <span
            className={`text-sm font-medium ${
              changePositive ? 'text-green-600' : 'text-red-600'
            }`}
          >
            {change}
          </span>
        )}
      </div>
    </div>
  );
};

export default Card;
