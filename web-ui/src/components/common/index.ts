/**
 * Common components barrel export.
 *
 * This module exports all shared UI components for use throughout the application.
 */

// Button component
export { Button } from './Button';
export type { ButtonProps, ButtonVariant, ButtonSize } from './Button';

// Input components
export { Input, Textarea } from './Input';
export type { InputProps, InputSize, TextareaProps } from './Input';

// Modal components
export { Modal, ConfirmModal } from './Modal';
export type { ModalProps, ModalSize, ConfirmModalProps } from './Modal';

// Card components
export { Card, CardGrid, StatCard } from './Card';
export type { CardProps, CardPadding, CardGridProps, StatCardProps } from './Card';

// Spinner and loading components
export {
  Spinner,
  LoadingOverlay,
  InlineLoading,
  Skeleton,
  ContentSkeleton,
} from './Spinner';
export type {
  SpinnerProps,
  SpinnerSize,
  SpinnerColor,
  LoadingOverlayProps,
  InlineLoadingProps,
  SkeletonProps,
  ContentSkeletonProps,
} from './Spinner';

// Alert and notification components
export { Alert, Toast, Banner, InlineMessage } from './Alert';
export type {
  AlertProps,
  AlertVariant,
  ToastProps,
  BannerProps,
  InlineMessageProps,
} from './Alert';

// Validation components (existing)
export { ValidationError, FieldError } from './ValidationError';
export type {
  ValidationErrorProps,
  ValidationMessage,
  ValidationSeverity,
  FieldErrorProps,
} from './ValidationError';
