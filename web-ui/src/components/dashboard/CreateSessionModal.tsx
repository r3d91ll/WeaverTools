/**
 * CreateSessionModal component for creating new sessions.
 *
 * Provides a modal dialog with a form for entering session details
 * including name, description, and optional configuration settings.
 */
import { useState, useCallback, useEffect, useRef } from 'react';
import { createSession, type CreateSessionRequest } from '@/services/sessionApi';
import type { Session } from '@/types';
import { FieldError } from '@/components/common/ValidationError';

/**
 * CreateSessionModal component props.
 */
export interface CreateSessionModalProps {
  /** Whether the modal is open */
  isOpen: boolean;
  /** Callback when modal is closed */
  onClose: () => void;
  /** Callback when session is created successfully */
  onCreated: (session: Session) => void;
}

/**
 * Form data for creating a session.
 */
interface FormData {
  name: string;
  description: string;
}

/**
 * Form validation errors.
 */
interface FormErrors {
  name?: string;
  description?: string;
}

/**
 * Validate form data.
 */
function validateForm(data: FormData): FormErrors {
  const errors: FormErrors = {};

  if (!data.name.trim()) {
    errors.name = 'Session name is required';
  } else if (data.name.trim().length < 2) {
    errors.name = 'Session name must be at least 2 characters';
  } else if (data.name.trim().length > 100) {
    errors.name = 'Session name must be less than 100 characters';
  }

  if (data.description.length > 500) {
    errors.description = 'Description must be less than 500 characters';
  }

  return errors;
}

/**
 * Create session modal component with form validation.
 */
export const CreateSessionModal: React.FC<CreateSessionModalProps> = ({
  isOpen,
  onClose,
  onCreated,
}) => {
  // Form state
  const [formData, setFormData] = useState<FormData>({
    name: '',
    description: '',
  });
  const [errors, setErrors] = useState<FormErrors>({});
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  // Ref for initial focus
  const nameInputRef = useRef<HTMLInputElement>(null);

  // Reset form when modal opens
  useEffect(() => {
    if (isOpen) {
      setFormData({
        name: `Session ${new Date().toLocaleDateString()}`,
        description: '',
      });
      setErrors({});
      setSubmitError(null);
      // Focus name input after a short delay to ensure modal is rendered
      setTimeout(() => {
        nameInputRef.current?.focus();
        nameInputRef.current?.select();
      }, 50);
    }
  }, [isOpen]);

  // Handle escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen && !submitting) {
        onClose();
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [isOpen, submitting, onClose]);

  // Handle input change
  const handleChange = useCallback((field: keyof FormData, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
    // Clear field error when user types
    setErrors((prev) => ({ ...prev, [field]: undefined }));
    setSubmitError(null);
  }, []);

  // Handle form submission
  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();

      // Validate form
      const validationErrors = validateForm(formData);
      if (Object.keys(validationErrors).length > 0) {
        setErrors(validationErrors);
        return;
      }

      setSubmitting(true);
      setSubmitError(null);

      try {
        const request: CreateSessionRequest = {
          name: formData.name.trim(),
          description: formData.description.trim() || undefined,
        };

        const session = await createSession(request);
        onCreated(session);
      } catch (err) {
        setSubmitError(
          err instanceof Error ? err.message : 'Failed to create session'
        );
      } finally {
        setSubmitting(false);
      }
    },
    [formData, onCreated]
  );

  // Handle backdrop click
  const handleBackdropClick = useCallback(
    (e: React.MouseEvent) => {
      if (e.target === e.currentTarget && !submitting) {
        onClose();
      }
    },
    [submitting, onClose]
  );

  if (!isOpen) {
    return null;
  }

  return (
    <div
      className="fixed inset-0 z-50 overflow-y-auto"
      aria-labelledby="create-session-title"
      role="dialog"
      aria-modal="true"
    >
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-gray-900/50 transition-opacity"
        onClick={handleBackdropClick}
      />

      {/* Modal container */}
      <div className="flex min-h-full items-center justify-center p-4">
        {/* Modal panel */}
        <div className="relative w-full max-w-md transform rounded-lg bg-white shadow-xl transition-all">
          {/* Header */}
          <div className="border-b border-gray-200 px-6 py-4">
            <div className="flex items-center justify-between">
              <h2
                id="create-session-title"
                className="text-lg font-semibold text-gray-900"
              >
                Create New Session
              </h2>
              <button
                type="button"
                onClick={onClose}
                disabled={submitting}
                className="text-gray-400 hover:text-gray-500 disabled:opacity-50"
                aria-label="Close"
              >
                <svg
                  className="h-5 w-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </div>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit}>
            <div className="px-6 py-4 space-y-4">
              {/* Submit error */}
              {submitError && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                  <div className="flex items-center gap-2 text-red-600 text-sm">
                    <svg
                      className="w-4 h-4 flex-shrink-0"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                      />
                    </svg>
                    <span>{submitError}</span>
                  </div>
                </div>
              )}

              {/* Name field */}
              <div>
                <label
                  htmlFor="session-name"
                  className="block text-sm font-medium text-gray-700 mb-1"
                >
                  Session Name <span className="text-red-500">*</span>
                </label>
                <input
                  ref={nameInputRef}
                  type="text"
                  id="session-name"
                  value={formData.name}
                  onChange={(e) => handleChange('name', e.target.value)}
                  disabled={submitting}
                  className={`input w-full ${errors.name ? 'border-red-300 focus:ring-red-500 focus:border-red-500' : ''}`}
                  placeholder="Enter session name"
                  autoComplete="off"
                />
                <FieldError error={errors.name} />
              </div>

              {/* Description field */}
              <div>
                <label
                  htmlFor="session-description"
                  className="block text-sm font-medium text-gray-700 mb-1"
                >
                  Description
                </label>
                <textarea
                  id="session-description"
                  value={formData.description}
                  onChange={(e) => handleChange('description', e.target.value)}
                  disabled={submitting}
                  rows={3}
                  className={`input w-full resize-none ${errors.description ? 'border-red-300 focus:ring-red-500 focus:border-red-500' : ''}`}
                  placeholder="Describe the purpose of this session (optional)"
                />
                <div className="flex justify-between mt-1">
                  <FieldError error={errors.description} />
                  <span className="text-xs text-gray-400">
                    {formData.description.length}/500
                  </span>
                </div>
              </div>
            </div>

            {/* Footer */}
            <div className="border-t border-gray-200 px-6 py-4 flex justify-end gap-3">
              <button
                type="button"
                onClick={onClose}
                disabled={submitting}
                className="btn-secondary"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={submitting}
                className="btn-primary flex items-center gap-2"
              >
                {submitting ? (
                  <>
                    <svg
                      className="animate-spin h-4 w-4"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                        fill="none"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      />
                    </svg>
                    Creating...
                  </>
                ) : (
                  <>
                    <svg
                      className="w-4 h-4"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M12 4v16m8-8H4"
                      />
                    </svg>
                    Create Session
                  </>
                )}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default CreateSessionModal;
