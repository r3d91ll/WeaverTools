/**
 * DeleteSessionModal component for confirming session deletion.
 *
 * Provides a confirmation dialog before deleting a session,
 * showing session details and warning about data loss.
 */
import { useState, useCallback, useEffect, useRef } from 'react';
import { deleteSession } from '@/services/sessionApi';
import type { SessionSummary } from '@/services/sessionApi';

/**
 * DeleteSessionModal component props.
 */
export interface DeleteSessionModalProps {
  /** Whether the modal is open */
  isOpen: boolean;
  /** Session to delete (null if modal is closed) */
  session: SessionSummary | null;
  /** Callback when modal is closed */
  onClose: () => void;
  /** Callback when session is deleted successfully */
  onDeleted: (sessionId: string) => void;
}

/**
 * Delete session confirmation modal component.
 */
export const DeleteSessionModal: React.FC<DeleteSessionModalProps> = ({
  isOpen,
  session,
  onClose,
  onDeleted,
}) => {
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Ref for cancel button focus
  const cancelButtonRef = useRef<HTMLButtonElement>(null);

  // Reset state when modal opens
  useEffect(() => {
    if (isOpen) {
      setError(null);
      // Focus cancel button for safety
      setTimeout(() => {
        cancelButtonRef.current?.focus();
      }, 50);
    }
  }, [isOpen]);

  // Handle escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen && !deleting) {
        onClose();
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [isOpen, deleting, onClose]);

  // Handle delete confirmation
  const handleDelete = useCallback(async () => {
    if (!session) return;

    setDeleting(true);
    setError(null);

    try {
      await deleteSession(session.id);
      onDeleted(session.id);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Failed to delete session'
      );
      setDeleting(false);
    }
  }, [session, onDeleted]);

  // Handle backdrop click
  const handleBackdropClick = useCallback(
    (e: React.MouseEvent) => {
      if (e.target === e.currentTarget && !deleting) {
        onClose();
      }
    },
    [deleting, onClose]
  );

  if (!isOpen || !session) {
    return null;
  }

  return (
    <div
      className="fixed inset-0 z-50 overflow-y-auto"
      aria-labelledby="delete-session-title"
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
          {/* Content */}
          <div className="px-6 py-6">
            {/* Icon and title */}
            <div className="flex items-start gap-4">
              <div className="flex-shrink-0 flex items-center justify-center h-12 w-12 rounded-full bg-red-100">
                <svg
                  className="h-6 w-6 text-red-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                  />
                </svg>
              </div>

              <div className="flex-1">
                <h3
                  id="delete-session-title"
                  className="text-lg font-semibold text-gray-900"
                >
                  Delete Session
                </h3>
                <p className="mt-2 text-sm text-gray-500">
                  Are you sure you want to delete this session? This action
                  cannot be undone.
                </p>

                {/* Session info */}
                <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-start justify-between">
                    <div className="min-w-0">
                      <p className="font-medium text-gray-900 truncate">
                        {session.name}
                      </p>
                      {session.description && (
                        <p className="text-sm text-gray-500 truncate mt-1">
                          {session.description}
                        </p>
                      )}
                    </div>
                    <span
                      className={`flex-shrink-0 ml-2 inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${
                        session.isActive
                          ? 'bg-green-100 text-green-800'
                          : 'bg-gray-100 text-gray-600'
                      }`}
                    >
                      {session.isActive ? 'Active' : 'Completed'}
                    </span>
                  </div>

                  {/* Stats summary */}
                  {session.stats && (
                    <div className="mt-3 pt-3 border-t border-gray-200 grid grid-cols-3 gap-2 text-xs text-gray-500">
                      <div>
                        <span className="font-medium text-gray-700">
                          {session.stats.messageCount ?? 0}
                        </span>{' '}
                        messages
                      </div>
                      <div>
                        <span className="font-medium text-gray-700">
                          {session.stats.measurementCount ?? 0}
                        </span>{' '}
                        measurements
                      </div>
                      <div>
                        <span className="font-medium text-gray-700">
                          {session.stats.conversationCount ?? 0}
                        </span>{' '}
                        conversations
                      </div>
                    </div>
                  )}
                </div>

                {/* Warning for active sessions */}
                {session.isActive && (
                  <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                    <div className="flex items-center gap-2 text-yellow-800 text-sm">
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
                          d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                        />
                      </svg>
                      <span className="font-medium">
                        This is an active session!
                      </span>
                    </div>
                    <p className="mt-1 text-xs text-yellow-700 ml-6">
                      Deleting an active session will terminate all ongoing
                      conversations.
                    </p>
                  </div>
                )}

                {/* Error message */}
                {error && (
                  <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
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
                      <span>{error}</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Footer */}
          <div className="border-t border-gray-200 px-6 py-4 flex justify-end gap-3">
            <button
              ref={cancelButtonRef}
              type="button"
              onClick={onClose}
              disabled={deleting}
              className="btn-secondary"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={handleDelete}
              disabled={deleting}
              className="px-4 py-2 bg-red-600 text-white font-medium rounded-lg hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {deleting ? (
                <>
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
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
                  Deleting...
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
                      d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                    />
                  </svg>
                  Delete Session
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DeleteSessionModal;
