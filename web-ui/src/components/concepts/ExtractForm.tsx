/**
 * ExtractForm component for adding samples to concepts.
 *
 * Provides a form to create new concepts or add samples to existing ones.
 * Supports specifying sample content, model, and batch operations.
 */
import { useState, useCallback, useEffect } from 'react';
import { addSample } from '@/services/conceptApi';
import { listConcepts } from '@/services/conceptApi';
import type { ConceptStats } from '@/types/concept';

/**
 * ExtractForm component props.
 */
export interface ExtractFormProps {
  /** Pre-populated concept name (for adding to existing concept) */
  conceptName?: string;
  /** Callback when samples are successfully added */
  onSamplesAdded?: (conceptName: string, count: number) => void;
  /** Callback when the form is cancelled */
  onCancel?: () => void;
  /** Whether to show as a modal overlay */
  isModal?: boolean;
  /** Whether the form is loading (external control) */
  externalLoading?: boolean;
  /** Optional className for the form container */
  className?: string;
}

/**
 * Form state for sample extraction.
 */
interface FormState {
  conceptName: string;
  content: string;
  model: string;
  isNewConcept: boolean;
}

/**
 * Validation errors.
 */
interface ValidationErrors {
  conceptName?: string;
  content?: string;
}

/**
 * Validate concept name format.
 */
function isValidConceptName(name: string): boolean {
  // Concept names should be alphanumeric with underscores/hyphens, 2-50 chars
  return /^[a-zA-Z][a-zA-Z0-9_-]{1,49}$/.test(name);
}

/**
 * ExtractForm component for adding samples to concepts.
 */
export const ExtractForm: React.FC<ExtractFormProps> = ({
  conceptName: initialConceptName,
  onSamplesAdded,
  onCancel,
  isModal = false,
  externalLoading = false,
  className = '',
}) => {
  // Form state
  const [formState, setFormState] = useState<FormState>({
    conceptName: initialConceptName || '',
    content: '',
    model: '',
    isNewConcept: !initialConceptName,
  });

  // UI state
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [validationErrors, setValidationErrors] = useState<ValidationErrors>({});
  const [existingConcepts, setExistingConcepts] = useState<ConceptStats[]>([]);
  const [showConceptSuggestions, setShowConceptSuggestions] = useState(false);
  const [addedCount, setAddedCount] = useState(0);

  // Load existing concepts for suggestions
  useEffect(() => {
    const loadConcepts = async () => {
      try {
        const concepts = await listConcepts();
        setExistingConcepts(concepts);
      } catch {
        // Ignore errors loading concepts - just won't show suggestions
      }
    };
    loadConcepts();
  }, []);

  // Filter concept suggestions based on input
  const filteredConcepts = existingConcepts.filter((c) =>
    c.name.toLowerCase().includes(formState.conceptName.toLowerCase())
  ).slice(0, 5);

  /** Validate form */
  const validateForm = useCallback((): boolean => {
    const errors: ValidationErrors = {};

    // Validate concept name
    if (!formState.conceptName.trim()) {
      errors.conceptName = 'Concept name is required';
    } else if (!isValidConceptName(formState.conceptName.trim())) {
      errors.conceptName =
        'Concept name must start with a letter and contain only letters, numbers, underscores, or hyphens (2-50 chars)';
    }

    // Validate content
    if (!formState.content.trim()) {
      errors.content = 'Sample content is required';
    } else if (formState.content.trim().length < 3) {
      errors.content = 'Sample content must be at least 3 characters';
    } else if (formState.content.trim().length > 10000) {
      errors.content = 'Sample content must not exceed 10,000 characters';
    }

    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  }, [formState.conceptName, formState.content]);

  /** Handle form submission */
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccess(null);

    if (!validateForm()) {
      return;
    }

    setIsSubmitting(true);

    try {
      await addSample(formState.conceptName.trim(), {
        content: formState.content.trim(),
        model: formState.model.trim() || undefined,
      });

      const newCount = addedCount + 1;
      setAddedCount(newCount);
      setSuccess(`Sample added to "${formState.conceptName}" (${newCount} total this session)`);

      // Clear content for next sample but keep concept name
      setFormState((prev) => ({
        ...prev,
        content: '',
      }));

      onSamplesAdded?.(formState.conceptName.trim(), 1);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to add sample');
    } finally {
      setIsSubmitting(false);
    }
  };

  /** Handle concept name change */
  const handleConceptNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setFormState((prev) => ({
      ...prev,
      conceptName: value,
      isNewConcept: !existingConcepts.some((c) => c.name === value),
    }));
    setValidationErrors((prev) => ({ ...prev, conceptName: undefined }));
    setShowConceptSuggestions(true);
  };

  /** Handle content change */
  const handleContentChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setFormState((prev) => ({ ...prev, content: e.target.value }));
    setValidationErrors((prev) => ({ ...prev, content: undefined }));
  };

  /** Handle model change */
  const handleModelChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormState((prev) => ({ ...prev, model: e.target.value }));
  };

  /** Handle concept suggestion click */
  const handleSelectConcept = (concept: ConceptStats) => {
    setFormState((prev) => ({
      ...prev,
      conceptName: concept.name,
      isNewConcept: false,
    }));
    setShowConceptSuggestions(false);
    setValidationErrors((prev) => ({ ...prev, conceptName: undefined }));
  };

  /** Handle cancel */
  const handleCancel = () => {
    onCancel?.();
  };

  /** Handle keyboard events */
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape' && onCancel) {
      handleCancel();
    }
  };

  const isLoading = isSubmitting || externalLoading;
  const conceptExists = existingConcepts.some(
    (c) => c.name.toLowerCase() === formState.conceptName.toLowerCase()
  );

  const formContent = (
    <form onSubmit={handleSubmit} onKeyDown={handleKeyDown} className="space-y-4">
      {/* Concept Name */}
      <div className="relative">
        <label htmlFor="conceptName" className="block text-sm font-medium text-gray-700 mb-1">
          Concept Name
          {!formState.isNewConcept && conceptExists && (
            <span className="ml-2 text-xs text-green-600 font-normal">
              (existing concept)
            </span>
          )}
          {formState.isNewConcept && formState.conceptName.trim() && isValidConceptName(formState.conceptName.trim()) && (
            <span className="ml-2 text-xs text-blue-600 font-normal">
              (new concept)
            </span>
          )}
        </label>
        <input
          id="conceptName"
          type="text"
          value={formState.conceptName}
          onChange={handleConceptNameChange}
          onFocus={() => setShowConceptSuggestions(true)}
          onBlur={() => setTimeout(() => setShowConceptSuggestions(false), 200)}
          placeholder="e.g., happiness, technical_writing, code_review"
          disabled={isLoading || !!initialConceptName}
          className={`input w-full ${
            validationErrors.conceptName
              ? 'border-red-300 focus:border-red-500 focus:ring-red-500'
              : ''
          }`}
          autoComplete="off"
        />
        {validationErrors.conceptName && (
          <p className="mt-1 text-sm text-red-600">{validationErrors.conceptName}</p>
        )}

        {/* Concept suggestions dropdown */}
        {showConceptSuggestions && filteredConcepts.length > 0 && formState.conceptName && (
          <div className="absolute z-10 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg max-h-48 overflow-auto">
            {filteredConcepts.map((concept) => (
              <button
                key={concept.name}
                type="button"
                onClick={() => handleSelectConcept(concept)}
                className="w-full px-4 py-2 text-left hover:bg-gray-50 flex items-center justify-between"
              >
                <span className="font-medium text-gray-900">{concept.name}</span>
                <span className="text-xs text-gray-500">
                  {concept.sampleCount} sample{concept.sampleCount !== 1 ? 's' : ''}
                </span>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Sample Content */}
      <div>
        <label htmlFor="content" className="block text-sm font-medium text-gray-700 mb-1">
          Sample Content
          <span className="ml-1 text-xs text-gray-400 font-normal">
            ({formState.content.length}/10000)
          </span>
        </label>
        <textarea
          id="content"
          value={formState.content}
          onChange={handleContentChange}
          placeholder="Enter the text sample that represents this concept..."
          disabled={isLoading}
          rows={4}
          className={`input w-full resize-y min-h-[100px] ${
            validationErrors.content
              ? 'border-red-300 focus:border-red-500 focus:ring-red-500'
              : ''
          }`}
        />
        {validationErrors.content && (
          <p className="mt-1 text-sm text-red-600">{validationErrors.content}</p>
        )}
      </div>

      {/* Model (optional) */}
      <div>
        <label htmlFor="model" className="block text-sm font-medium text-gray-700 mb-1">
          Model
          <span className="ml-1 text-xs text-gray-400 font-normal">(optional)</span>
        </label>
        <input
          id="model"
          type="text"
          value={formState.model}
          onChange={handleModelChange}
          placeholder="e.g., llama3.2:3b, mistral:7b"
          disabled={isLoading}
          className="input w-full"
        />
        <p className="mt-1 text-xs text-gray-500">
          The model used to generate hidden states for this sample.
        </p>
      </div>

      {/* Error message */}
      {error && (
        <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-center gap-2">
            <svg
              className="w-5 h-5 text-red-500 flex-shrink-0"
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                clipRule="evenodd"
              />
            </svg>
            <p className="text-sm text-red-700">{error}</p>
          </div>
        </div>
      )}

      {/* Success message */}
      {success && (
        <div className="p-3 bg-green-50 border border-green-200 rounded-lg">
          <div className="flex items-center gap-2">
            <svg
              className="w-5 h-5 text-green-500 flex-shrink-0"
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                clipRule="evenodd"
              />
            </svg>
            <p className="text-sm text-green-700">{success}</p>
          </div>
        </div>
      )}

      {/* Action buttons */}
      <div className="flex items-center justify-end gap-3 pt-2">
        {onCancel && (
          <button
            type="button"
            onClick={handleCancel}
            disabled={isLoading}
            className="btn-secondary"
          >
            {addedCount > 0 ? 'Done' : 'Cancel'}
          </button>
        )}
        <button
          type="submit"
          disabled={isLoading || !formState.conceptName.trim() || !formState.content.trim()}
          className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isSubmitting ? (
            <span className="flex items-center gap-2">
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
              Adding Sample...
            </span>
          ) : (
            <>
              <svg
                className="w-4 h-4 mr-1.5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 6v6m0 0v6m0-6h6m-6 0H6"
                />
              </svg>
              Add Sample
            </>
          )}
        </button>
      </div>
    </form>
  );

  // Render as modal or inline
  if (isModal) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
        <div
          className={`bg-white rounded-xl shadow-xl max-w-lg w-full mx-4 p-6 ${className}`}
          onClick={(e) => e.stopPropagation()}
        >
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-900">Add Sample to Concept</h2>
            {onCancel && (
              <button
                type="button"
                onClick={handleCancel}
                className="text-gray-400 hover:text-gray-600 transition-colors"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            )}
          </div>
          {formContent}
        </div>
      </div>
    );
  }

  return (
    <div className={`card ${className}`}>
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Add Sample to Concept</h3>
      {formContent}
    </div>
  );
};

export default ExtractForm;
