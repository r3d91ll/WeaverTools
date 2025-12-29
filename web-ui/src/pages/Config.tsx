/**
 * Config page - configuration management and YAML editor.
 *
 * Provides form-based and raw YAML editing for Weaver configuration.
 */
import { useCallback } from 'react';
import { ConfigEditor } from '@/components/config';
import type { Config as ConfigType } from '@/types';

/**
 * Config page component.
 */
export const Config: React.FC = () => {
  /** Handle config save */
  const handleSave = useCallback((config: ConfigType) => {
    // Config was saved successfully - could add toast notification here
  }, []);

  /** Handle validation errors */
  const handleValidationError = useCallback((errors: string[]) => {
    // Validation errors are displayed by ConfigEditor
    // Could add analytics tracking here
  }, []);

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Configuration</h1>
        <p className="mt-2 text-gray-600">
          Manage Weaver configuration, agents, and backend settings
        </p>
      </div>

      {/* Config Editor */}
      <ConfigEditor
        onSave={handleSave}
        onValidationError={handleValidationError}
      />
    </div>
  );
};

export default Config;
