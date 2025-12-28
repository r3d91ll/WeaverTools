// Package shell provides the interactive REPL for Weaver.
package shell

import (
	"errors"
	"strings"
	"testing"

	werrors "github.com/r3d91ll/weaver/pkg/errors"
)

// TestCreateExportError verifies the createExportError function.
func TestCreateExportError(t *testing.T) {
	tests := []struct {
		name          string
		command       string
		format        string
		path          string
		cause         error
		expectedCode  string
		expectedInMsg string
	}{
		{
			name:          "permission denied",
			command:       "/export_latex",
			format:        "latex",
			path:          "/root/export",
			cause:         errors.New("open /root/export/file.tex: permission denied"),
			expectedCode:  werrors.ErrExportPermissionDenied,
			expectedInMsg: "permission denied",
		},
		{
			name:          "disk space full",
			command:       "/export_csv",
			format:        "csv",
			path:          "/tmp/export",
			cause:         errors.New("write: no space left on device"),
			expectedCode:  werrors.ErrExportDiskFull,
			expectedInMsg: "disk space full",
		},
		{
			name:          "disk quota exceeded",
			command:       "/export_csv",
			format:        "csv",
			path:          "/home/user/export",
			cause:         errors.New("write: disk quota exceeded"),
			expectedCode:  werrors.ErrExportDiskFull,
			expectedInMsg: "disk space full",
		},
		{
			name:          "read-only filesystem",
			command:       "/export_figures",
			format:        "svg",
			path:          "/mnt/readonly",
			cause:         errors.New("open: read-only file system"),
			expectedCode:  werrors.ErrExportReadOnly,
			expectedInMsg: "read-only",
		},
		{
			name:          "path too long",
			command:       "/export_bibtex",
			format:        "bibtex",
			path:          strings.Repeat("a", 300),
			cause:         errors.New("open: file name too long"),
			expectedCode:  werrors.ErrExportPathTooLong,
			expectedInMsg: "path too long",
		},
		{
			name:          "invalid path",
			command:       "/export_repro",
			format:        "markdown",
			path:          "/tmp/\x00invalid",
			cause:         errors.New("open: invalid argument"),
			expectedCode:  werrors.ErrExportInvalidPath,
			expectedInMsg: "invalid path",
		},
		{
			name:          "directory not found",
			command:       "/export_latex",
			format:        "latex",
			path:          "/nonexistent/path/export",
			cause:         errors.New("mkdir: no such file or directory"),
			expectedCode:  werrors.ErrExportDirCreateFailed,
			expectedInMsg: "directory does not exist",
		},
		{
			name:          "path is directory",
			command:       "/export_csv",
			format:        "csv",
			path:          "/tmp",
			cause:         errors.New("open /tmp: is a directory"),
			expectedCode:  werrors.ErrExportWriteFailed,
			expectedInMsg: "is a directory",
		},
		{
			name:          "too many open files",
			command:       "/export_figures",
			format:        "pdf",
			path:          "/tmp/export",
			cause:         errors.New("open: too many open files"),
			expectedCode:  werrors.ErrExportWriteFailed,
			expectedInMsg: "too many open files",
		},
		{
			name:          "generic error",
			command:       "/export_all",
			format:        "all",
			path:          "/tmp/export",
			cause:         errors.New("some unknown error"),
			expectedCode:  werrors.ErrExportWriteFailed,
			expectedInMsg: "export failed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := createExportError(tt.command, tt.format, tt.path, tt.cause)

			if err == nil {
				t.Fatal("expected error, got nil")
			}

			// Check error code
			if err.Code != tt.expectedCode {
				t.Errorf("expected code %q, got %q", tt.expectedCode, err.Code)
			}

			// Check message contains expected text
			if !strings.Contains(err.Message, tt.expectedInMsg) {
				t.Errorf("expected message to contain %q, got %q", tt.expectedInMsg, err.Message)
			}

			// Check context is set
			if err.Context["command"] != tt.command {
				t.Errorf("expected command context %q, got %q", tt.command, err.Context["command"])
			}
			if err.Context["format"] != tt.format {
				t.Errorf("expected format context %q, got %q", tt.format, err.Context["format"])
			}
			if err.Context["path"] != tt.path {
				t.Errorf("expected path context %q, got %q", tt.path, err.Context["path"])
			}

			// Check cause is wrapped
			if err.Cause == nil {
				t.Error("expected cause to be set")
			}

			// Check suggestions are provided
			if len(err.Suggestions) == 0 {
				t.Error("expected at least one suggestion")
			}
		})
	}
}

// TestCreateExportNoDataError verifies the createExportNoDataError function.
func TestCreateExportNoDataError(t *testing.T) {
	tests := []struct {
		name    string
		command string
		format  string
	}{
		{
			name:    "latex",
			command: "/export_latex",
			format:  "latex",
		},
		{
			name:    "csv",
			command: "/export_csv",
			format:  "csv",
		},
		{
			name:    "figures",
			command: "/export_figures",
			format:  "figures",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := createExportNoDataError(tt.command, tt.format)

			if err == nil {
				t.Fatal("expected error, got nil")
			}

			// Check error code
			if err.Code != werrors.ErrExportNoData {
				t.Errorf("expected code %q, got %q", werrors.ErrExportNoData, err.Code)
			}

			// Check message
			if !strings.Contains(err.Message, "no data available") {
				t.Errorf("expected message to contain 'no data available', got %q", err.Message)
			}

			// Check context
			if err.Context["command"] != tt.command {
				t.Errorf("expected command context %q, got %q", tt.command, err.Context["command"])
			}
			if err.Context["format"] != tt.format {
				t.Errorf("expected format context %q, got %q", tt.format, err.Context["format"])
			}

			// Check suggestions
			if len(err.Suggestions) < 2 {
				t.Errorf("expected at least 2 suggestions, got %d", len(err.Suggestions))
			}

			// Check for helpful suggestions
			hasMeasurementSuggestion := false
			for _, s := range err.Suggestions {
				if strings.Contains(s, "/extract") || strings.Contains(s, "/analyze") {
					hasMeasurementSuggestion = true
					break
				}
			}
			if !hasMeasurementSuggestion {
				t.Error("expected suggestion about /extract or /analyze commands")
			}
		})
	}
}

// TestCreateExportDirError verifies the createExportDirError function.
func TestCreateExportDirError(t *testing.T) {
	tests := []struct {
		name          string
		command       string
		format        string
		path          string
		cause         error
		expectedCode  string
		expectedInMsg string
	}{
		{
			name:          "permission denied",
			command:       "/export_latex",
			format:        "latex",
			path:          "/root/export/latex",
			cause:         errors.New("mkdir: permission denied"),
			expectedCode:  werrors.ErrExportDirCreateFailed,
			expectedInMsg: "permission denied",
		},
		{
			name:          "disk full",
			command:       "/export_csv",
			format:        "csv",
			path:          "/tmp/export/csv",
			cause:         errors.New("mkdir: no space left on device"),
			expectedCode:  werrors.ErrExportDiskFull,
			expectedInMsg: "disk full",
		},
		{
			name:          "file exists at path",
			command:       "/export_figures",
			format:        "svg",
			path:          "/tmp/export/svg",
			cause:         errors.New("mkdir: file exists"),
			expectedCode:  werrors.ErrExportDirCreateFailed,
			expectedInMsg: "path exists as file",
		},
		{
			name:          "generic error",
			command:       "/export_bibtex",
			format:        "bibtex",
			path:          "/tmp/export/bibtex",
			cause:         errors.New("some unknown mkdir error"),
			expectedCode:  werrors.ErrExportDirCreateFailed,
			expectedInMsg: "failed to create export directory",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := createExportDirError(tt.command, tt.format, tt.path, tt.cause)

			if err == nil {
				t.Fatal("expected error, got nil")
			}

			// Check error code
			if err.Code != tt.expectedCode {
				t.Errorf("expected code %q, got %q", tt.expectedCode, err.Code)
			}

			// Check message contains expected text
			if !strings.Contains(err.Message, tt.expectedInMsg) {
				t.Errorf("expected message to contain %q, got %q", tt.expectedInMsg, err.Message)
			}

			// Check context
			if err.Context["command"] != tt.command {
				t.Errorf("expected command context %q, got %q", tt.command, err.Context["command"])
			}

			// Check cause is wrapped
			if err.Cause == nil {
				t.Error("expected cause to be set")
			}

			// Check suggestions
			if len(err.Suggestions) == 0 {
				t.Error("expected at least one suggestion")
			}
		})
	}
}

// TestCreateExportMultipleError verifies the createExportMultipleError function.
func TestCreateExportMultipleError(t *testing.T) {
	tests := []struct {
		name          string
		failedFormats []string
		firstError    error
	}{
		{
			name:          "single format failed",
			failedFormats: []string{"LaTeX"},
			firstError:    errors.New("permission denied"),
		},
		{
			name:          "multiple formats failed",
			failedFormats: []string{"LaTeX", "CSV", "Figures"},
			firstError:    errors.New("disk full"),
		},
		{
			name:          "all formats failed",
			failedFormats: []string{"LaTeX", "CSV", "Figures", "BibTeX", "Reproducibility"},
			firstError:    errors.New("read-only filesystem"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := createExportMultipleError(tt.failedFormats, tt.firstError)

			if err == nil {
				t.Fatal("expected error, got nil")
			}

			// Check error code
			if err.Code != werrors.ErrExportFailed {
				t.Errorf("expected code %q, got %q", werrors.ErrExportFailed, err.Code)
			}

			// Check message
			if !strings.Contains(err.Message, "exports failed") {
				t.Errorf("expected message to contain 'exports failed', got %q", err.Message)
			}

			// Check context
			if err.Context["command"] != "/export_all" {
				t.Errorf("expected command context '/export_all', got %q", err.Context["command"])
			}

			// Check failed formats are listed
			for _, format := range tt.failedFormats {
				if !strings.Contains(err.Context["failed_formats"], format) {
					t.Errorf("expected failed_formats to contain %q", format)
				}
			}

			// Check cause is wrapped
			if err.Cause == nil {
				t.Error("expected cause to be set")
			}

			// Check suggestions
			if len(err.Suggestions) < 2 {
				t.Errorf("expected at least 2 suggestions, got %d", len(err.Suggestions))
			}
		})
	}
}

// TestExportErrorCategories verifies error codes are in the IO category.
func TestExportErrorCategories(t *testing.T) {
	codes := []string{
		werrors.ErrExportFailed,
		werrors.ErrExportNoData,
		werrors.ErrExportDirCreateFailed,
		werrors.ErrExportWriteFailed,
		werrors.ErrExportPermissionDenied,
		werrors.ErrExportDiskFull,
		werrors.ErrExportReadOnly,
		werrors.ErrExportPathTooLong,
		werrors.ErrExportInvalidPath,
	}

	for _, code := range codes {
		t.Run(code, func(t *testing.T) {
			category := werrors.CodeCategory(code)
			if category != werrors.CategoryIO {
				t.Errorf("expected category %q for code %q, got %q",
					werrors.CategoryIO, code, category)
			}

			// Also verify IsExportCode works
			if !werrors.IsExportCode(code) {
				t.Errorf("expected IsExportCode(%q) to return true", code)
			}
		})
	}
}

// TestExportErrorFormatting verifies errors format correctly for display.
func TestExportErrorFormatting(t *testing.T) {
	err := createExportError("/export_latex", "latex", "/tmp/test", errors.New("permission denied"))

	// Get formatted output
	formatted := werrors.Sprint(err)

	// Check key elements are present
	if !strings.Contains(formatted, werrors.ErrExportPermissionDenied) {
		t.Error("formatted output should contain error code")
	}
	if !strings.Contains(formatted, "permission denied") {
		t.Error("formatted output should contain error message")
	}
	if !strings.Contains(formatted, "/export_latex") {
		t.Error("formatted output should contain command")
	}
	if !strings.Contains(formatted, "→") {
		t.Error("formatted output should contain suggestion arrows")
	}
}
