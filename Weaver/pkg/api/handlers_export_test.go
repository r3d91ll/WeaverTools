package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// -----------------------------------------------------------------------------
// ExportHandler Tests
// -----------------------------------------------------------------------------

func TestNewExportHandler(t *testing.T) {
	t.Run("creates handler with session store", func(t *testing.T) {
		store := NewMemorySessionStore()
		h := NewExportHandler(store)
		if h == nil {
			t.Error("Expected non-nil handler")
		}
	})

	t.Run("creates handler with nil store", func(t *testing.T) {
		h := NewExportHandler(nil)
		if h == nil {
			t.Error("Expected non-nil handler even with nil store")
		}
	})
}

func TestExportHandler_RegisterRoutes(t *testing.T) {
	h := NewExportHandler(nil)
	router := NewRouter()
	h.RegisterRoutes(router)

	// Test that routes are registered by checking if they respond
	tests := []struct {
		method string
		path   string
	}{
		{http.MethodPost, "/api/export/latex"},
		{http.MethodPost, "/api/export/csv"},
		{http.MethodPost, "/api/export/pdf"},
		{http.MethodPost, "/api/export/bibtex"},
	}

	for _, tt := range tests {
		t.Run(tt.method+" "+tt.path, func(t *testing.T) {
			body := bytes.NewReader([]byte("{}"))
			req := httptest.NewRequest(tt.method, tt.path, body)
			req.Header.Set("Content-Type", "application/json")
			rec := httptest.NewRecorder()

			router.ServeHTTP(rec, req)

			// Should not get 404
			if rec.Code == http.StatusNotFound {
				t.Errorf("Route %s %s not found", tt.method, tt.path)
			}
		})
	}
}

// -----------------------------------------------------------------------------
// LaTeX Export Tests
// -----------------------------------------------------------------------------

func TestExportHandler_ExportLaTeX(t *testing.T) {
	h := NewExportHandler(nil)
	router := NewRouter()
	h.RegisterRoutes(router)

	t.Run("exports valid measurement data to LaTeX", func(t *testing.T) {
		requestBody := ExportRequest{
			Measurements: []ExportMeasurement{
				{
					Turn:       1,
					Sender:     "Alice",
					Receiver:   "Bob",
					DEff:       128,
					Beta:       1.75,
					Alignment:  0.85,
					CPair:      0.72,
					BetaStatus: "optimal",
				},
				{
					Turn:       2,
					Sender:     "Bob",
					Receiver:   "Alice",
					DEff:       100,
					Beta:       2.10,
					Alignment:  0.78,
					CPair:      0.65,
					BetaStatus: "monitor",
				},
			},
		}

		bodyBytes, _ := json.Marshal(requestBody)
		req := httptest.NewRequest(http.MethodPost, "/api/export/latex", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", rec.Code)
			t.Logf("Response: %s", rec.Body.String())
		}

		resp := parseAPIResponse(t, rec.Body)
		if !resp.Success {
			t.Error("Expected success response")
		}

		data := resp.Data.(map[string]interface{})
		content := data["content"].(string)

		// Verify LaTeX structure
		if !strings.Contains(content, "\\begin{tabular}") {
			t.Error("Expected tabular environment in LaTeX output")
		}
		if !strings.Contains(content, "Alice") {
			t.Error("Expected sender name in output")
		}
		if !strings.Contains(content, "Bob") {
			t.Error("Expected receiver name in output")
		}
		if !strings.Contains(content, "$D_{eff}$") || !strings.Contains(content, "$\\beta$") {
			t.Error("Expected metric headers in output")
		}

		// Verify format and filename
		if format := data["format"].(string); format != "latex" {
			t.Errorf("Expected format 'latex', got %s", format)
		}
		if filename := data["filename"].(string); filename != "measurements.tex" {
			t.Errorf("Expected filename 'measurements.tex', got %s", filename)
		}
		if mimeType := data["mimeType"].(string); mimeType != "application/x-latex" {
			t.Errorf("Expected mimeType 'application/x-latex', got %s", mimeType)
		}
	})

	t.Run("exports with booktabs style", func(t *testing.T) {
		requestBody := ExportRequest{
			Measurements: []ExportMeasurement{
				{Turn: 1, Sender: "A", Receiver: "B", DEff: 100, Beta: 1.5, Alignment: 0.8, CPair: 0.7},
			},
			Options: ExportOptions{
				Style: "booktabs",
			},
		}

		bodyBytes, _ := json.Marshal(requestBody)
		req := httptest.NewRequest(http.MethodPost, "/api/export/latex", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		data := resp.Data.(map[string]interface{})
		content := data["content"].(string)

		// Booktabs style uses \toprule, \midrule, \bottomrule
		if !strings.Contains(content, "\\toprule") {
			t.Error("Expected booktabs \\toprule in output")
		}
	})

	t.Run("includes summary when requested", func(t *testing.T) {
		requestBody := ExportRequest{
			Measurements: []ExportMeasurement{
				{Turn: 1, Sender: "A", Receiver: "B", DEff: 100, Beta: 1.5, Alignment: 0.8, CPair: 0.7},
				{Turn: 2, Sender: "B", Receiver: "A", DEff: 120, Beta: 1.8, Alignment: 0.75, CPair: 0.65},
			},
			Options: ExportOptions{
				IncludeSummary: true,
			},
		}

		bodyBytes, _ := json.Marshal(requestBody)
		req := httptest.NewRequest(http.MethodPost, "/api/export/latex", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		data := resp.Data.(map[string]interface{})

		// Should include stats in response
		if data["stats"] == nil {
			t.Error("Expected stats in response when includeSummary is true")
		}
	})

	t.Run("returns error for empty measurements", func(t *testing.T) {
		requestBody := ExportRequest{
			Measurements: []ExportMeasurement{},
		}

		bodyBytes, _ := json.Marshal(requestBody)
		req := httptest.NewRequest(http.MethodPost, "/api/export/latex", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusBadRequest {
			t.Errorf("Expected status 400, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if resp.Success {
			t.Error("Expected error response")
		}
		if resp.Error == nil || resp.Error.Code != "no_data" {
			t.Error("Expected error code 'no_data'")
		}
	})

	t.Run("returns error for invalid JSON", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodPost, "/api/export/latex", bytes.NewReader([]byte("invalid json")))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusBadRequest {
			t.Errorf("Expected status 400, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if resp.Error == nil || resp.Error.Code != "invalid_json" {
			t.Error("Expected error code 'invalid_json'")
		}
	})
}

// -----------------------------------------------------------------------------
// CSV Export Tests
// -----------------------------------------------------------------------------

func TestExportHandler_ExportCSV(t *testing.T) {
	h := NewExportHandler(nil)
	router := NewRouter()
	h.RegisterRoutes(router)

	t.Run("exports valid measurement data to CSV", func(t *testing.T) {
		requestBody := ExportRequest{
			Measurements: []ExportMeasurement{
				{
					Turn:       1,
					Sender:     "Alice",
					Receiver:   "Bob",
					DEff:       128,
					Beta:       1.75,
					Alignment:  0.85,
					CPair:      0.72,
					BetaStatus: "optimal",
				},
			},
		}

		bodyBytes, _ := json.Marshal(requestBody)
		req := httptest.NewRequest(http.MethodPost, "/api/export/csv", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", rec.Code)
			t.Logf("Response: %s", rec.Body.String())
		}

		resp := parseAPIResponse(t, rec.Body)
		if !resp.Success {
			t.Error("Expected success response")
		}

		data := resp.Data.(map[string]interface{})
		content := data["content"].(string)

		// Verify CSV structure
		if !strings.Contains(content, "d_eff") {
			t.Error("Expected d_eff header in CSV output")
		}
		if !strings.Contains(content, "beta") {
			t.Error("Expected beta header in CSV output")
		}
		if !strings.Contains(content, "Alice") {
			t.Error("Expected sender name in output")
		}

		// Verify format and filename
		if format := data["format"].(string); format != "csv" {
			t.Errorf("Expected format 'csv', got %s", format)
		}
		if filename := data["filename"].(string); filename != "measurements.csv" {
			t.Errorf("Expected filename 'measurements.csv', got %s", filename)
		}
		if mimeType := data["mimeType"].(string); mimeType != "text/csv" {
			t.Errorf("Expected mimeType 'text/csv', got %s", mimeType)
		}
	})

	t.Run("exports with TSV dialect", func(t *testing.T) {
		requestBody := ExportRequest{
			Measurements: []ExportMeasurement{
				{Turn: 1, Sender: "A", Receiver: "B", DEff: 100, Beta: 1.5, Alignment: 0.8, CPair: 0.7},
			},
			Options: ExportOptions{
				Dialect: "tsv",
			},
		}

		bodyBytes, _ := json.Marshal(requestBody)
		req := httptest.NewRequest(http.MethodPost, "/api/export/csv", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		data := resp.Data.(map[string]interface{})
		content := data["content"].(string)

		// TSV uses tabs instead of commas
		if !strings.Contains(content, "\t") {
			t.Error("Expected tab separators in TSV output")
		}
	})

	t.Run("returns error for empty measurements", func(t *testing.T) {
		requestBody := ExportRequest{
			Measurements: []ExportMeasurement{},
		}

		bodyBytes, _ := json.Marshal(requestBody)
		req := httptest.NewRequest(http.MethodPost, "/api/export/csv", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusBadRequest {
			t.Errorf("Expected status 400, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if resp.Error == nil || resp.Error.Code != "no_data" {
			t.Error("Expected error code 'no_data'")
		}
	})
}

// -----------------------------------------------------------------------------
// PDF Export Tests
// -----------------------------------------------------------------------------

func TestExportHandler_ExportPDF(t *testing.T) {
	h := NewExportHandler(nil)
	router := NewRouter()
	h.RegisterRoutes(router)

	t.Run("exports valid measurement data to PDF (LaTeX document)", func(t *testing.T) {
		requestBody := ExportRequest{
			Measurements: []ExportMeasurement{
				{
					Turn:       1,
					Sender:     "Alice",
					Receiver:   "Bob",
					DEff:       128,
					Beta:       1.75,
					Alignment:  0.85,
					CPair:      0.72,
				},
			},
			Options: ExportOptions{
				Caption: "My Test Results",
			},
		}

		bodyBytes, _ := json.Marshal(requestBody)
		req := httptest.NewRequest(http.MethodPost, "/api/export/pdf", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", rec.Code)
			t.Logf("Response: %s", rec.Body.String())
		}

		resp := parseAPIResponse(t, rec.Body)
		if !resp.Success {
			t.Error("Expected success response")
		}

		data := resp.Data.(map[string]interface{})
		content := data["content"].(string)

		// Verify it's a complete LaTeX document
		if !strings.Contains(content, "\\documentclass") {
			t.Error("Expected \\documentclass in PDF LaTeX output")
		}
		if !strings.Contains(content, "\\begin{document}") {
			t.Error("Expected \\begin{document} in PDF LaTeX output")
		}
		if !strings.Contains(content, "\\end{document}") {
			t.Error("Expected \\end{document} in PDF LaTeX output")
		}
		if !strings.Contains(content, "booktabs") {
			t.Error("Expected booktabs package in PDF LaTeX output")
		}

		// Verify format
		if format := data["format"].(string); format != "pdf" {
			t.Errorf("Expected format 'pdf', got %s", format)
		}
	})

	t.Run("returns error for empty measurements", func(t *testing.T) {
		requestBody := ExportRequest{
			Measurements: []ExportMeasurement{},
		}

		bodyBytes, _ := json.Marshal(requestBody)
		req := httptest.NewRequest(http.MethodPost, "/api/export/pdf", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusBadRequest {
			t.Errorf("Expected status 400, got %d", rec.Code)
		}
	})
}

// -----------------------------------------------------------------------------
// BibTeX Export Tests
// -----------------------------------------------------------------------------

func TestExportHandler_ExportBibTeX(t *testing.T) {
	h := NewExportHandler(nil)
	router := NewRouter()
	h.RegisterRoutes(router)

	t.Run("exports valid BibTeX entry", func(t *testing.T) {
		requestBody := ExportRequest{
			SessionID: "test-session-123",
			Options: ExportOptions{
				Title:    "My Research Dataset",
				Author:   "Test Author",
				Year:     "2024",
				Keywords: "AI, conveyance, measurement",
			},
		}

		bodyBytes, _ := json.Marshal(requestBody)
		req := httptest.NewRequest(http.MethodPost, "/api/export/bibtex", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", rec.Code)
			t.Logf("Response: %s", rec.Body.String())
		}

		resp := parseAPIResponse(t, rec.Body)
		if !resp.Success {
			t.Error("Expected success response")
		}

		data := resp.Data.(map[string]interface{})
		content := data["content"].(string)

		// Verify BibTeX structure
		if !strings.Contains(content, "@misc{") {
			t.Error("Expected @misc{ in BibTeX output")
		}
		if !strings.Contains(content, "title = {My Research Dataset}") {
			t.Error("Expected title in BibTeX output")
		}
		if !strings.Contains(content, "author = {Test Author}") {
			t.Error("Expected author in BibTeX output")
		}
		if !strings.Contains(content, "year = {2024}") {
			t.Error("Expected year in BibTeX output")
		}
		if !strings.Contains(content, "keywords = {AI, conveyance, measurement}") {
			t.Error("Expected keywords in BibTeX output")
		}

		// Verify format and filename
		if format := data["format"].(string); format != "bibtex" {
			t.Errorf("Expected format 'bibtex', got %s", format)
		}
		if filename := data["filename"].(string); filename != "dataset.bib" {
			t.Errorf("Expected filename 'dataset.bib', got %s", filename)
		}
		if mimeType := data["mimeType"].(string); mimeType != "application/x-bibtex" {
			t.Errorf("Expected mimeType 'application/x-bibtex', got %s", mimeType)
		}
	})

	t.Run("exports with default values", func(t *testing.T) {
		requestBody := ExportRequest{}

		bodyBytes, _ := json.Marshal(requestBody)
		req := httptest.NewRequest(http.MethodPost, "/api/export/bibtex", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		data := resp.Data.(map[string]interface{})
		content := data["content"].(string)

		// Should have default title
		if !strings.Contains(content, "Multi-Agent Conveyance Measurement Dataset") {
			t.Error("Expected default title in BibTeX output")
		}
		// Should have default author
		if !strings.Contains(content, "WeaverTools") {
			t.Error("Expected default author in BibTeX output")
		}
	})

	t.Run("includes measurement count when measurements provided", func(t *testing.T) {
		requestBody := ExportRequest{
			Measurements: []ExportMeasurement{
				{Turn: 1, Sender: "A", Receiver: "B", DEff: 100, Beta: 1.5, Alignment: 0.8, CPair: 0.7},
				{Turn: 2, Sender: "B", Receiver: "A", DEff: 120, Beta: 1.8, Alignment: 0.75, CPair: 0.65},
				{Turn: 3, Sender: "A", Receiver: "B", DEff: 110, Beta: 1.6, Alignment: 0.82, CPair: 0.68},
			},
		}

		bodyBytes, _ := json.Marshal(requestBody)
		req := httptest.NewRequest(http.MethodPost, "/api/export/bibtex", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		data := resp.Data.(map[string]interface{})
		content := data["content"].(string)

		// Should include measurement count
		if !strings.Contains(content, "3 measurements") {
			t.Error("Expected measurement count in BibTeX note")
		}
	})

	t.Run("escapes special characters in BibTeX", func(t *testing.T) {
		requestBody := ExportRequest{
			Options: ExportOptions{
				Title: "Test & Special % Characters",
			},
		}

		bodyBytes, _ := json.Marshal(requestBody)
		req := httptest.NewRequest(http.MethodPost, "/api/export/bibtex", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		data := resp.Data.(map[string]interface{})
		content := data["content"].(string)

		// Special characters should be escaped
		if !strings.Contains(content, "\\&") {
			t.Error("Expected escaped ampersand in BibTeX output")
		}
		if !strings.Contains(content, "\\%") {
			t.Error("Expected escaped percent in BibTeX output")
		}
	})
}

// -----------------------------------------------------------------------------
// Helper Function Tests
// -----------------------------------------------------------------------------

func TestConvertToMeasurementRows(t *testing.T) {
	measurements := []ExportMeasurement{
		{
			Turn:       1,
			Sender:     "Alice",
			Receiver:   "Bob",
			DEff:       100,
			Beta:       1.5,
			Alignment:  0.8,
			CPair:      0.7,
			BetaStatus: "optimal",
		},
		{
			Turn:       2,
			Sender:     "Bob",
			Receiver:   "Alice",
			DEff:       120,
			Beta:       1.8,
			Alignment:  0.75,
			CPair:      0.65,
			BetaStatus: "monitor",
		},
	}

	rows := convertToMeasurementRows(measurements)

	if len(rows) != 2 {
		t.Errorf("Expected 2 rows, got %d", len(rows))
	}

	// Check first row
	if rows[0].Turn != 1 {
		t.Errorf("Expected turn 1, got %d", rows[0].Turn)
	}
	if rows[0].Sender != "Alice" {
		t.Errorf("Expected sender 'Alice', got %s", rows[0].Sender)
	}
	if rows[0].DEff != 100 {
		t.Errorf("Expected DEff 100, got %d", rows[0].DEff)
	}
}

func TestConvertToCSVMeasurements(t *testing.T) {
	measurements := []ExportMeasurement{
		{
			Turn:           1,
			Sender:         "Alice",
			Receiver:       "Bob",
			DEff:           100,
			Beta:           1.5,
			Alignment:      0.8,
			CPair:          0.7,
			BetaStatus:     "optimal",
			ID:             "test-id",
			SessionID:      "session-1",
			ConversationID: "conv-1",
			SenderID:       "alice-id",
			ReceiverID:     "bob-id",
			IsUnilateral:   false,
			MessageContent: "Hello",
			TokenCount:     5,
		},
	}

	csvMeasurements := convertToCSVMeasurements(measurements)

	if len(csvMeasurements) != 1 {
		t.Errorf("Expected 1 measurement, got %d", len(csvMeasurements))
	}

	m := csvMeasurements[0]
	if m.ID != "test-id" {
		t.Errorf("Expected ID 'test-id', got %s", m.ID)
	}
	if m.SenderName != "Alice" {
		t.Errorf("Expected sender name 'Alice', got %s", m.SenderName)
	}
	if m.MessageContent != "Hello" {
		t.Errorf("Expected message content 'Hello', got %s", m.MessageContent)
	}
	if m.TokenCount != 5 {
		t.Errorf("Expected token count 5, got %d", m.TokenCount)
	}
}

func TestBuildLaTeXConfig(t *testing.T) {
	t.Run("with default options", func(t *testing.T) {
		opts := &ExportOptions{}
		config := buildLaTeXConfig(opts)

		// Should have defaults
		if config.Style != "booktabs" {
			t.Errorf("Expected default style 'booktabs', got %s", config.Style)
		}
		if !config.IncludeTurn {
			t.Error("Expected IncludeTurn to be true by default")
		}
		if !config.IncludeParticipants {
			t.Error("Expected IncludeParticipants to be true by default")
		}
	})

	t.Run("with custom options", func(t *testing.T) {
		includeTurn := false
		includeParticipants := false
		includeBetaStatus := true

		opts := &ExportOptions{
			Style:               "plain",
			Caption:             "Test Caption",
			Label:               "tab:test",
			IncludeTurn:         &includeTurn,
			IncludeParticipants: &includeParticipants,
			IncludeBetaStatus:   &includeBetaStatus,
			Precision:           4,
		}
		config := buildLaTeXConfig(opts)

		if config.Style != "plain" {
			t.Errorf("Expected style 'plain', got %s", config.Style)
		}
		if config.Caption != "Test Caption" {
			t.Errorf("Expected caption 'Test Caption', got %s", config.Caption)
		}
		if config.Label != "tab:test" {
			t.Errorf("Expected label 'tab:test', got %s", config.Label)
		}
		if config.IncludeTurn {
			t.Error("Expected IncludeTurn to be false")
		}
		if config.IncludeParticipants {
			t.Error("Expected IncludeParticipants to be false")
		}
		if !config.IncludeBetaStatus {
			t.Error("Expected IncludeBetaStatus to be true")
		}
		if config.Precision != 4 {
			t.Errorf("Expected precision 4, got %d", config.Precision)
		}
	})
}

func TestBuildCSVConfig(t *testing.T) {
	t.Run("with default options", func(t *testing.T) {
		opts := &ExportOptions{}
		config := buildCSVConfig(opts)

		// Should have defaults
		if config.Dialect != "standard" {
			t.Errorf("Expected default dialect 'standard', got %s", string(config.Dialect))
		}
		if !config.IncludeHeader {
			t.Error("Expected IncludeHeader to be true by default")
		}
	})

	t.Run("with TSV dialect", func(t *testing.T) {
		opts := &ExportOptions{
			Dialect: "tsv",
		}
		config := buildCSVConfig(opts)

		if config.Dialect != "tsv" {
			t.Errorf("Expected dialect 'tsv', got %s", string(config.Dialect))
		}
	})

	t.Run("with Excel dialect", func(t *testing.T) {
		opts := &ExportOptions{
			Dialect: "excel",
		}
		config := buildCSVConfig(opts)

		if config.Dialect != "excel" {
			t.Errorf("Expected dialect 'excel', got %s", string(config.Dialect))
		}
	})
}

func TestSanitizeBibTeXKey(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"test-session-123", "test_session_123"},
		{"abc123", "abc123"},
		{"With Spaces", "With_Spaces"},
		{"special!@#chars", "specialchars"},
		{"", ""},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := sanitizeBibTeXKey(tt.input)
			if result != tt.expected {
				t.Errorf("sanitizeBibTeXKey(%q) = %q, expected %q", tt.input, result, tt.expected)
			}
		})
	}
}

func TestEscapeBibTeX(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"Normal text", "Normal text"},
		{"Test & and", "Test \\& and"},
		{"100%", "100\\%"},
		{"$100", "\\$100"},
		{"#hashtag", "\\#hashtag"},
		{"under_score", "under\\_score"},
		{"{braces}", "\\{braces\\}"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := escapeBibTeX(tt.input)
			if result != tt.expected {
				t.Errorf("escapeBibTeX(%q) = %q, expected %q", tt.input, result, tt.expected)
			}
		})
	}
}

func TestFormatInt(t *testing.T) {
	tests := []struct {
		input    int
		expected string
	}{
		{0, "0"},
		{1, "1"},
		{123, "123"},
		{-5, "-5"},
		{-123, "-123"},
	}

	for _, tt := range tests {
		result := formatInt(tt.input)
		if result != tt.expected {
			t.Errorf("formatInt(%d) = %q, expected %q", tt.input, result, tt.expected)
		}
	}
}

func TestGetIntOrDefault(t *testing.T) {
	tests := []struct {
		value    int
		def      int
		expected int
	}{
		{5, 3, 5},
		{0, 3, 3},
		{-1, 3, 3},
		{10, 5, 10},
	}

	for _, tt := range tests {
		result := getIntOrDefault(tt.value, tt.def)
		if result != tt.expected {
			t.Errorf("getIntOrDefault(%d, %d) = %d, expected %d", tt.value, tt.def, result, tt.expected)
		}
	}
}
