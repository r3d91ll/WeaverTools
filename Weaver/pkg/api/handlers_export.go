// Package api provides the HTTP/WebSocket server for the Weaver web UI.
package api

import (
	"bytes"
	"net/http"
	"time"

	"github.com/r3d91ll/weaver/pkg/export"
)

// ExportHandler handles export-related API requests.
type ExportHandler struct {
	// sessionStore provides access to session data for export
	sessionStore SessionStore
}

// NewExportHandler creates a new ExportHandler.
func NewExportHandler(sessionStore SessionStore) *ExportHandler {
	return &ExportHandler{
		sessionStore: sessionStore,
	}
}

// RegisterRoutes registers the export API routes on the router.
func (h *ExportHandler) RegisterRoutes(router *Router) {
	router.POST("/api/export/latex", h.ExportLaTeX)
	router.POST("/api/export/csv", h.ExportCSV)
	router.POST("/api/export/pdf", h.ExportPDF)
	router.POST("/api/export/bibtex", h.ExportBibTeX)
}

// -----------------------------------------------------------------------------
// API Request Types
// -----------------------------------------------------------------------------

// ExportRequest is the base request structure for export endpoints.
type ExportRequest struct {
	// SessionID is the session to export data from (optional, uses sample data if empty)
	SessionID string `json:"sessionId,omitempty"`

	// Measurements is an array of measurement data to export
	Measurements []ExportMeasurement `json:"measurements,omitempty"`

	// Format-specific options
	Options ExportOptions `json:"options,omitempty"`
}

// ExportMeasurement represents a measurement row for export.
type ExportMeasurement struct {
	Turn       int     `json:"turn"`
	Sender     string  `json:"sender"`
	Receiver   string  `json:"receiver"`
	DEff       int     `json:"deff"`
	Beta       float64 `json:"beta"`
	Alignment  float64 `json:"alignment"`
	CPair      float64 `json:"cpair"`
	BetaStatus string  `json:"betaStatus,omitempty"`

	// Extended fields for CSV export
	ID             string `json:"id,omitempty"`
	Timestamp      string `json:"timestamp,omitempty"`
	SessionID      string `json:"sessionId,omitempty"`
	ConversationID string `json:"conversationId,omitempty"`
	SenderID       string `json:"senderId,omitempty"`
	SenderRole     string `json:"senderRole,omitempty"`
	ReceiverID     string `json:"receiverId,omitempty"`
	ReceiverRole   string `json:"receiverRole,omitempty"`
	IsUnilateral   bool   `json:"isUnilateral,omitempty"`
	MessageContent string `json:"messageContent,omitempty"`
	TokenCount     int    `json:"tokenCount,omitempty"`
}

// ExportOptions contains format-specific export options.
type ExportOptions struct {
	// LaTeX options
	Style             string   `json:"style,omitempty"`             // "plain" or "booktabs"
	Caption           string   `json:"caption,omitempty"`           // Table caption
	Label             string   `json:"label,omitempty"`             // LaTeX label for cross-references
	IncludeTurn       *bool    `json:"includeTurn,omitempty"`       // Include turn column
	IncludeParticipants *bool  `json:"includeParticipants,omitempty"` // Include sender/receiver columns
	IncludeBetaStatus *bool    `json:"includeBetaStatus,omitempty"` // Include beta status column
	AlignmentAsPercent bool    `json:"alignmentAsPercent,omitempty"` // Format alignment as percentage
	Precision         int      `json:"precision,omitempty"`         // Decimal places for floats
	ColumnAlignments  []string `json:"columnAlignments,omitempty"`  // Column alignment overrides

	// CSV options
	Dialect              string `json:"dialect,omitempty"`              // "standard", "excel", "tsv"
	IncludeHeader        *bool  `json:"includeHeader,omitempty"`        // Include header row
	IncludeMessageContent bool  `json:"includeMessageContent,omitempty"` // Include message content
	IncludeTokenCount     bool  `json:"includeTokenCount,omitempty"`    // Include token count

	// BibTeX options
	Title    string `json:"title,omitempty"`    // Entry title
	Author   string `json:"author,omitempty"`   // Author name
	Year     string `json:"year,omitempty"`     // Publication year
	Keywords string `json:"keywords,omitempty"` // Keywords

	// Common options
	IncludeSummary bool `json:"includeSummary,omitempty"` // Include summary statistics
}

// -----------------------------------------------------------------------------
// API Response Types
// -----------------------------------------------------------------------------

// ExportResponse is the JSON response for export endpoints.
type ExportResponse struct {
	// Content is the exported content as a string
	Content string `json:"content"`

	// Format is the export format (latex, csv, pdf, bibtex)
	Format string `json:"format"`

	// Filename is the suggested filename for download
	Filename string `json:"filename"`

	// MimeType is the MIME type for the content
	MimeType string `json:"mimeType"`

	// Stats contains summary statistics if requested
	Stats *ExportStats `json:"stats,omitempty"`
}

// ExportStats contains summary statistics from the export.
type ExportStats struct {
	MeasurementCount int     `json:"measurementCount"`
	AvgDEff          float64 `json:"avgDEff"`
	AvgBeta          float64 `json:"avgBeta"`
	AvgAlignment     float64 `json:"avgAlignment"`
	AvgCPair         float64 `json:"avgCPair"`
	MinBeta          float64 `json:"minBeta"`
	MaxBeta          float64 `json:"maxBeta"`
	BilateralCount   int     `json:"bilateralCount"`
}

// -----------------------------------------------------------------------------
// Handlers
// -----------------------------------------------------------------------------

// ExportLaTeX handles POST /api/export/latex.
// It generates a LaTeX table from measurement data.
func (h *ExportHandler) ExportLaTeX(w http.ResponseWriter, r *http.Request) {
	// Parse request body
	var req ExportRequest
	if err := ReadJSON(r, &req); err != nil {
		WriteError(w, http.StatusBadRequest, "invalid_json",
			"Failed to parse request body: "+err.Error())
		return
	}

	// Convert request measurements to export format
	rows := convertToMeasurementRows(req.Measurements)

	if len(rows) == 0 {
		WriteError(w, http.StatusBadRequest, "no_data",
			"No measurement data provided for export")
		return
	}

	// Build LaTeX table configuration
	config := buildLaTeXConfig(&req.Options)

	// Generate LaTeX content
	content := export.GenerateMeasurementTable(rows, config)

	// Add summary table if requested
	if req.Options.IncludeSummary {
		stats := export.ComputeSummaryStats(rows)
		summaryConfig := &export.SummaryTableConfig{
			TableConfig: export.TableConfig{
				Style:   config.Style,
				Caption: "Summary Statistics",
				Label:   config.Label + "_summary",
			},
			Precision:             getIntOrDefault(req.Options.Precision, 3),
			IncludeMinMax:         true,
			IncludeBilateralCount: true,
		}
		content += "\n\n" + export.GenerateSummaryTable(stats, summaryConfig)
	}

	// Build response
	response := ExportResponse{
		Content:  content,
		Format:   "latex",
		Filename: "measurements.tex",
		MimeType: "application/x-latex",
	}

	// Add stats if summary was requested
	if req.Options.IncludeSummary {
		stats := export.ComputeSummaryStats(rows)
		response.Stats = convertToExportStats(stats)
	}

	WriteJSON(w, http.StatusOK, response)
}

// ExportCSV handles POST /api/export/csv.
// It generates a CSV file from measurement data.
func (h *ExportHandler) ExportCSV(w http.ResponseWriter, r *http.Request) {
	// Parse request body
	var req ExportRequest
	if err := ReadJSON(r, &req); err != nil {
		WriteError(w, http.StatusBadRequest, "invalid_json",
			"Failed to parse request body: "+err.Error())
		return
	}

	// Convert request measurements to CSV format
	csvMeasurements := convertToCSVMeasurements(req.Measurements)

	if len(csvMeasurements) == 0 {
		WriteError(w, http.StatusBadRequest, "no_data",
			"No measurement data provided for export")
		return
	}

	// Build CSV configuration
	config := buildCSVConfig(&req.Options)

	// Generate CSV content
	var buf bytes.Buffer
	if err := export.ExportMeasurementsToCSV(&buf, csvMeasurements, config); err != nil {
		WriteError(w, http.StatusInternalServerError, "export_error",
			"Failed to generate CSV: "+err.Error())
		return
	}

	// Build response
	response := ExportResponse{
		Content:  buf.String(),
		Format:   "csv",
		Filename: "measurements.csv",
		MimeType: "text/csv",
	}

	// Add stats if summary was requested
	if req.Options.IncludeSummary {
		rows := convertToMeasurementRows(req.Measurements)
		stats := export.ComputeSummaryStats(rows)
		response.Stats = convertToExportStats(stats)
	}

	WriteJSON(w, http.StatusOK, response)
}

// ExportPDF handles POST /api/export/pdf.
// It generates a PDF document from measurement data.
// Note: PDF generation is implemented as a wrapper that returns LaTeX
// with instructions for compilation. Full PDF generation requires
// a LaTeX installation which may not be available on all systems.
func (h *ExportHandler) ExportPDF(w http.ResponseWriter, r *http.Request) {
	// Parse request body
	var req ExportRequest
	if err := ReadJSON(r, &req); err != nil {
		WriteError(w, http.StatusBadRequest, "invalid_json",
			"Failed to parse request body: "+err.Error())
		return
	}

	// Convert request measurements to export format
	rows := convertToMeasurementRows(req.Measurements)

	if len(rows) == 0 {
		WriteError(w, http.StatusBadRequest, "no_data",
			"No measurement data provided for export")
		return
	}

	// Build LaTeX table configuration for PDF
	config := buildLaTeXConfig(&req.Options)

	// Generate LaTeX table
	tableContent := export.GenerateMeasurementTable(rows, config)

	// Generate full LaTeX document for PDF compilation
	content := generateLaTeXDocument(tableContent, &req.Options)

	// Build response
	response := ExportResponse{
		Content:  content,
		Format:   "pdf",
		Filename: "measurements.tex", // .tex because it needs compilation
		MimeType: "application/x-latex",
	}

	// Add stats if summary was requested
	if req.Options.IncludeSummary {
		stats := export.ComputeSummaryStats(rows)
		response.Stats = convertToExportStats(stats)
	}

	WriteJSON(w, http.StatusOK, response)
}

// ExportBibTeX handles POST /api/export/bibtex.
// It generates a BibTeX entry for the dataset.
func (h *ExportHandler) ExportBibTeX(w http.ResponseWriter, r *http.Request) {
	// Parse request body
	var req ExportRequest
	if err := ReadJSON(r, &req); err != nil {
		WriteError(w, http.StatusBadRequest, "invalid_json",
			"Failed to parse request body: "+err.Error())
		return
	}

	// Generate BibTeX content
	content := generateBibTeX(&req)

	// Build response
	response := ExportResponse{
		Content:  content,
		Format:   "bibtex",
		Filename: "dataset.bib",
		MimeType: "application/x-bibtex",
	}

	// Add stats if summary was requested and measurements provided
	if req.Options.IncludeSummary && len(req.Measurements) > 0 {
		rows := convertToMeasurementRows(req.Measurements)
		stats := export.ComputeSummaryStats(rows)
		response.Stats = convertToExportStats(stats)
	}

	WriteJSON(w, http.StatusOK, response)
}

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

// convertToMeasurementRows converts API measurements to export.MeasurementRow.
func convertToMeasurementRows(measurements []ExportMeasurement) []export.MeasurementRow {
	rows := make([]export.MeasurementRow, len(measurements))
	for i, m := range measurements {
		rows[i] = export.MeasurementRow{
			Turn:       m.Turn,
			Sender:     m.Sender,
			Receiver:   m.Receiver,
			DEff:       m.DEff,
			Beta:       m.Beta,
			Alignment:  m.Alignment,
			CPair:      m.CPair,
			BetaStatus: m.BetaStatus,
		}
	}
	return rows
}

// convertToCSVMeasurements converts API measurements to export.CSVMeasurement.
func convertToCSVMeasurements(measurements []ExportMeasurement) []*export.CSVMeasurement {
	result := make([]*export.CSVMeasurement, len(measurements))
	for i, m := range measurements {
		csvM := &export.CSVMeasurement{
			ID:             m.ID,
			SessionID:      m.SessionID,
			ConversationID: m.ConversationID,
			TurnNumber:     m.Turn,
			SenderID:       m.SenderID,
			SenderName:     m.Sender,
			SenderRole:     m.SenderRole,
			ReceiverID:     m.ReceiverID,
			ReceiverName:   m.Receiver,
			ReceiverRole:   m.ReceiverRole,
			DEff:           m.DEff,
			Beta:           m.Beta,
			Alignment:      m.Alignment,
			CPair:          m.CPair,
			BetaStatus:     m.BetaStatus,
			IsUnilateral:   m.IsUnilateral,
			MessageContent: m.MessageContent,
			TokenCount:     m.TokenCount,
		}

		// Parse timestamp if provided
		if m.Timestamp != "" {
			if t, err := time.Parse(time.RFC3339, m.Timestamp); err == nil {
				csvM.Timestamp = t
			}
		}

		result[i] = csvM
	}
	return result
}

// convertToExportStats converts export.SummaryStats to API ExportStats.
func convertToExportStats(stats export.SummaryStats) *ExportStats {
	return &ExportStats{
		MeasurementCount: stats.MeasurementCount,
		AvgDEff:          stats.AvgDEff,
		AvgBeta:          stats.AvgBeta,
		AvgAlignment:     stats.AvgAlignment,
		AvgCPair:         stats.AvgCPair,
		MinBeta:          stats.MinBeta,
		MaxBeta:          stats.MaxBeta,
		BilateralCount:   stats.BilateralCount,
	}
}

// buildLaTeXConfig builds a MeasurementTableConfig from export options.
func buildLaTeXConfig(opts *ExportOptions) *export.MeasurementTableConfig {
	config := export.DefaultMeasurementTableConfig()

	if opts.Style != "" {
		config.Style = opts.Style
	}
	if opts.Caption != "" {
		config.Caption = opts.Caption
	}
	if opts.Label != "" {
		config.Label = opts.Label
	}
	if opts.IncludeTurn != nil {
		config.IncludeTurn = *opts.IncludeTurn
	}
	if opts.IncludeParticipants != nil {
		config.IncludeParticipants = *opts.IncludeParticipants
	}
	if opts.IncludeBetaStatus != nil {
		config.IncludeBetaStatus = *opts.IncludeBetaStatus
	}
	config.AlignmentAsPercent = opts.AlignmentAsPercent
	if opts.Precision > 0 {
		config.Precision = opts.Precision
	}

	return config
}

// buildCSVConfig builds a CSVConfig from export options.
func buildCSVConfig(opts *ExportOptions) *export.CSVConfig {
	config := export.DefaultCSVConfig()

	if opts.Dialect != "" {
		switch opts.Dialect {
		case "excel":
			config.Dialect = export.DialectExcel
		case "tsv":
			config.Dialect = export.DialectTSV
		default:
			config.Dialect = export.DialectStandard
		}
	}
	if opts.IncludeHeader != nil {
		config.IncludeHeader = *opts.IncludeHeader
	}
	if opts.IncludeBetaStatus != nil {
		config.IncludeBetaStatus = *opts.IncludeBetaStatus
	} else {
		config.IncludeBetaStatus = true
	}
	config.IncludeMessageContent = opts.IncludeMessageContent
	config.IncludeTokenCount = opts.IncludeTokenCount
	if opts.Precision > 0 {
		config.Precision = opts.Precision
	}

	return config
}

// generateLaTeXDocument generates a complete LaTeX document for PDF export.
func generateLaTeXDocument(tableContent string, opts *ExportOptions) string {
	title := opts.Caption
	if title == "" {
		title = "Conveyance Measurement Results"
	}

	doc := `\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{geometry}
\geometry{margin=1in}

\title{` + export.Escape(title) + `}
\author{WeaverTools Export}
\date{\today}

\begin{document}

\maketitle

\section{Measurement Data}

` + tableContent + `

\end{document}
`
	return doc
}

// generateBibTeX generates a BibTeX entry for the dataset.
func generateBibTeX(req *ExportRequest) string {
	// Use provided values or defaults
	title := req.Options.Title
	if title == "" {
		title = "Multi-Agent Conveyance Measurement Dataset"
	}

	author := req.Options.Author
	if author == "" {
		author = "WeaverTools"
	}

	year := req.Options.Year
	if year == "" {
		year = time.Now().Format("2006")
	}

	keywords := req.Options.Keywords
	if keywords == "" {
		keywords = "conveyance, multi-agent, AI, measurement"
	}

	// Generate a unique key based on session or timestamp
	key := "weavertools_" + time.Now().Format("20060102")
	if req.SessionID != "" {
		key = "weavertools_" + sanitizeBibTeXKey(req.SessionID)
	}

	// Count measurements if available
	measurementCount := len(req.Measurements)

	// Build BibTeX entry
	bib := "@misc{" + key + ",\n"
	bib += "  title = {" + escapeBibTeX(title) + "},\n"
	bib += "  author = {" + escapeBibTeX(author) + "},\n"
	bib += "  year = {" + year + "},\n"
	bib += "  howpublished = {WeaverTools Multi-Agent Research Platform},\n"
	if measurementCount > 0 {
		bib += "  note = {Dataset contains " + formatInt(measurementCount) + " measurements},\n"
	}
	bib += "  keywords = {" + escapeBibTeX(keywords) + "},\n"
	bib += "}\n"

	return bib
}

// sanitizeBibTeXKey removes or replaces characters invalid in BibTeX keys.
func sanitizeBibTeXKey(s string) string {
	result := make([]byte, 0, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' {
			result = append(result, c)
		} else if c == '-' || c == ' ' {
			result = append(result, '_')
		}
	}
	return string(result)
}

// escapeBibTeX escapes special characters in BibTeX values.
func escapeBibTeX(s string) string {
	// Replace characters that need escaping in BibTeX
	result := make([]byte, 0, len(s)*2)
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch c {
		case '&':
			result = append(result, '\\', '&')
		case '%':
			result = append(result, '\\', '%')
		case '$':
			result = append(result, '\\', '$')
		case '#':
			result = append(result, '\\', '#')
		case '_':
			result = append(result, '\\', '_')
		case '{', '}':
			result = append(result, '\\', c)
		case '~':
			result = append(result, '\\', '~', '{', '}')
		case '^':
			result = append(result, '\\', '^', '{', '}')
		default:
			result = append(result, c)
		}
	}
	return string(result)
}

// formatInt converts an int to a string.
func formatInt(n int) string {
	// Simple int to string conversion
	if n == 0 {
		return "0"
	}

	negative := n < 0
	if negative {
		n = -n
	}

	digits := make([]byte, 0, 10)
	for n > 0 {
		digits = append([]byte{byte('0' + n%10)}, digits...)
		n /= 10
	}

	if negative {
		digits = append([]byte{'-'}, digits...)
	}

	return string(digits)
}

// getIntOrDefault returns the value if positive, otherwise the default.
func getIntOrDefault(value, defaultValue int) int {
	if value > 0 {
		return value
	}
	return defaultValue
}
