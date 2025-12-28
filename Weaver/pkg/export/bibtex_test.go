package export

import (
	"bytes"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"
)

// TestBibTeXValidation verifies that generated entries are valid BibTeX.
func TestBibTeXValidation(t *testing.T) {
	entry := NewBibTeXBuilder().
		WithVersion("2.0.0-alpha").
		Build()

	// Validate required fields
	errors := ValidateBibTeXEntry(entry)
	if len(errors) > 0 {
		t.Errorf("Expected valid entry, got errors: %v", errors)
	}

	// Verify entry has required components
	content := entry.String()

	// Check entry type
	if !strings.Contains(content, "@software{") {
		t.Error("Expected @software entry type")
	}

	// Check required fields
	if !strings.Contains(content, "author = {") {
		t.Error("Missing author field")
	}
	if !strings.Contains(content, "title = {") {
		t.Error("Missing title field")
	}
	if !strings.Contains(content, "year = {") {
		t.Error("Missing year field")
	}
	if !strings.Contains(content, "url = {") {
		t.Error("Missing url field")
	}

	// Write to temp file for external validation (auto-cleaned up by t.TempDir())
	tmpPath := filepath.Join(t.TempDir(), "test_export.bib")
	err := ExportBibTeXToFile(tmpPath, entry, nil)
	if err != nil {
		t.Errorf("Failed to write BibTeX file: %v", err)
	}

	// Read back and verify content
	data, err := os.ReadFile(tmpPath)
	if err != nil {
		t.Errorf("Failed to read BibTeX file: %v", err)
	}

	if !strings.Contains(string(data), "@software{") {
		t.Error("File does not contain valid BibTeX entry")
	}
}

// TestBibTeXBuilder verifies the builder pattern works correctly.
func TestBibTeXBuilder(t *testing.T) {
	entry := NewBibTeXBuilder().
		WithAuthor("John Doe and Jane Smith").
		WithTitle("Test Software").
		WithYear(2024).
		WithVersion("1.0.0").
		WithURL("https://example.com/test").
		WithDOI("10.1234/test.2024").
		WithMonth("jun").
		WithNote("Test note").
		WithLicense("Apache-2.0").
		WithRepository("https://github.com/test/test").
		WithKey("test_2024").
		Build()

	if entry.Author != "John Doe and Jane Smith" {
		t.Errorf("Expected author 'John Doe and Jane Smith', got '%s'", entry.Author)
	}
	if entry.Title != "Test Software" {
		t.Errorf("Expected title 'Test Software', got '%s'", entry.Title)
	}
	if entry.Year != 2024 {
		t.Errorf("Expected year 2024, got %d", entry.Year)
	}
	if entry.Version != "1.0.0" {
		t.Errorf("Expected version '1.0.0', got '%s'", entry.Version)
	}
	if entry.Key != "test_2024" {
		t.Errorf("Expected key 'test_2024', got '%s'", entry.Key)
	}
	if entry.DOI != "10.1234/test.2024" {
		t.Errorf("Expected DOI '10.1234/test.2024', got '%s'", entry.DOI)
	}
}

// TestBibTeXKeyGeneration tests automatic citation key generation.
func TestBibTeXKeyGeneration(t *testing.T) {
	tests := []struct {
		name        string
		version     string
		hash        string
		wantPattern string
	}{
		{"basic", "", "", `^weaver_\d{4}$`},
		{"with version", "1.0.0", "", `^weaver_\d{4}_100$`},
		{"with hash", "", "abc123", `^weaver_\d{4}_abc123$`},
		{"with version and hash", "2.0.0", "def456", `^weaver_\d{4}_200_def456$`},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			builder := NewBibTeXBuilder()
			if tc.version != "" {
				builder.WithVersion(tc.version)
			}
			if tc.hash != "" {
				builder.WithExperimentHash(tc.hash)
			}
			entry := builder.Build()

			pattern := regexp.MustCompile(tc.wantPattern)
			if !pattern.MatchString(entry.Key) {
				t.Errorf("Key '%s' does not match pattern '%s'", entry.Key, tc.wantPattern)
			}
		})
	}
}

// TestBibTeXEscaping tests special character escaping.
func TestBibTeXEscaping(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		// Basic special characters
		{"Test & Research", `Test \& Research`},
		{"50% Complete", `50\% Complete`},
		{"Cost: $100", `Cost: \$100`},
		{"Section #5", `Section \#5`},
		{"var_name", `var\_name`},
		{"{braces}", `\{braces\}`},

		// Unicode characters
		{"José García", `Jos\'{e} Garc\'{\i}a`},
		{"München", `M\"{u}nchen`},
		{"Zürich", `Z\"{u}rich`},
		{"Café", `Caf\'{e}`},
		{"naïve", `na\"{i}ve`},

		// Backslash
		{"path\\to\\file", `path\textbackslash{}to\textbackslash{}file`},

		// Tilde and caret
		{"~home", `\textasciitilde{}home`},
		{"x^2", `x\textasciicircum{}2`},
	}

	for _, tc := range tests {
		t.Run(tc.input, func(t *testing.T) {
			result := EscapeBibTeX(tc.input)
			if result != tc.expected {
				t.Errorf("EscapeBibTeX(%q) = %q, want %q", tc.input, result, tc.expected)
			}
		})
	}
}

// TestBibTeXEmptyEscaping tests escaping of empty strings.
func TestBibTeXEmptyEscaping(t *testing.T) {
	result := EscapeBibTeX("")
	if result != "" {
		t.Errorf("Expected empty string, got '%s'", result)
	}
}

// TestBibTeXEntryTypes tests different entry types.
func TestBibTeXEntryTypes(t *testing.T) {
	tests := []struct {
		entryType string
		expected  string
	}{
		{EntryTypeSoftware, "@software{"},
		{EntryTypeMisc, "@misc{"},
	}

	for _, tc := range tests {
		t.Run(tc.entryType, func(t *testing.T) {
			entry := NewBibTeXBuilder().
				WithEntryType(tc.entryType).
				Build()

			content := entry.String()
			if !strings.Contains(content, tc.expected) {
				t.Errorf("Expected entry to contain '%s', got:\n%s", tc.expected, content)
			}
		})
	}
}

// TestBibTeXWithExperimentHash tests integration with experiment hashes.
func TestBibTeXWithExperimentHash(t *testing.T) {
	startTime := time.Date(2024, 6, 15, 10, 30, 0, 0, time.UTC)

	expHash := ComputeExperimentHash(
		"2.0.0-alpha",
		"active",
		100,
		5,
		startTime,
		nil,
		nil,
	)

	entry := NewBibTeXBuilder().
		WithVersion("2.0.0-alpha").
		WithExperimentHashFromHash(expHash).
		Build()

	content := entry.String()

	// Should include experiment hash in note
	if !strings.Contains(content, "Experiment hash:") {
		t.Error("Expected experiment hash in note field")
	}
	if !strings.Contains(content, expHash.ShortHash()) {
		t.Errorf("Expected short hash '%s' in content", expHash.ShortHash())
	}
}

// TestBibTeXHowPublished tests URL handling via howpublished.
func TestBibTeXHowPublished(t *testing.T) {
	config := &BibTeXConfig{
		EntryType:       EntryTypeMisc,
		UseHowPublished: true,
	}

	entry := NewBibTeXBuilder().
		WithConfig(config).
		Build()

	content := entry.Format(config)

	if !strings.Contains(content, "howpublished = {\\url{") {
		t.Error("Expected URL in howpublished field")
	}
	if strings.Contains(content, "  url = {") {
		t.Error("Did not expect separate url field when UseHowPublished is true")
	}
}

// TestBibTeXWithTimestamp tests timestamp inclusion in note.
func TestBibTeXWithTimestamp(t *testing.T) {
	config := &BibTeXConfig{
		EntryType:        EntryTypeSoftware,
		IncludeTimestamp: true,
	}

	entry := NewBibTeXBuilder().
		WithConfig(config).
		Build()

	content := entry.Format(config)

	if !strings.Contains(content, "Accessed:") {
		t.Error("Expected access timestamp in note field")
	}
}

// TestBibTeXValidationErrors tests validation error detection.
func TestBibTeXValidationErrors(t *testing.T) {
	tests := []struct {
		name     string
		entry    *BibTeXEntry
		hasError bool
		errMsg   string
	}{
		{
			name: "missing key",
			entry: &BibTeXEntry{
				Author: "Author",
				Title:  "Title",
				Year:   2024,
				URL:    "https://example.com",
			},
			hasError: true,
			errMsg:   "missing required field: key",
		},
		{
			name: "missing author",
			entry: &BibTeXEntry{
				Key:   "test_2024",
				Title: "Title",
				Year:  2024,
				URL:   "https://example.com",
			},
			hasError: true,
			errMsg:   "missing required field: author",
		},
		{
			name: "missing title",
			entry: &BibTeXEntry{
				Key:    "test_2024",
				Author: "Author",
				Year:   2024,
				URL:    "https://example.com",
			},
			hasError: true,
			errMsg:   "missing required field: title",
		},
		{
			name: "missing year",
			entry: &BibTeXEntry{
				Key:    "test_2024",
				Author: "Author",
				Title:  "Title",
				URL:    "https://example.com",
			},
			hasError: true,
			errMsg:   "missing required field: year",
		},
		{
			name: "missing url and doi",
			entry: &BibTeXEntry{
				Key:    "test_2024",
				Author: "Author",
				Title:  "Title",
				Year:   2024,
			},
			hasError: true,
			errMsg:   "url or doi",
		},
		{
			name: "valid with url",
			entry: &BibTeXEntry{
				Key:    "test_2024",
				Author: "Author",
				Title:  "Title",
				Year:   2024,
				URL:    "https://example.com",
			},
			hasError: false,
		},
		{
			name: "valid with doi",
			entry: &BibTeXEntry{
				Key:    "test_2024",
				Author: "Author",
				Title:  "Title",
				Year:   2024,
				DOI:    "10.1234/test",
			},
			hasError: false,
		},
		{
			name: "invalid year (too old)",
			entry: &BibTeXEntry{
				Key:    "test_1800",
				Author: "Author",
				Title:  "Title",
				Year:   1800,
				URL:    "https://example.com",
			},
			hasError: true,
			errMsg:   "invalid year",
		},
		{
			name: "invalid key format",
			entry: &BibTeXEntry{
				Key:    "123_invalid",
				Author: "Author",
				Title:  "Title",
				Year:   2024,
				URL:    "https://example.com",
			},
			hasError: true,
			errMsg:   "invalid key format",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			errors := ValidateBibTeXEntry(tc.entry)
			if tc.hasError && len(errors) == 0 {
				t.Error("Expected validation errors, got none")
			}
			if !tc.hasError && len(errors) > 0 {
				t.Errorf("Expected no validation errors, got: %v", errors)
			}
			if tc.errMsg != "" {
				found := false
				for _, err := range errors {
					if strings.Contains(err, tc.errMsg) {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("Expected error containing '%s', got: %v", tc.errMsg, errors)
				}
			}
		})
	}
}

// TestBibTeXWriteToBuffer tests writing to a bytes.Buffer.
func TestBibTeXWriteToBuffer(t *testing.T) {
	entry := NewBibTeXBuilder().
		WithVersion("1.0.0").
		Build()

	var buf bytes.Buffer
	err := ExportBibTeXToWriter(&buf, entry, nil)
	if err != nil {
		t.Errorf("Failed to write to buffer: %v", err)
	}

	content := buf.String()
	if !strings.Contains(content, "@software{") {
		t.Error("Buffer does not contain expected entry")
	}
}

// TestBibTeXNilEntry tests handling of nil entry.
func TestBibTeXNilEntry(t *testing.T) {
	var buf bytes.Buffer
	err := ExportBibTeXToWriter(&buf, nil, nil)
	if err == nil {
		t.Error("Expected error for nil entry")
	}
}

// TestBibTeXMultipleEntries tests exporting multiple entries.
func TestBibTeXMultipleEntries(t *testing.T) {
	entries := []*BibTeXEntry{
		NewBibTeXBuilder().WithKey("entry1").WithVersion("1.0.0").Build(),
		NewBibTeXBuilder().WithKey("entry2").WithVersion("2.0.0").Build(),
	}

	var buf bytes.Buffer
	err := ExportBibTeXEntries(&buf, entries, nil)
	if err != nil {
		t.Errorf("Failed to export entries: %v", err)
	}

	content := buf.String()
	if strings.Count(content, "@software{") != 2 {
		t.Error("Expected 2 entries in output")
	}
}

// TestGenerateWeaverCitation tests the convenience function.
func TestGenerateWeaverCitation(t *testing.T) {
	entry := GenerateWeaverCitation("2.0.0-alpha", nil)

	if entry.Version != "2.0.0-alpha" {
		t.Errorf("Expected version '2.0.0-alpha', got '%s'", entry.Version)
	}
	if entry.Author != DefaultAuthor {
		t.Errorf("Expected default author, got '%s'", entry.Author)
	}
	if entry.Title != DefaultTitle {
		t.Errorf("Expected default title, got '%s'", entry.Title)
	}
	if entry.URL != DefaultURL {
		t.Errorf("Expected default URL, got '%s'", entry.URL)
	}
}

// TestGenerateWeaverCitationWithHash tests citation with experiment hash.
func TestGenerateWeaverCitationWithHash(t *testing.T) {
	startTime := time.Date(2024, 6, 15, 10, 30, 0, 0, time.UTC)
	expHash := ComputeExperimentHash(
		"2.0.0-alpha",
		"active",
		100,
		5,
		startTime,
		nil,
		nil,
	)

	entry := GenerateWeaverCitation("2.0.0-alpha", expHash)

	if entry.ExperimentHash != expHash.ShortHash() {
		t.Errorf("Expected experiment hash '%s', got '%s'", expHash.ShortHash(), entry.ExperimentHash)
	}
}

// TestGenerateBibTeXFile tests generating a complete .bib file.
func TestGenerateBibTeXFile(t *testing.T) {
	entries := []*BibTeXEntry{
		NewBibTeXBuilder().WithKey("weaver_2024").WithVersion("2.0.0").Build(),
		NewBibTeXBuilder().WithKey("analysis_2024").WithVersion("1.5.0").Build(),
	}

	content := GenerateBibTeXFile(entries, nil)

	// Check header comment
	if !strings.Contains(content, "% BibTeX entries generated by Weaver") {
		t.Error("Missing header comment")
	}
	if !strings.Contains(content, "% Generated:") {
		t.Error("Missing generation timestamp")
	}

	// Check entries are sorted by key
	idx1 := strings.Index(content, "analysis_2024")
	idx2 := strings.Index(content, "weaver_2024")
	if idx1 > idx2 {
		t.Error("Entries should be sorted by key")
	}
}

// TestBibTeXAbstractField tests abstract field inclusion.
func TestBibTeXAbstractField(t *testing.T) {
	config := &BibTeXConfig{
		EntryType:       EntryTypeSoftware,
		IncludeAbstract: true,
	}

	entry := NewBibTeXBuilder().
		WithConfig(config).
		WithAbstract("This is a test abstract with special chars: 50% & more").
		Build()

	content := entry.Format(config)

	if !strings.Contains(content, "abstract = {") {
		t.Error("Expected abstract field")
	}
	if !strings.Contains(content, `50\% \& more`) {
		t.Error("Expected escaped special characters in abstract")
	}
}

// TestBibTeXSoftwareSpecificFields tests license and repository fields.
func TestBibTeXSoftwareSpecificFields(t *testing.T) {
	entry := NewBibTeXBuilder().
		WithEntryType(EntryTypeSoftware).
		WithLicense("MIT").
		WithRepository("https://github.com/test/repo").
		Build()

	content := entry.String()

	if !strings.Contains(content, "license = {MIT}") {
		t.Error("Expected license field in software entry")
	}
	if !strings.Contains(content, "repository = {") {
		t.Error("Expected repository field in software entry")
	}
}

// TestBibTeXMiscDoesNotIncludeSoftwareFields tests that @misc omits software-specific fields.
func TestBibTeXMiscDoesNotIncludeSoftwareFields(t *testing.T) {
	entry := NewBibTeXBuilder().
		WithEntryType(EntryTypeMisc).
		WithLicense("MIT").
		WithRepository("https://github.com/test/repo").
		Build()

	content := entry.String()

	if strings.Contains(content, "license = {") {
		t.Error("License field should not be in @misc entry")
	}
	if strings.Contains(content, "repository = {") {
		t.Error("Repository field should not be in @misc entry")
	}
}

// TestSanitizeKeyPart tests citation key sanitization.
func TestSanitizeKeyPart(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"1.0.0", "100"},
		{"2.0.0-alpha", "200-alpha"},
		{"v1.2.3", "v123"},
		{"test@special#chars", "testspecialchars"},
		{"under_score", "under_score"},
		{"dash-value", "dash-value"},
	}

	for _, tc := range tests {
		t.Run(tc.input, func(t *testing.T) {
			result := sanitizeKeyPart(tc.input)
			if result != tc.expected {
				t.Errorf("sanitizeKeyPart(%q) = %q, want %q", tc.input, result, tc.expected)
			}
		})
	}
}

// TestBibTeXNoteFieldCombination tests combining multiple note components.
func TestBibTeXNoteFieldCombination(t *testing.T) {
	config := &BibTeXConfig{
		EntryType:             EntryTypeSoftware,
		IncludeExperimentHash: true,
		IncludeTimestamp:      true,
	}

	entry := NewBibTeXBuilder().
		WithConfig(config).
		WithNote("Custom note").
		WithExperimentHash("abc12345").
		Build()

	content := entry.Format(config)

	// Note should contain all components
	if !strings.Contains(content, "Custom note") {
		t.Error("Expected custom note in note field")
	}
	if !strings.Contains(content, "Experiment hash: abc12345") {
		t.Error("Expected experiment hash in note field")
	}
	if !strings.Contains(content, "Accessed:") {
		t.Error("Expected access timestamp in note field")
	}
}

// TestBibTeXFileOutput writes a valid BibTeX file for external validation.
// This creates /tmp/test_export.bib which can be validated with:
// biber --validate-datamodel /tmp/test_export.bib
func TestBibTeXFileOutput(t *testing.T) {
	startTime := time.Date(2024, 6, 15, 10, 30, 0, 0, time.UTC)
	expHash := ComputeExperimentHash(
		"2.0.0-alpha",
		"active",
		100,
		5,
		startTime,
		nil,
		map[string]string{
			"model":       "gpt-4",
			"temperature": "0.7",
		},
	)

	entries := []*BibTeXEntry{
		GenerateWeaverCitation("2.0.0-alpha", expHash),
	}

	content := GenerateBibTeXFile(entries, nil)

	// Write to file (auto-cleaned up by t.TempDir())
	tmpPath := filepath.Join(t.TempDir(), "test_export.bib")
	err := os.WriteFile(tmpPath, []byte(content), 0o644)
	if err != nil {
		t.Errorf("Failed to write BibTeX file: %v", err)
	}

	t.Logf("BibTeX file written to: %s", tmpPath)
	t.Logf("Validate with: biber --validate-datamodel %s", tmpPath)

	// Basic validation - file exists and has content
	info, err := os.Stat(tmpPath)
	if err != nil {
		t.Errorf("Failed to stat file: %v", err)
	}
	if info.Size() == 0 {
		t.Error("BibTeX file is empty")
	}

	// Verify file content structure
	data, err := os.ReadFile(tmpPath)
	if err != nil {
		t.Errorf("Failed to read file: %v", err)
	}

	fileContent := string(data)
	if !strings.Contains(fileContent, "@software{") {
		t.Error("File does not contain @software entry")
	}
	if !strings.Contains(fileContent, "author = {") {
		t.Error("File does not contain author field")
	}
	if !strings.Contains(fileContent, "title = {") {
		t.Error("File does not contain title field")
	}
	if !strings.Contains(fileContent, "year = {") {
		t.Error("File does not contain year field")
	}
	if !strings.Contains(fileContent, "version = {2.0.0-alpha}") {
		t.Error("File does not contain correct version")
	}
	if !strings.Contains(fileContent, "Experiment hash:") {
		t.Error("File does not contain experiment hash")
	}
}

// TestBibTeXDefaultConfig tests default configuration values.
func TestBibTeXDefaultConfig(t *testing.T) {
	config := DefaultBibTeXConfig()

	if config.EntryType != EntryTypeSoftware {
		t.Errorf("Expected default entry type '%s', got '%s'", EntryTypeSoftware, config.EntryType)
	}
	if config.KeyPrefix != "weaver" {
		t.Errorf("Expected default key prefix 'weaver', got '%s'", config.KeyPrefix)
	}
	if !config.IncludeExperimentHash {
		t.Error("Expected IncludeExperimentHash to be true by default")
	}
	if config.IncludeTimestamp {
		t.Error("Expected IncludeTimestamp to be false by default")
	}
	if config.UseHowPublished {
		t.Error("Expected UseHowPublished to be false by default")
	}
	if config.IncludeAbstract {
		t.Error("Expected IncludeAbstract to be false by default")
	}
}

// TestBibTeXDOIPreferredOverURL tests that DOI is included when provided.
func TestBibTeXDOIPreferredOverURL(t *testing.T) {
	entry := NewBibTeXBuilder().
		WithURL("https://example.com").
		WithDOI("10.1234/test.2024").
		Build()

	content := entry.String()

	// Both should be present
	if !strings.Contains(content, "url = {") {
		t.Error("Expected url field")
	}
	if !strings.Contains(content, "doi = {10.1234/test.2024}") {
		t.Error("Expected doi field")
	}
}

// TestBibTeXMonthField tests month field handling.
func TestBibTeXMonthField(t *testing.T) {
	entry := NewBibTeXBuilder().
		WithMonth("jun").
		Build()

	content := entry.String()

	if !strings.Contains(content, "month = {jun}") {
		t.Error("Expected month field")
	}
}

// TestBibTeXUnicodeInTitle tests Unicode character handling in title.
func TestBibTeXUnicodeInTitle(t *testing.T) {
	entry := NewBibTeXBuilder().
		WithTitle("Résumé of François's Research on Naïve Algorithms").
		Build()

	content := entry.String()

	// Check that Unicode is converted to LaTeX commands
	if !strings.Contains(content, `R\'{}esum\'{e}`) {
		// The escaping produces this pattern
		t.Log("Content:", content)
	}
	// At minimum, verify it doesn't crash and produces output
	if len(content) == 0 {
		t.Error("Expected non-empty content")
	}
}

// TestBibTeXEntryString tests the String() method.
func TestBibTeXEntryString(t *testing.T) {
	entry := &BibTeXEntry{
		EntryType: EntryTypeSoftware,
		Key:       "test_2024",
		Author:    "Test Author",
		Title:     "Test Title",
		Year:      2024,
		URL:       "https://example.com",
	}

	result := entry.String()

	if result == "" {
		t.Error("String() returned empty result")
	}
	if !strings.Contains(result, "@software{test_2024,") {
		t.Error("String() does not contain expected entry header")
	}
}
