// Package export provides academic export format utilities.
// Generates LaTeX tables, CSV data, and other publication-ready outputs.
package export

import (
	"fmt"
	"io"
	"os"
	"regexp"
	"sort"
	"strings"
	"time"
	"unicode"
)

// BibTeX entry type constants.
const (
	// EntryTypeSoftware is the recommended entry type for software citations.
	// Supported by BibLaTeX and some modern BibTeX styles.
	EntryTypeSoftware = "software"

	// EntryTypeMisc is a fallback entry type with broad compatibility.
	// Use when @software is not recognized by the target bibliography style.
	EntryTypeMisc = "misc"
)

// Default values for Weaver citations.
const (
	DefaultAuthor     = "Weaver Development Team"
	DefaultTitle      = "Weaver: Multi-Agent Orchestrator for AI Research"
	DefaultURL        = "https://github.com/r3d91ll/weaver"
	DefaultLicense    = "MIT"
	DefaultRepository = "https://github.com/r3d91ll/weaver"
)

// BibTeXEntry represents a BibTeX bibliography entry.
type BibTeXEntry struct {
	// EntryType is the BibTeX entry type (@software, @misc, etc.)
	EntryType string

	// Key is the citation key used for \cite{} commands.
	Key string

	// Required fields
	Author string
	Title  string
	Year   int

	// Common optional fields
	Version      string
	URL          string
	DOI          string
	Month        string
	Note         string
	HowPublished string
	Abstract     string

	// Software-specific fields (BibLaTeX)
	License    string
	Repository string

	// Experiment traceability
	ExperimentHash string
}

// BibTeXConfig holds configuration for BibTeX generation.
type BibTeXConfig struct {
	// EntryType specifies the BibTeX entry type.
	// Default: EntryTypeSoftware
	EntryType string

	// KeyPrefix is prepended to generated citation keys.
	// Default: "weaver"
	KeyPrefix string

	// IncludeExperimentHash adds the experiment hash to the note field.
	// Default: true
	IncludeExperimentHash bool

	// IncludeTimestamp adds the access timestamp to the note field.
	// Default: false
	IncludeTimestamp bool

	// UseHowPublished uses howpublished field for URL (for @misc compatibility).
	// Default: false (uses url field)
	UseHowPublished bool

	// IncludeAbstract includes an abstract field.
	// Default: false
	IncludeAbstract bool
}

// DefaultBibTeXConfig returns a configuration with sensible defaults.
func DefaultBibTeXConfig() *BibTeXConfig {
	return &BibTeXConfig{
		EntryType:             EntryTypeSoftware,
		KeyPrefix:             "weaver",
		IncludeExperimentHash: true,
		IncludeTimestamp:      false,
		UseHowPublished:       false,
		IncludeAbstract:       false,
	}
}

// BibTeXBuilder constructs BibTeX entries with a fluent API.
type BibTeXBuilder struct {
	config *BibTeXConfig
	entry  *BibTeXEntry
}

// NewBibTeXBuilder creates a new BibTeX builder with default configuration.
func NewBibTeXBuilder() *BibTeXBuilder {
	return &BibTeXBuilder{
		config: DefaultBibTeXConfig(),
		entry: &BibTeXEntry{
			Author: DefaultAuthor,
			Title:  DefaultTitle,
			URL:    DefaultURL,
			Year:   time.Now().Year(),
		},
	}
}

// WithConfig sets the configuration for the builder.
func (bb *BibTeXBuilder) WithConfig(config *BibTeXConfig) *BibTeXBuilder {
	if config != nil {
		bb.config = config
	}
	return bb
}

// WithEntryType sets the BibTeX entry type.
func (bb *BibTeXBuilder) WithEntryType(entryType string) *BibTeXBuilder {
	bb.config.EntryType = entryType
	return bb
}

// WithKey sets the citation key.
func (bb *BibTeXBuilder) WithKey(key string) *BibTeXBuilder {
	bb.entry.Key = key
	return bb
}

// WithAuthor sets the author field.
func (bb *BibTeXBuilder) WithAuthor(author string) *BibTeXBuilder {
	bb.entry.Author = author
	return bb
}

// WithTitle sets the title field.
func (bb *BibTeXBuilder) WithTitle(title string) *BibTeXBuilder {
	bb.entry.Title = title
	return bb
}

// WithYear sets the year field.
func (bb *BibTeXBuilder) WithYear(year int) *BibTeXBuilder {
	bb.entry.Year = year
	return bb
}

// WithVersion sets the version field.
func (bb *BibTeXBuilder) WithVersion(version string) *BibTeXBuilder {
	bb.entry.Version = version
	return bb
}

// WithURL sets the URL field.
func (bb *BibTeXBuilder) WithURL(url string) *BibTeXBuilder {
	bb.entry.URL = url
	return bb
}

// WithDOI sets the DOI field.
func (bb *BibTeXBuilder) WithDOI(doi string) *BibTeXBuilder {
	bb.entry.DOI = doi
	return bb
}

// WithMonth sets the month field.
// Accepts month number (1-12) or month name.
func (bb *BibTeXBuilder) WithMonth(month string) *BibTeXBuilder {
	bb.entry.Month = month
	return bb
}

// WithNote sets the note field.
func (bb *BibTeXBuilder) WithNote(note string) *BibTeXBuilder {
	bb.entry.Note = note
	return bb
}

// WithHowPublished sets the howpublished field.
func (bb *BibTeXBuilder) WithHowPublished(howPublished string) *BibTeXBuilder {
	bb.entry.HowPublished = howPublished
	return bb
}

// WithAbstract sets the abstract field.
func (bb *BibTeXBuilder) WithAbstract(abstract string) *BibTeXBuilder {
	bb.entry.Abstract = abstract
	return bb
}

// WithLicense sets the license field (BibLaTeX software entries).
func (bb *BibTeXBuilder) WithLicense(license string) *BibTeXBuilder {
	bb.entry.License = license
	return bb
}

// WithRepository sets the repository URL field (BibLaTeX software entries).
func (bb *BibTeXBuilder) WithRepository(repository string) *BibTeXBuilder {
	bb.entry.Repository = repository
	return bb
}

// WithExperimentHash sets the experiment hash for traceability.
// The hash will be included in the note field.
func (bb *BibTeXBuilder) WithExperimentHash(hash string) *BibTeXBuilder {
	bb.entry.ExperimentHash = hash
	return bb
}

// WithExperimentHashFromHash sets the experiment hash from an ExperimentHash object.
func (bb *BibTeXBuilder) WithExperimentHashFromHash(eh *ExperimentHash) *BibTeXBuilder {
	if eh != nil {
		bb.entry.ExperimentHash = eh.ShortHash()
	}
	return bb
}

// Build generates the BibTeX entry.
func (bb *BibTeXBuilder) Build() *BibTeXEntry {
	entry := bb.entry

	// Set entry type
	if bb.config.EntryType != "" {
		entry.EntryType = bb.config.EntryType
	} else {
		entry.EntryType = EntryTypeSoftware
	}

	// Generate key if not set
	if entry.Key == "" {
		entry.Key = bb.generateKey()
	}

	return entry
}

// generateKey creates a citation key from the entry data.
// Format: prefix_year[_version][_hash]
func (bb *BibTeXBuilder) generateKey() string {
	parts := []string{bb.config.KeyPrefix}

	parts = append(parts, fmt.Sprintf("%d", bb.entry.Year))

	if bb.entry.Version != "" {
		// Sanitize version for use in key
		version := sanitizeKeyPart(bb.entry.Version)
		if version != "" {
			parts = append(parts, version)
		}
	}

	if bb.config.IncludeExperimentHash && bb.entry.ExperimentHash != "" {
		parts = append(parts, bb.entry.ExperimentHash)
	}

	return strings.Join(parts, "_")
}

// sanitizeKeyPart removes or replaces invalid characters for BibTeX keys.
// Valid characters: letters, digits, underscore, hyphen, colon, period.
func sanitizeKeyPart(s string) string {
	var sb strings.Builder
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' || r == '-' {
			sb.WriteRune(r)
		}
	}
	return sb.String()
}

// String generates the BibTeX formatted entry string.
func (entry *BibTeXEntry) String() string {
	return entry.Format(DefaultBibTeXConfig())
}

// Format generates the BibTeX formatted entry string with the given configuration.
func (entry *BibTeXEntry) Format(config *BibTeXConfig) string {
	if config == nil {
		config = DefaultBibTeXConfig()
	}

	var sb strings.Builder

	// Entry type and key
	entryType := entry.EntryType
	if entryType == "" {
		entryType = config.EntryType
	}
	if entryType == "" {
		entryType = EntryTypeSoftware
	}

	sb.WriteString("@")
	sb.WriteString(entryType)
	sb.WriteString("{")
	sb.WriteString(entry.Key)
	sb.WriteString(",\n")

	// Required fields
	writeField(&sb, "author", entry.Author)
	writeField(&sb, "title", escapeBibTeX(entry.Title))
	writeField(&sb, "year", fmt.Sprintf("%d", entry.Year))

	// Optional fields - URL handling
	if config.UseHowPublished && entry.URL != "" {
		writeField(&sb, "howpublished", fmt.Sprintf("\\url{%s}", entry.URL))
	} else if entry.URL != "" {
		writeField(&sb, "url", entry.URL)
	}

	// DOI (preferred over URL in academic contexts)
	if entry.DOI != "" {
		writeField(&sb, "doi", entry.DOI)
	}

	// Version
	if entry.Version != "" {
		writeField(&sb, "version", entry.Version)
	}

	// Month
	if entry.Month != "" {
		writeField(&sb, "month", entry.Month)
	}

	// License (for @software entries)
	if entry.License != "" && entryType == EntryTypeSoftware {
		writeField(&sb, "license", entry.License)
	}

	// Repository (for @software entries)
	if entry.Repository != "" && entryType == EntryTypeSoftware {
		writeField(&sb, "repository", entry.Repository)
	}

	// HowPublished (if not used for URL)
	if !config.UseHowPublished && entry.HowPublished != "" {
		writeField(&sb, "howpublished", entry.HowPublished)
	}

	// Abstract
	if config.IncludeAbstract && entry.Abstract != "" {
		writeField(&sb, "abstract", escapeBibTeX(entry.Abstract))
	}

	// Note field (may include experiment hash and timestamp)
	note := entry.buildNoteField(config)
	if note != "" {
		writeField(&sb, "note", escapeBibTeX(note))
	}

	sb.WriteString("}\n")

	return sb.String()
}

// buildNoteField constructs the note field from various sources.
func (entry *BibTeXEntry) buildNoteField(config *BibTeXConfig) string {
	var parts []string

	// User-provided note
	if entry.Note != "" {
		parts = append(parts, entry.Note)
	}

	// Experiment hash
	if config.IncludeExperimentHash && entry.ExperimentHash != "" {
		parts = append(parts, fmt.Sprintf("Experiment hash: %s", entry.ExperimentHash))
	}

	// Access timestamp
	if config.IncludeTimestamp {
		timestamp := time.Now().Format("2006-01-02")
		parts = append(parts, fmt.Sprintf("Accessed: %s", timestamp))
	}

	return strings.Join(parts, ". ")
}

// writeField writes a BibTeX field to the string builder.
func writeField(sb *strings.Builder, name, value string) {
	if value == "" {
		return
	}
	sb.WriteString("  ")
	sb.WriteString(name)
	sb.WriteString(" = {")
	sb.WriteString(value)
	sb.WriteString("},\n")
}

// BibTeX special characters that need escaping.
// In BibTeX, most special characters are handled within braces,
// but some still need escaping.
var bibTeXSpecialChars = map[rune]string{
	'&': `\&`,
	'%': `\%`,
	'$': `\$`,
	'#': `\#`,
	'_': `\_`,
	'{': `\{`,
	'}': `\}`,
	'~': `\textasciitilde{}`,
	'^': `\textasciicircum{}`,
}

// Unicode to LaTeX mappings for common characters.
var unicodeToLaTeX = map[rune]string{
	'\u00e0': `\`{a}`,  // à
	'\u00e1': `\'{a}`,  // á
	'\u00e2': `\^{a}`,  // â
	'\u00e4': `\"{a}`,  // ä
	'\u00e8': `\`{e}`,  // è
	'\u00e9': `\'{e}`,  // é
	'\u00ea': `\^{e}`,  // ê
	'\u00eb': `\"{e}`,  // ë
	'\u00ec': `\`{i}`,  // ì
	'\u00ed': `\'{i}`,  // í
	'\u00ee': `\^{i}`,  // î
	'\u00ef': `\"{i}`,  // ï
	'\u00f2': `\`{o}`,  // ò
	'\u00f3': `\'{o}`,  // ó
	'\u00f4': `\^{o}`,  // ô
	'\u00f6': `\"{o}`,  // ö
	'\u00f9': `\`{u}`,  // ù
	'\u00fa': `\'{u}`,  // ú
	'\u00fb': `\^{u}`,  // û
	'\u00fc': `\"{u}`,  // ü
	'\u00f1': `\~{n}`,  // ñ
	'\u00e7': `\c{c}`,  // ç
	'\u00df': `{\ss}`,  // ß
	'\u00c0': `\`{A}`,  // À
	'\u00c1': `\'{A}`,  // Á
	'\u00c2': `\^{A}`,  // Â
	'\u00c4': `\"{A}`,  // Ä
	'\u00c8': `\`{E}`,  // È
	'\u00c9': `\'{E}`,  // É
	'\u00ca': `\^{E}`,  // Ê
	'\u00cb': `\"{E}`,  // Ë
	'\u00cc': `\`{I}`,  // Ì
	'\u00cd': `\'{I}`,  // Í
	'\u00ce': `\^{I}`,  // Î
	'\u00cf': `\"{I}`,  // Ï
	'\u00d2': `\`{O}`,  // Ò
	'\u00d3': `\'{O}`,  // Ó
	'\u00d4': `\^{O}`,  // Ô
	'\u00d6': `\"{O}`,  // Ö
	'\u00d9': `\`{U}`,  // Ù
	'\u00da': `\'{U}`,  // Ú
	'\u00db': `\^{U}`,  // Û
	'\u00dc': `\"{U}`,  // Ü
	'\u00d1': `\~{N}`,  // Ñ
	'\u00c7': `\c{C}`,  // Ç
}

// escapeBibTeX escapes special characters for BibTeX field values.
// Note: BibTeX is generally more permissive within braces, but we escape
// for maximum compatibility across different BibTeX/BibLaTeX engines.
func escapeBibTeX(s string) string {
	if s == "" {
		return ""
	}

	var sb strings.Builder
	sb.Grow(len(s) * 2)

	for _, r := range s {
		// Check for special characters that need escaping
		if escaped, ok := bibTeXSpecialChars[r]; ok {
			sb.WriteString(escaped)
			continue
		}

		// Check for Unicode characters that need LaTeX encoding
		if escaped, ok := unicodeToLaTeX[r]; ok {
			sb.WriteString(escaped)
			continue
		}

		// Backslash needs special handling - only escape if not part of a command
		if r == '\\' {
			sb.WriteString(`\textbackslash{}`)
			continue
		}

		sb.WriteRune(r)
	}

	return sb.String()
}

// EscapeBibTeX is the public function for escaping BibTeX text.
func EscapeBibTeX(s string) string {
	return escapeBibTeX(s)
}

// ValidateBibTeXEntry checks if a BibTeX entry has all required fields.
// Returns a list of validation errors, or nil if valid.
func ValidateBibTeXEntry(entry *BibTeXEntry) []string {
	var errors []string

	if entry.Key == "" {
		errors = append(errors, "missing required field: key")
	}

	if entry.Author == "" {
		errors = append(errors, "missing required field: author")
	}

	if entry.Title == "" {
		errors = append(errors, "missing required field: title")
	}

	if entry.Year == 0 {
		errors = append(errors, "missing required field: year")
	}

	if entry.URL == "" && entry.DOI == "" {
		errors = append(errors, "missing required field: url or doi (at least one required)")
	}

	// Validate year is reasonable
	if entry.Year < 1900 || entry.Year > time.Now().Year()+1 {
		errors = append(errors, fmt.Sprintf("invalid year: %d", entry.Year))
	}

	// Validate entry type
	validTypes := map[string]bool{
		"software": true,
		"misc":     true,
		"article":  true,
		"book":     true,
		"inbook":   true,
		"manual":   true,
		"online":   true,
	}
	if entry.EntryType != "" && !validTypes[strings.ToLower(entry.EntryType)] {
		errors = append(errors, fmt.Sprintf("unrecognized entry type: %s", entry.EntryType))
	}

	// Validate key format (alphanumeric, underscore, hyphen, colon, period)
	keyPattern := regexp.MustCompile(`^[a-zA-Z][a-zA-Z0-9_:\-.]*$`)
	if entry.Key != "" && !keyPattern.MatchString(entry.Key) {
		errors = append(errors, fmt.Sprintf("invalid key format: %s (must start with letter, contain only letters, digits, _, :, -, .)", entry.Key))
	}

	if len(errors) > 0 {
		return errors
	}
	return nil
}

// ExportBibTeXToWriter writes a BibTeX entry to a writer.
func ExportBibTeXToWriter(w io.Writer, entry *BibTeXEntry, config *BibTeXConfig) error {
	if entry == nil {
		return fmt.Errorf("entry is nil")
	}

	content := entry.Format(config)
	_, err := w.Write([]byte(content))
	if err != nil {
		return fmt.Errorf("failed to write BibTeX entry: %w", err)
	}

	return nil
}

// ExportBibTeXToFile writes a BibTeX entry to a file.
func ExportBibTeXToFile(path string, entry *BibTeXEntry, config *BibTeXConfig) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()

	return ExportBibTeXToWriter(file, entry, config)
}

// ExportBibTeXEntries writes multiple BibTeX entries to a writer.
func ExportBibTeXEntries(w io.Writer, entries []*BibTeXEntry, config *BibTeXConfig) error {
	for i, entry := range entries {
		if i > 0 {
			if _, err := w.Write([]byte("\n")); err != nil {
				return fmt.Errorf("failed to write separator: %w", err)
			}
		}
		if err := ExportBibTeXToWriter(w, entry, config); err != nil {
			return err
		}
	}
	return nil
}

// GenerateWeaverCitation generates a BibTeX citation for Weaver.
// This is a convenience function for the common use case.
func GenerateWeaverCitation(version string, experimentHash *ExperimentHash) *BibTeXEntry {
	builder := NewBibTeXBuilder().
		WithVersion(version).
		WithLicense(DefaultLicense).
		WithRepository(DefaultRepository)

	if experimentHash != nil {
		builder.WithExperimentHashFromHash(experimentHash)
	}

	return builder.Build()
}

// GenerateWeaverCitationWithConfig generates a BibTeX citation with custom configuration.
func GenerateWeaverCitationWithConfig(version string, experimentHash *ExperimentHash, config *BibTeXConfig) *BibTeXEntry {
	builder := NewBibTeXBuilder().
		WithConfig(config).
		WithVersion(version).
		WithLicense(DefaultLicense).
		WithRepository(DefaultRepository)

	if experimentHash != nil {
		builder.WithExperimentHashFromHash(experimentHash)
	}

	return builder.Build()
}

// GenerateBibTeXFile generates a complete .bib file with header comments.
func GenerateBibTeXFile(entries []*BibTeXEntry, config *BibTeXConfig) string {
	var sb strings.Builder

	// Header comment
	sb.WriteString("% BibTeX entries generated by Weaver\n")
	sb.WriteString(fmt.Sprintf("%% Generated: %s\n", time.Now().Format(time.RFC3339)))
	sb.WriteString("% For use with BibLaTeX or traditional BibTeX\n")
	sb.WriteString("%\n")
	sb.WriteString("% If using @software entry type, ensure your document preamble includes:\n")
	sb.WriteString("%   \\usepackage[backend=biber]{biblatex}\n")
	sb.WriteString("% Or use @misc for broader compatibility.\n")
	sb.WriteString("\n")

	// Sort entries by key for consistent output
	sortedEntries := make([]*BibTeXEntry, len(entries))
	copy(sortedEntries, entries)
	sort.Slice(sortedEntries, func(i, j int) bool {
		return sortedEntries[i].Key < sortedEntries[j].Key
	})

	// Write entries
	for i, entry := range sortedEntries {
		if i > 0 {
			sb.WriteString("\n")
		}
		sb.WriteString(entry.Format(config))
	}

	return sb.String()
}
