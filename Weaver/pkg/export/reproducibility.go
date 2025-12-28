// Package export provides academic export format utilities.
// Generates LaTeX tables, CSV data, and other publication-ready outputs.
package export

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"time"
)

// ReportFormat identifies the output format for reproducibility reports.
type ReportFormat string

const (
	// FormatMarkdown generates a Markdown-formatted report.
	FormatMarkdown ReportFormat = "markdown"

	// FormatLaTeX generates a LaTeX-formatted report.
	FormatLaTeX ReportFormat = "latex"

	// FormatJSON generates a JSON-formatted report.
	FormatJSON ReportFormat = "json"
)

// IsValid returns true if this is a valid report format.
func (rf ReportFormat) IsValid() bool {
	switch rf {
	case FormatMarkdown, FormatLaTeX, FormatJSON:
		return true
	default:
		return false
	}
}

// AgentConfig represents the configuration of an agent in the experiment.
type AgentConfig struct {
	// ID is the unique identifier for the agent.
	ID string `json:"id"`

	// Name is the display name of the agent.
	Name string `json:"name"`

	// Type identifies the agent type (e.g., "human", "llm", "system").
	Type string `json:"type"`

	// Model is the model identifier for LLM agents.
	Model string `json:"model,omitempty"`

	// Provider is the provider of the agent (e.g., "openai", "anthropic").
	Provider string `json:"provider,omitempty"`

	// Parameters holds agent-specific configuration parameters.
	Parameters map[string]string `json:"parameters,omitempty"`
}

// DataSource describes a data input used in the experiment.
type DataSource struct {
	// Name is the identifier for this data source.
	Name string `json:"name"`

	// Type indicates the type of data source (e.g., "file", "api", "stream").
	Type string `json:"type"`

	// Path is the file path or URI of the data source.
	Path string `json:"path,omitempty"`

	// Description provides additional context about the data source.
	Description string `json:"description,omitempty"`

	// Hash is an optional hash of the data source content.
	Hash string `json:"hash,omitempty"`

	// RecordCount is the number of records in the data source.
	RecordCount int `json:"record_count,omitempty"`
}

// ReproducibilityConfig holds configuration for report generation.
type ReproducibilityConfig struct {
	// Format specifies the output format (markdown, latex, json).
	// Default: FormatMarkdown
	Format ReportFormat `json:"format"`

	// IncludeAgentConfigs includes agent configuration details.
	// Default: true
	IncludeAgentConfigs bool `json:"include_agent_configs"`

	// IncludeDataSources includes data source information.
	// Default: true
	IncludeDataSources bool `json:"include_data_sources"`

	// IncludeParameters includes all experiment parameters.
	// Default: true
	IncludeParameters bool `json:"include_parameters"`

	// IncludeEnvironment includes environment information.
	// Default: false
	IncludeEnvironment bool `json:"include_environment"`

	// IncludeVerificationInstructions adds verification instructions.
	// Default: true
	IncludeVerificationInstructions bool `json:"include_verification_instructions"`

	// Title is the report title.
	// Default: "Experiment Reproducibility Report"
	Title string `json:"title"`

	// Author is the report author.
	Author string `json:"author,omitempty"`

	// SessionName is the name of the session being reported.
	SessionName string `json:"session_name,omitempty"`

	// SessionDescription is a description of the session.
	SessionDescription string `json:"session_description,omitempty"`
}

// DefaultReproducibilityConfig returns a configuration with sensible defaults.
func DefaultReproducibilityConfig() *ReproducibilityConfig {
	return &ReproducibilityConfig{
		Format:                          FormatMarkdown,
		IncludeAgentConfigs:             true,
		IncludeDataSources:              true,
		IncludeParameters:               true,
		IncludeEnvironment:              false,
		IncludeVerificationInstructions: true,
		Title:                           "Experiment Reproducibility Report",
	}
}

// ReproducibilityReport represents a complete reproducibility report.
type ReproducibilityReport struct {
	// Title is the report title.
	Title string `json:"title"`

	// Author is the report author.
	Author string `json:"author,omitempty"`

	// GeneratedAt is when the report was generated.
	GeneratedAt time.Time `json:"generated_at"`

	// Session information
	SessionID          string `json:"session_id,omitempty"`
	SessionName        string `json:"session_name,omitempty"`
	SessionDescription string `json:"session_description,omitempty"`

	// Tool information
	ToolVersion string `json:"tool_version"`

	// Experiment timing
	StartTime time.Time  `json:"start_time"`
	EndTime   *time.Time `json:"end_time,omitempty"`
	Duration  string     `json:"duration,omitempty"`

	// Configuration
	MeasurementMode   string `json:"measurement_mode"`
	MeasurementCount  int    `json:"measurement_count"`
	ConversationCount int    `json:"conversation_count"`

	// Agents
	Agents []AgentConfig `json:"agents,omitempty"`

	// Data sources
	DataSources []DataSource `json:"data_sources,omitempty"`

	// Parameters
	Parameters map[string]string `json:"parameters,omitempty"`

	// Environment
	Environment map[string]string `json:"environment,omitempty"`

	// Experiment hash
	ExperimentHash *ExperimentHash `json:"experiment_hash"`
}

// ReportBuilder constructs reproducibility reports with a fluent API.
type ReportBuilder struct {
	config *ReproducibilityConfig
	report *ReproducibilityReport
}

// NewReportBuilder creates a new ReportBuilder with default configuration.
func NewReportBuilder() *ReportBuilder {
	return &ReportBuilder{
		config: DefaultReproducibilityConfig(),
		report: &ReproducibilityReport{
			Agents:      make([]AgentConfig, 0),
			DataSources: make([]DataSource, 0),
			Parameters:  make(map[string]string),
			Environment: make(map[string]string),
			GeneratedAt: time.Now(),
		},
	}
}

// WithConfig sets the configuration for the builder.
func (rb *ReportBuilder) WithConfig(config *ReproducibilityConfig) *ReportBuilder {
	if config != nil {
		rb.config = config
	}
	return rb
}

// WithFormat sets the output format.
func (rb *ReportBuilder) WithFormat(format ReportFormat) *ReportBuilder {
	rb.config.Format = format
	return rb
}

// WithTitle sets the report title.
func (rb *ReportBuilder) WithTitle(title string) *ReportBuilder {
	rb.config.Title = title
	rb.report.Title = title
	return rb
}

// WithAuthor sets the report author.
func (rb *ReportBuilder) WithAuthor(author string) *ReportBuilder {
	rb.config.Author = author
	rb.report.Author = author
	return rb
}

// WithSessionID sets the session ID.
func (rb *ReportBuilder) WithSessionID(id string) *ReportBuilder {
	rb.report.SessionID = id
	return rb
}

// WithSessionName sets the session name.
func (rb *ReportBuilder) WithSessionName(name string) *ReportBuilder {
	rb.config.SessionName = name
	rb.report.SessionName = name
	return rb
}

// WithSessionDescription sets the session description.
func (rb *ReportBuilder) WithSessionDescription(description string) *ReportBuilder {
	rb.config.SessionDescription = description
	rb.report.SessionDescription = description
	return rb
}

// WithToolVersion sets the tool version.
func (rb *ReportBuilder) WithToolVersion(version string) *ReportBuilder {
	rb.report.ToolVersion = version
	return rb
}

// WithTimeRange sets the experiment time range.
func (rb *ReportBuilder) WithTimeRange(startTime time.Time, endTime *time.Time) *ReportBuilder {
	rb.report.StartTime = startTime
	rb.report.EndTime = endTime

	// Calculate duration if both times are available
	if endTime != nil && !endTime.IsZero() && !startTime.IsZero() {
		duration := endTime.Sub(startTime)
		rb.report.Duration = formatDuration(duration)
	}

	return rb
}

// formatDuration formats a duration in a human-readable format.
func formatDuration(d time.Duration) string {
	if d < time.Second {
		return fmt.Sprintf("%dms", d.Milliseconds())
	}
	if d < time.Minute {
		return fmt.Sprintf("%.1fs", d.Seconds())
	}
	if d < time.Hour {
		minutes := int(d.Minutes())
		seconds := int(d.Seconds()) % 60
		return fmt.Sprintf("%dm %ds", minutes, seconds)
	}
	hours := int(d.Hours())
	minutes := int(d.Minutes()) % 60
	return fmt.Sprintf("%dh %dm", hours, minutes)
}

// WithMeasurementMode sets the measurement mode.
func (rb *ReportBuilder) WithMeasurementMode(mode string) *ReportBuilder {
	rb.report.MeasurementMode = mode
	return rb
}

// WithMeasurementCount sets the measurement count.
func (rb *ReportBuilder) WithMeasurementCount(count int) *ReportBuilder {
	rb.report.MeasurementCount = count
	return rb
}

// WithConversationCount sets the conversation count.
func (rb *ReportBuilder) WithConversationCount(count int) *ReportBuilder {
	rb.report.ConversationCount = count
	return rb
}

// WithAgent adds an agent configuration.
func (rb *ReportBuilder) WithAgent(agent AgentConfig) *ReportBuilder {
	rb.report.Agents = append(rb.report.Agents, agent)
	return rb
}

// WithAgents adds multiple agent configurations.
func (rb *ReportBuilder) WithAgents(agents []AgentConfig) *ReportBuilder {
	rb.report.Agents = append(rb.report.Agents, agents...)
	return rb
}

// WithDataSource adds a data source.
func (rb *ReportBuilder) WithDataSource(ds DataSource) *ReportBuilder {
	rb.report.DataSources = append(rb.report.DataSources, ds)
	return rb
}

// WithDataSources adds multiple data sources.
func (rb *ReportBuilder) WithDataSources(dataSources []DataSource) *ReportBuilder {
	rb.report.DataSources = append(rb.report.DataSources, dataSources...)
	return rb
}

// WithParameter adds a configuration parameter.
func (rb *ReportBuilder) WithParameter(key, value string) *ReportBuilder {
	rb.report.Parameters[key] = value
	return rb
}

// WithParameters adds multiple configuration parameters.
func (rb *ReportBuilder) WithParameters(params map[string]string) *ReportBuilder {
	for k, v := range params {
		rb.report.Parameters[k] = v
	}
	return rb
}

// WithEnvironmentVar adds an environment variable.
func (rb *ReportBuilder) WithEnvironmentVar(key, value string) *ReportBuilder {
	rb.report.Environment[key] = value
	return rb
}

// WithEnvironment adds multiple environment variables.
func (rb *ReportBuilder) WithEnvironment(env map[string]string) *ReportBuilder {
	for k, v := range env {
		rb.report.Environment[k] = v
	}
	return rb
}

// WithExperimentHash sets the experiment hash.
func (rb *ReportBuilder) WithExperimentHash(hash *ExperimentHash) *ReportBuilder {
	rb.report.ExperimentHash = hash
	return rb
}

// Build generates the reproducibility report.
// If no experiment hash is set, it computes one from the report data.
func (rb *ReportBuilder) Build() *ReproducibilityReport {
	report := rb.report

	// Set title from config if not already set
	if report.Title == "" {
		report.Title = rb.config.Title
	}

	// Set author from config if not already set
	if report.Author == "" && rb.config.Author != "" {
		report.Author = rb.config.Author
	}

	// Compute experiment hash if not set
	if report.ExperimentHash == nil {
		report.ExperimentHash = rb.computeHash()
	}

	return report
}

// computeHash generates an experiment hash from the report data.
func (rb *ReportBuilder) computeHash() *ExperimentHash {
	builder := NewHashBuilder().
		WithToolVersion(rb.report.ToolVersion).
		WithMeasurementMode(rb.report.MeasurementMode).
		WithMeasurementCount(rb.report.MeasurementCount).
		WithConversationCount(rb.report.ConversationCount).
		WithTimeRange(rb.report.StartTime, rb.report.EndTime)

	// Add parameters
	if len(rb.report.Parameters) > 0 {
		builder.WithParameters(rb.report.Parameters)
	}

	return builder.Build()
}

// Format generates the report in the specified format.
func (r *ReproducibilityReport) Format(config *ReproducibilityConfig) string {
	if config == nil {
		config = DefaultReproducibilityConfig()
	}

	switch config.Format {
	case FormatLaTeX:
		return r.formatLaTeX(config)
	case FormatJSON:
		return r.formatJSON()
	default:
		return r.formatMarkdown(config)
	}
}

// formatMarkdown generates a Markdown-formatted report.
func (r *ReproducibilityReport) formatMarkdown(config *ReproducibilityConfig) string {
	var sb strings.Builder

	// Title
	sb.WriteString("# ")
	sb.WriteString(r.Title)
	sb.WriteString("\n\n")

	// Author and date
	if r.Author != "" {
		sb.WriteString("**Author:** ")
		sb.WriteString(r.Author)
		sb.WriteString("\n\n")
	}
	sb.WriteString("**Generated:** ")
	sb.WriteString(r.GeneratedAt.UTC().Format(time.RFC3339))
	sb.WriteString("\n\n")

	// Session information
	if r.SessionID != "" || r.SessionName != "" || r.SessionDescription != "" {
		sb.WriteString("## Session Information\n\n")
		if r.SessionID != "" {
			sb.WriteString("- **Session ID:** `")
			sb.WriteString(r.SessionID)
			sb.WriteString("`\n")
		}
		if r.SessionName != "" {
			sb.WriteString("- **Session Name:** ")
			sb.WriteString(r.SessionName)
			sb.WriteString("\n")
		}
		if r.SessionDescription != "" {
			sb.WriteString("- **Description:** ")
			sb.WriteString(r.SessionDescription)
			sb.WriteString("\n")
		}
		sb.WriteString("\n")
	}

	// Experiment hash section
	sb.WriteString("## Experiment Hash\n\n")
	if r.ExperimentHash != nil {
		sb.WriteString("```\n")
		sb.WriteString("Hash:      ")
		sb.WriteString(r.ExperimentHash.Hash)
		sb.WriteString("\n")
		sb.WriteString("Short:     ")
		sb.WriteString(r.ExperimentHash.ShortHash())
		sb.WriteString("\n")
		sb.WriteString("Algorithm: ")
		sb.WriteString(r.ExperimentHash.Algorithm)
		sb.WriteString("\n")
		sb.WriteString("Computed:  ")
		sb.WriteString(r.ExperimentHash.ComputedAt.UTC().Format(time.RFC3339))
		sb.WriteString("\n")
		sb.WriteString("```\n\n")
	} else {
		sb.WriteString("*No experiment hash available*\n\n")
	}

	// Tool version
	sb.WriteString("## Tool Information\n\n")
	sb.WriteString("- **Tool Version:** ")
	if r.ToolVersion != "" {
		sb.WriteString("`")
		sb.WriteString(r.ToolVersion)
		sb.WriteString("`")
	} else {
		sb.WriteString("*Unknown*")
	}
	sb.WriteString("\n\n")

	// Experiment timing
	sb.WriteString("## Experiment Timing\n\n")
	sb.WriteString("| Metric | Value |\n")
	sb.WriteString("|--------|-------|\n")
	sb.WriteString("| Start Time | ")
	if !r.StartTime.IsZero() {
		sb.WriteString(r.StartTime.UTC().Format(time.RFC3339))
	} else {
		sb.WriteString("*Not recorded*")
	}
	sb.WriteString(" |\n")
	sb.WriteString("| End Time | ")
	if r.EndTime != nil && !r.EndTime.IsZero() {
		sb.WriteString(r.EndTime.UTC().Format(time.RFC3339))
	} else {
		sb.WriteString("*In progress*")
	}
	sb.WriteString(" |\n")
	if r.Duration != "" {
		sb.WriteString("| Duration | ")
		sb.WriteString(r.Duration)
		sb.WriteString(" |\n")
	}
	sb.WriteString("\n")

	// Configuration section
	sb.WriteString("## Configuration\n\n")
	sb.WriteString("| Parameter | Value |\n")
	sb.WriteString("|-----------|-------|\n")
	sb.WriteString("| Measurement Mode | ")
	if r.MeasurementMode != "" {
		sb.WriteString("`")
		sb.WriteString(r.MeasurementMode)
		sb.WriteString("`")
	} else {
		sb.WriteString("*Default*")
	}
	sb.WriteString(" |\n")
	sb.WriteString(fmt.Sprintf("| Measurement Count | %d |\n", r.MeasurementCount))
	sb.WriteString(fmt.Sprintf("| Conversation Count | %d |\n", r.ConversationCount))
	sb.WriteString("\n")

	// Agent configurations
	if config.IncludeAgentConfigs && len(r.Agents) > 0 {
		sb.WriteString("## Agent Configurations\n\n")
		for i, agent := range r.Agents {
			sb.WriteString(fmt.Sprintf("### Agent %d: %s\n\n", i+1, agent.Name))
			sb.WriteString("| Property | Value |\n")
			sb.WriteString("|----------|-------|\n")
			sb.WriteString(fmt.Sprintf("| ID | `%s` |\n", agent.ID))
			sb.WriteString(fmt.Sprintf("| Type | %s |\n", agent.Type))
			if agent.Model != "" {
				sb.WriteString(fmt.Sprintf("| Model | %s |\n", agent.Model))
			}
			if agent.Provider != "" {
				sb.WriteString(fmt.Sprintf("| Provider | %s |\n", agent.Provider))
			}
			sb.WriteString("\n")

			// Agent-specific parameters
			if len(agent.Parameters) > 0 {
				sb.WriteString("**Agent Parameters:**\n\n")
				sb.WriteString("| Parameter | Value |\n")
				sb.WriteString("|-----------|-------|\n")
				// Sort parameters for deterministic output
				keys := sortedKeys(agent.Parameters)
				for _, k := range keys {
					sb.WriteString(fmt.Sprintf("| %s | %s |\n", k, agent.Parameters[k]))
				}
				sb.WriteString("\n")
			}
		}
	}

	// Data sources
	if config.IncludeDataSources && len(r.DataSources) > 0 {
		sb.WriteString("## Data Sources\n\n")
		for i, ds := range r.DataSources {
			sb.WriteString(fmt.Sprintf("### Source %d: %s\n\n", i+1, ds.Name))
			sb.WriteString("| Property | Value |\n")
			sb.WriteString("|----------|-------|\n")
			sb.WriteString(fmt.Sprintf("| Type | %s |\n", ds.Type))
			if ds.Path != "" {
				sb.WriteString(fmt.Sprintf("| Path | `%s` |\n", ds.Path))
			}
			if ds.Description != "" {
				sb.WriteString(fmt.Sprintf("| Description | %s |\n", ds.Description))
			}
			if ds.Hash != "" {
				sb.WriteString(fmt.Sprintf("| Hash | `%s` |\n", ds.Hash))
			}
			if ds.RecordCount > 0 {
				sb.WriteString(fmt.Sprintf("| Record Count | %d |\n", ds.RecordCount))
			}
			sb.WriteString("\n")
		}
	}

	// Parameters
	if config.IncludeParameters && len(r.Parameters) > 0 {
		sb.WriteString("## Experiment Parameters\n\n")
		sb.WriteString("| Parameter | Value |\n")
		sb.WriteString("|-----------|-------|\n")
		// Sort parameters for deterministic output
		keys := sortedKeys(r.Parameters)
		for _, k := range keys {
			sb.WriteString(fmt.Sprintf("| %s | %s |\n", k, r.Parameters[k]))
		}
		sb.WriteString("\n")
	}

	// Environment
	if config.IncludeEnvironment && len(r.Environment) > 0 {
		sb.WriteString("## Environment\n\n")
		sb.WriteString("| Variable | Value |\n")
		sb.WriteString("|----------|-------|\n")
		// Sort for deterministic output
		keys := sortedKeys(r.Environment)
		for _, k := range keys {
			sb.WriteString(fmt.Sprintf("| %s | %s |\n", k, r.Environment[k]))
		}
		sb.WriteString("\n")
	}

	// Verification instructions
	if config.IncludeVerificationInstructions {
		sb.WriteString("## Verification\n\n")
		sb.WriteString("To verify this experiment's reproducibility:\n\n")
		sb.WriteString("1. **Hash Verification**: The experiment hash is computed from:\n")
		sb.WriteString("   - Tool version\n")
		sb.WriteString("   - Measurement mode\n")
		sb.WriteString("   - Measurement count\n")
		sb.WriteString("   - Conversation count\n")
		sb.WriteString("   - Start/end timestamps\n")
		sb.WriteString("   - All experiment parameters (sorted alphabetically)\n\n")
		sb.WriteString("2. **Algorithm**: SHA-256 hash of canonical string representation\n\n")
		sb.WriteString("3. **Recomputation**: Use the following configuration to recompute the hash:\n\n")
		sb.WriteString("```json\n")
		sb.WriteString(r.formatHashConfig())
		sb.WriteString("```\n\n")
		sb.WriteString("4. **Expected Result**: The recomputed hash should match:\n")
		sb.WriteString("   ```\n")
		if r.ExperimentHash != nil {
			sb.WriteString("   ")
			sb.WriteString(r.ExperimentHash.Hash)
		} else {
			sb.WriteString("   *No hash available*")
		}
		sb.WriteString("\n   ```\n\n")
	}

	// Footer
	sb.WriteString("---\n\n")
	sb.WriteString("*This report was generated by Weaver for academic reproducibility.*\n")

	return sb.String()
}

// formatHashConfig generates a JSON representation of the hash configuration.
func (r *ReproducibilityReport) formatHashConfig() string {
	config := struct {
		ToolVersion       string            `json:"tool_version"`
		MeasurementMode   string            `json:"measurement_mode"`
		MeasurementCount  int               `json:"measurement_count"`
		ConversationCount int               `json:"conversation_count"`
		StartTime         string            `json:"start_time"`
		EndTime           string            `json:"end_time,omitempty"`
		Parameters        map[string]string `json:"parameters,omitempty"`
	}{
		ToolVersion:       r.ToolVersion,
		MeasurementMode:   r.MeasurementMode,
		MeasurementCount:  r.MeasurementCount,
		ConversationCount: r.ConversationCount,
		StartTime:         r.StartTime.UTC().Format(time.RFC3339),
		Parameters:        r.Parameters,
	}

	if r.EndTime != nil && !r.EndTime.IsZero() {
		config.EndTime = r.EndTime.UTC().Format(time.RFC3339)
	}

	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return "{}"
	}
	return string(data)
}

// formatLaTeX generates a LaTeX-formatted report.
func (r *ReproducibilityReport) formatLaTeX(config *ReproducibilityConfig) string {
	var sb strings.Builder

	// Document header
	sb.WriteString("% Reproducibility Report generated by Weaver\n")
	sb.WriteString(fmt.Sprintf("%% Generated: %s\n", r.GeneratedAt.UTC().Format(time.RFC3339)))
	sb.WriteString("% Include this in your LaTeX document\n\n")

	// Section
	sb.WriteString("\\section{Reproducibility Report}\n\n")

	// Title and metadata
	if r.Author != "" {
		sb.WriteString(fmt.Sprintf("\\noindent\\textbf{Author:} %s\\\\\n", EscapeLaTeX(r.Author)))
	}
	sb.WriteString(fmt.Sprintf("\\noindent\\textbf{Generated:} %s\n\n", r.GeneratedAt.UTC().Format(time.RFC3339)))

	// Session information
	if r.SessionID != "" || r.SessionName != "" {
		sb.WriteString("\\subsection{Session Information}\n")
		sb.WriteString("\\begin{itemize}\n")
		if r.SessionID != "" {
			sb.WriteString(fmt.Sprintf("  \\item \\textbf{Session ID:} \\texttt{%s}\n", EscapeLaTeX(r.SessionID)))
		}
		if r.SessionName != "" {
			sb.WriteString(fmt.Sprintf("  \\item \\textbf{Session Name:} %s\n", EscapeLaTeX(r.SessionName)))
		}
		if r.SessionDescription != "" {
			sb.WriteString(fmt.Sprintf("  \\item \\textbf{Description:} %s\n", EscapeLaTeX(r.SessionDescription)))
		}
		sb.WriteString("\\end{itemize}\n\n")
	}

	// Experiment hash
	sb.WriteString("\\subsection{Experiment Hash}\n")
	if r.ExperimentHash != nil {
		sb.WriteString("\\begin{verbatim}\n")
		sb.WriteString(fmt.Sprintf("Hash:      %s\n", r.ExperimentHash.Hash))
		sb.WriteString(fmt.Sprintf("Short:     %s\n", r.ExperimentHash.ShortHash()))
		sb.WriteString(fmt.Sprintf("Algorithm: %s\n", r.ExperimentHash.Algorithm))
		sb.WriteString("\\end{verbatim}\n\n")
	}

	// Configuration table
	sb.WriteString("\\subsection{Configuration}\n")
	sb.WriteString("\\begin{tabular}{ll}\n")
	sb.WriteString("\\hline\n")
	sb.WriteString("\\textbf{Parameter} & \\textbf{Value} \\\\\n")
	sb.WriteString("\\hline\n")
	sb.WriteString(fmt.Sprintf("Tool Version & \\texttt{%s} \\\\\n", EscapeLaTeX(r.ToolVersion)))
	sb.WriteString(fmt.Sprintf("Measurement Mode & \\texttt{%s} \\\\\n", EscapeLaTeX(r.MeasurementMode)))
	sb.WriteString(fmt.Sprintf("Measurement Count & %d \\\\\n", r.MeasurementCount))
	sb.WriteString(fmt.Sprintf("Conversation Count & %d \\\\\n", r.ConversationCount))
	if !r.StartTime.IsZero() {
		sb.WriteString(fmt.Sprintf("Start Time & %s \\\\\n", r.StartTime.UTC().Format(time.RFC3339)))
	}
	if r.EndTime != nil && !r.EndTime.IsZero() {
		sb.WriteString(fmt.Sprintf("End Time & %s \\\\\n", r.EndTime.UTC().Format(time.RFC3339)))
	}
	if r.Duration != "" {
		sb.WriteString(fmt.Sprintf("Duration & %s \\\\\n", EscapeLaTeX(r.Duration)))
	}
	sb.WriteString("\\hline\n")
	sb.WriteString("\\end{tabular}\n\n")

	// Parameters
	if config.IncludeParameters && len(r.Parameters) > 0 {
		sb.WriteString("\\subsection{Parameters}\n")
		sb.WriteString("\\begin{tabular}{ll}\n")
		sb.WriteString("\\hline\n")
		sb.WriteString("\\textbf{Parameter} & \\textbf{Value} \\\\\n")
		sb.WriteString("\\hline\n")
		keys := sortedKeys(r.Parameters)
		for _, k := range keys {
			sb.WriteString(fmt.Sprintf("%s & %s \\\\\n", EscapeLaTeX(k), EscapeLaTeX(r.Parameters[k])))
		}
		sb.WriteString("\\hline\n")
		sb.WriteString("\\end{tabular}\n\n")
	}

	return sb.String()
}

// formatJSON generates a JSON-formatted report.
func (r *ReproducibilityReport) formatJSON() string {
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return "{}"
	}
	return string(data)
}

// sortedKeys returns the keys of a map sorted alphabetically.
func sortedKeys(m map[string]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// ToMarkdown generates a Markdown-formatted report.
func (r *ReproducibilityReport) ToMarkdown() string {
	return r.Format(DefaultReproducibilityConfig())
}

// ToLaTeX generates a LaTeX-formatted report.
func (r *ReproducibilityReport) ToLaTeX() string {
	config := DefaultReproducibilityConfig()
	config.Format = FormatLaTeX
	return r.Format(config)
}

// ToJSON generates a JSON-formatted report.
func (r *ReproducibilityReport) ToJSON() string {
	return r.formatJSON()
}

// Verify recomputes the experiment hash and checks if it matches.
func (r *ReproducibilityReport) Verify() bool {
	if r.ExperimentHash == nil {
		return false
	}
	return r.ExperimentHash.Verify()
}

// ExportReportToWriter writes the report to a writer.
func ExportReportToWriter(w io.Writer, report *ReproducibilityReport, config *ReproducibilityConfig) error {
	if report == nil {
		return fmt.Errorf("report is nil")
	}

	if config == nil {
		config = DefaultReproducibilityConfig()
	}

	content := report.Format(config)
	_, err := w.Write([]byte(content))
	if err != nil {
		return fmt.Errorf("failed to write report: %w", err)
	}

	return nil
}

// ExportReportToFile writes the report to a file.
func ExportReportToFile(path string, report *ReproducibilityReport, config *ReproducibilityConfig) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()

	return ExportReportToWriter(file, report, config)
}

// GenerateReproducibilityReport is a convenience function to generate a complete report.
func GenerateReproducibilityReport(
	toolVersion string,
	sessionID string,
	sessionName string,
	measurementMode string,
	measurementCount int,
	conversationCount int,
	startTime time.Time,
	endTime *time.Time,
	agents []AgentConfig,
	dataSources []DataSource,
	params map[string]string,
) *ReproducibilityReport {
	builder := NewReportBuilder().
		WithToolVersion(toolVersion).
		WithSessionID(sessionID).
		WithSessionName(sessionName).
		WithMeasurementMode(measurementMode).
		WithMeasurementCount(measurementCount).
		WithConversationCount(conversationCount).
		WithTimeRange(startTime, endTime)

	if len(agents) > 0 {
		builder.WithAgents(agents)
	}

	if len(dataSources) > 0 {
		builder.WithDataSources(dataSources)
	}

	if params != nil {
		builder.WithParameters(params)
	}

	return builder.Build()
}
