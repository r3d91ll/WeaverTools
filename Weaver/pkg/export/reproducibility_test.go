package export

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Test fixtures
var (
	testStartTime = time.Date(2024, 1, 15, 10, 0, 0, 0, time.UTC)
	testEndTime   = time.Date(2024, 1, 15, 11, 30, 0, 0, time.UTC)
)

func testEndTimePtr() *time.Time {
	t := testEndTime
	return &t
}

func TestReportFormatIsValid(t *testing.T) {
	tests := []struct {
		format ReportFormat
		valid  bool
	}{
		{FormatMarkdown, true},
		{FormatLaTeX, true},
		{FormatJSON, true},
		{ReportFormat("invalid"), false},
		{ReportFormat(""), false},
	}

	for _, tt := range tests {
		t.Run(string(tt.format), func(t *testing.T) {
			if got := tt.format.IsValid(); got != tt.valid {
				t.Errorf("IsValid() = %v, want %v", got, tt.valid)
			}
		})
	}
}

func TestDefaultReproducibilityConfig(t *testing.T) {
	config := DefaultReproducibilityConfig()

	if config.Format != FormatMarkdown {
		t.Errorf("Format = %v, want %v", config.Format, FormatMarkdown)
	}
	if !config.IncludeAgentConfigs {
		t.Error("IncludeAgentConfigs should be true by default")
	}
	if !config.IncludeDataSources {
		t.Error("IncludeDataSources should be true by default")
	}
	if !config.IncludeParameters {
		t.Error("IncludeParameters should be true by default")
	}
	if config.IncludeEnvironment {
		t.Error("IncludeEnvironment should be false by default")
	}
	if !config.IncludeVerificationInstructions {
		t.Error("IncludeVerificationInstructions should be true by default")
	}
	if config.Title != "Experiment Reproducibility Report" {
		t.Errorf("Title = %v, want 'Experiment Reproducibility Report'", config.Title)
	}
}

func TestNewReportBuilder(t *testing.T) {
	builder := NewReportBuilder()

	if builder == nil {
		t.Fatal("NewReportBuilder() returned nil")
	}
	if builder.config == nil {
		t.Error("builder.config is nil")
	}
	if builder.report == nil {
		t.Error("builder.report is nil")
	}
	if builder.report.Agents == nil {
		t.Error("builder.report.Agents is nil")
	}
	if builder.report.DataSources == nil {
		t.Error("builder.report.DataSources is nil")
	}
	if builder.report.Parameters == nil {
		t.Error("builder.report.Parameters is nil")
	}
}

func TestReportBuilderFluentAPI(t *testing.T) {
	builder := NewReportBuilder().
		WithTitle("Test Report").
		WithAuthor("Test Author").
		WithSessionID("session-123").
		WithSessionName("Test Session").
		WithSessionDescription("A test session").
		WithToolVersion("1.0.0").
		WithMeasurementMode("active").
		WithMeasurementCount(10).
		WithConversationCount(5).
		WithTimeRange(testStartTime, testEndTimePtr())

	report := builder.Build()

	if report.Title != "Test Report" {
		t.Errorf("Title = %v, want 'Test Report'", report.Title)
	}
	if report.Author != "Test Author" {
		t.Errorf("Author = %v, want 'Test Author'", report.Author)
	}
	if report.SessionID != "session-123" {
		t.Errorf("SessionID = %v, want 'session-123'", report.SessionID)
	}
	if report.SessionName != "Test Session" {
		t.Errorf("SessionName = %v, want 'Test Session'", report.SessionName)
	}
	if report.ToolVersion != "1.0.0" {
		t.Errorf("ToolVersion = %v, want '1.0.0'", report.ToolVersion)
	}
	if report.MeasurementMode != "active" {
		t.Errorf("MeasurementMode = %v, want 'active'", report.MeasurementMode)
	}
	if report.MeasurementCount != 10 {
		t.Errorf("MeasurementCount = %v, want 10", report.MeasurementCount)
	}
	if report.ConversationCount != 5 {
		t.Errorf("ConversationCount = %v, want 5", report.ConversationCount)
	}
	if report.Duration == "" {
		t.Error("Duration should be set when both start and end times are provided")
	}
}

func TestReportBuilderWithAgents(t *testing.T) {
	agent := AgentConfig{
		ID:       "agent-1",
		Name:     "Claude",
		Type:     "llm",
		Model:    "claude-3-opus",
		Provider: "anthropic",
		Parameters: map[string]string{
			"temperature": "0.7",
			"max_tokens":  "1000",
		},
	}

	builder := NewReportBuilder().
		WithAgent(agent).
		WithToolVersion("1.0.0").
		WithTimeRange(testStartTime, nil)

	report := builder.Build()

	if len(report.Agents) != 1 {
		t.Fatalf("len(Agents) = %d, want 1", len(report.Agents))
	}
	if report.Agents[0].ID != "agent-1" {
		t.Errorf("Agents[0].ID = %v, want 'agent-1'", report.Agents[0].ID)
	}
	if report.Agents[0].Model != "claude-3-opus" {
		t.Errorf("Agents[0].Model = %v, want 'claude-3-opus'", report.Agents[0].Model)
	}
}

func TestReportBuilderWithMultipleAgents(t *testing.T) {
	agents := []AgentConfig{
		{ID: "agent-1", Name: "Agent 1", Type: "llm"},
		{ID: "agent-2", Name: "Agent 2", Type: "human"},
	}

	builder := NewReportBuilder().
		WithAgents(agents).
		WithToolVersion("1.0.0").
		WithTimeRange(testStartTime, nil)

	report := builder.Build()

	if len(report.Agents) != 2 {
		t.Fatalf("len(Agents) = %d, want 2", len(report.Agents))
	}
}

func TestReportBuilderWithDataSources(t *testing.T) {
	ds := DataSource{
		Name:        "input.csv",
		Type:        "file",
		Path:        "/data/input.csv",
		Description: "Test input data",
		Hash:        "abc123",
		RecordCount: 1000,
	}

	builder := NewReportBuilder().
		WithDataSource(ds).
		WithToolVersion("1.0.0").
		WithTimeRange(testStartTime, nil)

	report := builder.Build()

	if len(report.DataSources) != 1 {
		t.Fatalf("len(DataSources) = %d, want 1", len(report.DataSources))
	}
	if report.DataSources[0].Name != "input.csv" {
		t.Errorf("DataSources[0].Name = %v, want 'input.csv'", report.DataSources[0].Name)
	}
	if report.DataSources[0].RecordCount != 1000 {
		t.Errorf("DataSources[0].RecordCount = %v, want 1000", report.DataSources[0].RecordCount)
	}
}

func TestReportBuilderWithParameters(t *testing.T) {
	builder := NewReportBuilder().
		WithParameter("alpha", "0.5").
		WithParameter("beta", "0.8").
		WithParameters(map[string]string{
			"gamma": "0.3",
			"delta": "0.9",
		}).
		WithToolVersion("1.0.0").
		WithTimeRange(testStartTime, nil)

	report := builder.Build()

	if len(report.Parameters) != 4 {
		t.Fatalf("len(Parameters) = %d, want 4", len(report.Parameters))
	}
	if report.Parameters["alpha"] != "0.5" {
		t.Errorf("Parameters['alpha'] = %v, want '0.5'", report.Parameters["alpha"])
	}
	if report.Parameters["gamma"] != "0.3" {
		t.Errorf("Parameters['gamma'] = %v, want '0.3'", report.Parameters["gamma"])
	}
}

func TestReportBuilderWithEnvironment(t *testing.T) {
	builder := NewReportBuilder().
		WithEnvironmentVar("CUDA_VERSION", "11.7").
		WithEnvironment(map[string]string{
			"PYTHON_VERSION": "3.10",
			"GPU":            "A100",
		}).
		WithToolVersion("1.0.0").
		WithTimeRange(testStartTime, nil)

	report := builder.Build()

	if len(report.Environment) != 3 {
		t.Fatalf("len(Environment) = %d, want 3", len(report.Environment))
	}
	if report.Environment["CUDA_VERSION"] != "11.7" {
		t.Errorf("Environment['CUDA_VERSION'] = %v, want '11.7'", report.Environment["CUDA_VERSION"])
	}
}

func TestReportBuilderWithExperimentHash(t *testing.T) {
	hash := &ExperimentHash{
		Hash:       "abc123def456",
		Algorithm:  HashAlgorithm,
		ComputedAt: time.Now(),
	}

	builder := NewReportBuilder().
		WithExperimentHash(hash).
		WithToolVersion("1.0.0").
		WithTimeRange(testStartTime, nil)

	report := builder.Build()

	if report.ExperimentHash == nil {
		t.Fatal("ExperimentHash is nil")
	}
	if report.ExperimentHash.Hash != "abc123def456" {
		t.Errorf("ExperimentHash.Hash = %v, want 'abc123def456'", report.ExperimentHash.Hash)
	}
}

func TestReportBuilderAutoComputesHash(t *testing.T) {
	builder := NewReportBuilder().
		WithToolVersion("1.0.0").
		WithMeasurementMode("active").
		WithMeasurementCount(10).
		WithConversationCount(5).
		WithTimeRange(testStartTime, testEndTimePtr()).
		WithParameter("test_param", "value")

	report := builder.Build()

	if report.ExperimentHash == nil {
		t.Fatal("ExperimentHash should be auto-computed")
	}
	if report.ExperimentHash.Hash == "" {
		t.Error("ExperimentHash.Hash should not be empty")
	}
	if report.ExperimentHash.Algorithm != HashAlgorithm {
		t.Errorf("ExperimentHash.Algorithm = %v, want %v", report.ExperimentHash.Algorithm, HashAlgorithm)
	}
}

func TestReproducibilityReportHashDeterminism(t *testing.T) {
	// Create two identical reports
	builder1 := NewReportBuilder().
		WithToolVersion("1.0.0").
		WithMeasurementMode("active").
		WithMeasurementCount(10).
		WithConversationCount(5).
		WithTimeRange(testStartTime, testEndTimePtr()).
		WithParameter("alpha", "0.5").
		WithParameter("beta", "0.8")

	builder2 := NewReportBuilder().
		WithToolVersion("1.0.0").
		WithMeasurementMode("active").
		WithMeasurementCount(10).
		WithConversationCount(5).
		WithTimeRange(testStartTime, testEndTimePtr()).
		WithParameter("alpha", "0.5").
		WithParameter("beta", "0.8")

	report1 := builder1.Build()
	report2 := builder2.Build()

	if report1.ExperimentHash.Hash != report2.ExperimentHash.Hash {
		t.Errorf("Hashes should be identical:\n  Report1: %s\n  Report2: %s",
			report1.ExperimentHash.Hash, report2.ExperimentHash.Hash)
	}
}

func TestReproducibilityReportVerify(t *testing.T) {
	report := NewReportBuilder().
		WithToolVersion("1.0.0").
		WithMeasurementMode("active").
		WithMeasurementCount(10).
		WithConversationCount(5).
		WithTimeRange(testStartTime, testEndTimePtr()).
		Build()

	if !report.Verify() {
		t.Error("Report should verify successfully")
	}

	// Test with nil hash
	reportNoHash := &ReproducibilityReport{}
	if reportNoHash.Verify() {
		t.Error("Report with nil hash should fail verification")
	}
}

func TestReproducibilityReportMarkdownFormat(t *testing.T) {
	report := NewReportBuilder().
		WithTitle("Test Experiment").
		WithAuthor("Test Author").
		WithSessionID("session-123").
		WithSessionName("Test Session").
		WithToolVersion("1.0.0").
		WithMeasurementMode("active").
		WithMeasurementCount(10).
		WithConversationCount(5).
		WithTimeRange(testStartTime, testEndTimePtr()).
		WithAgent(AgentConfig{
			ID:       "agent-1",
			Name:     "Claude",
			Type:     "llm",
			Model:    "claude-3",
			Provider: "anthropic",
		}).
		WithDataSource(DataSource{
			Name: "input.csv",
			Type: "file",
			Path: "/data/input.csv",
		}).
		WithParameter("learning_rate", "0.001").
		Build()

	markdown := report.ToMarkdown()

	// Check required sections are present
	requiredContent := []string{
		"# Test Experiment",
		"**Author:** Test Author",
		"## Session Information",
		"session-123",
		"## Experiment Hash",
		"Algorithm: SHA-256",
		"## Tool Information",
		"1.0.0",
		"## Experiment Timing",
		"## Configuration",
		"Measurement Mode",
		"active",
		"Measurement Count | 10",
		"Conversation Count | 5",
		"## Agent Configurations",
		"Agent 1: Claude",
		"claude-3",
		"anthropic",
		"## Data Sources",
		"input.csv",
		"## Experiment Parameters",
		"learning_rate | 0.001",
		"## Verification",
		"SHA-256 hash",
	}

	for _, content := range requiredContent {
		if !strings.Contains(markdown, content) {
			t.Errorf("Markdown should contain '%s'", content)
		}
	}
}

func TestReproducibilityReportLaTeXFormat(t *testing.T) {
	report := NewReportBuilder().
		WithTitle("Test Experiment").
		WithAuthor("Test Author").
		WithSessionID("session-123").
		WithToolVersion("1.0.0").
		WithMeasurementMode("active").
		WithMeasurementCount(10).
		WithConversationCount(5).
		WithTimeRange(testStartTime, testEndTimePtr()).
		WithParameter("alpha", "0.5").
		Build()

	latex := report.ToLaTeX()

	// Check LaTeX structure
	requiredContent := []string{
		"\\section{Reproducibility Report}",
		"\\textbf{Author:}",
		"\\subsection{Session Information}",
		"\\texttt{session-123}",
		"\\subsection{Experiment Hash}",
		"\\begin{verbatim}",
		"\\end{verbatim}",
		"\\subsection{Configuration}",
		"\\begin{tabular}",
		"\\end{tabular}",
		"Tool Version",
		"1.0.0",
		"Measurement Mode",
		"active",
		"\\subsection{Parameters}",
		"alpha",
		"0.5",
	}

	for _, content := range requiredContent {
		if !strings.Contains(latex, content) {
			t.Errorf("LaTeX should contain '%s'", content)
		}
	}
}

func TestReproducibilityReportJSONFormat(t *testing.T) {
	report := NewReportBuilder().
		WithTitle("Test Experiment").
		WithToolVersion("1.0.0").
		WithMeasurementMode("active").
		WithMeasurementCount(10).
		WithConversationCount(5).
		WithTimeRange(testStartTime, testEndTimePtr()).
		WithParameter("alpha", "0.5").
		Build()

	jsonStr := report.ToJSON()

	// Verify it's valid JSON
	var parsed ReproducibilityReport
	if err := json.Unmarshal([]byte(jsonStr), &parsed); err != nil {
		t.Fatalf("JSON should be valid: %v", err)
	}

	// Verify key fields
	if parsed.Title != "Test Experiment" {
		t.Errorf("Title = %v, want 'Test Experiment'", parsed.Title)
	}
	if parsed.ToolVersion != "1.0.0" {
		t.Errorf("ToolVersion = %v, want '1.0.0'", parsed.ToolVersion)
	}
	if parsed.MeasurementMode != "active" {
		t.Errorf("MeasurementMode = %v, want 'active'", parsed.MeasurementMode)
	}
	if parsed.MeasurementCount != 10 {
		t.Errorf("MeasurementCount = %v, want 10", parsed.MeasurementCount)
	}
	if parsed.Parameters["alpha"] != "0.5" {
		t.Errorf("Parameters['alpha'] = %v, want '0.5'", parsed.Parameters["alpha"])
	}
}

func TestReproducibilityReportFormatWithConfig(t *testing.T) {
	report := NewReportBuilder().
		WithToolVersion("1.0.0").
		WithMeasurementMode("active").
		WithTimeRange(testStartTime, nil).
		Build()

	// Test with each format
	tests := []struct {
		format   ReportFormat
		contains string
	}{
		{FormatMarkdown, "#"},
		{FormatLaTeX, "\\section"},
		{FormatJSON, "{"},
	}

	for _, tt := range tests {
		t.Run(string(tt.format), func(t *testing.T) {
			config := DefaultReproducibilityConfig()
			config.Format = tt.format
			output := report.Format(config)
			if !strings.Contains(output, tt.contains) {
				t.Errorf("Format output should contain '%s'", tt.contains)
			}
		})
	}
}

func TestReproducibilityReportConfigOptions(t *testing.T) {
	report := NewReportBuilder().
		WithToolVersion("1.0.0").
		WithMeasurementMode("active").
		WithTimeRange(testStartTime, nil).
		WithAgent(AgentConfig{ID: "agent-1", Name: "Agent", Type: "llm"}).
		WithDataSource(DataSource{Name: "source", Type: "file"}).
		WithParameter("param", "value").
		WithEnvironmentVar("ENV", "value").
		Build()

	t.Run("exclude agent configs", func(t *testing.T) {
		config := DefaultReproducibilityConfig()
		config.IncludeAgentConfigs = false
		output := report.Format(config)
		if strings.Contains(output, "Agent Configurations") {
			t.Error("Should not include agent configurations when disabled")
		}
	})

	t.Run("exclude data sources", func(t *testing.T) {
		config := DefaultReproducibilityConfig()
		config.IncludeDataSources = false
		output := report.Format(config)
		if strings.Contains(output, "Data Sources") {
			t.Error("Should not include data sources when disabled")
		}
	})

	t.Run("exclude parameters", func(t *testing.T) {
		config := DefaultReproducibilityConfig()
		config.IncludeParameters = false
		output := report.Format(config)
		if strings.Contains(output, "Experiment Parameters") {
			t.Error("Should not include parameters when disabled")
		}
	})

	t.Run("include environment", func(t *testing.T) {
		config := DefaultReproducibilityConfig()
		config.IncludeEnvironment = true
		output := report.Format(config)
		if !strings.Contains(output, "Environment") {
			t.Error("Should include environment when enabled")
		}
	})

	t.Run("exclude verification instructions", func(t *testing.T) {
		config := DefaultReproducibilityConfig()
		config.IncludeVerificationInstructions = false
		output := report.Format(config)
		if strings.Contains(output, "## Verification") {
			t.Error("Should not include verification instructions when disabled")
		}
	})
}

func TestExportReportToWriter(t *testing.T) {
	report := NewReportBuilder().
		WithToolVersion("1.0.0").
		WithMeasurementMode("active").
		WithTimeRange(testStartTime, nil).
		Build()

	var buf bytes.Buffer
	err := ExportReportToWriter(&buf, report, nil)
	if err != nil {
		t.Fatalf("ExportReportToWriter() error = %v", err)
	}

	output := buf.String()
	if !strings.Contains(output, "Reproducibility Report") {
		t.Error("Output should contain report title")
	}
}

func TestExportReportToWriterNilReport(t *testing.T) {
	var buf bytes.Buffer
	err := ExportReportToWriter(&buf, nil, nil)
	if err == nil {
		t.Error("ExportReportToWriter() should error on nil report")
	}
}

func TestExportReportToFile(t *testing.T) {
	report := NewReportBuilder().
		WithTitle("File Export Test").
		WithToolVersion("1.0.0").
		WithMeasurementMode("active").
		WithTimeRange(testStartTime, nil).
		Build()

	// Create temp file
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "reproducibility_report.md")

	err := ExportReportToFile(filePath, report, nil)
	if err != nil {
		t.Fatalf("ExportReportToFile() error = %v", err)
	}

	// Read file and verify content
	content, err := os.ReadFile(filePath)
	if err != nil {
		t.Fatalf("Failed to read file: %v", err)
	}

	if !strings.Contains(string(content), "File Export Test") {
		t.Error("File should contain report title")
	}
}

func TestGenerateReproducibilityReport(t *testing.T) {
	agents := []AgentConfig{
		{ID: "agent-1", Name: "Agent 1", Type: "llm"},
	}
	dataSources := []DataSource{
		{Name: "source", Type: "file"},
	}
	params := map[string]string{
		"param1": "value1",
	}

	report := GenerateReproducibilityReport(
		"1.0.0",
		"session-123",
		"Test Session",
		"active",
		10,
		5,
		testStartTime,
		testEndTimePtr(),
		agents,
		dataSources,
		params,
	)

	if report == nil {
		t.Fatal("GenerateReproducibilityReport() returned nil")
	}
	if report.ToolVersion != "1.0.0" {
		t.Errorf("ToolVersion = %v, want '1.0.0'", report.ToolVersion)
	}
	if report.SessionID != "session-123" {
		t.Errorf("SessionID = %v, want 'session-123'", report.SessionID)
	}
	if len(report.Agents) != 1 {
		t.Errorf("len(Agents) = %d, want 1", len(report.Agents))
	}
	if len(report.DataSources) != 1 {
		t.Errorf("len(DataSources) = %d, want 1", len(report.DataSources))
	}
	if len(report.Parameters) != 1 {
		t.Errorf("len(Parameters) = %d, want 1", len(report.Parameters))
	}
	if report.ExperimentHash == nil {
		t.Error("ExperimentHash should be computed")
	}
}

func TestDurationFormatting(t *testing.T) {
	tests := []struct {
		name     string
		endTime  time.Time
		contains string
	}{
		{
			name:     "short duration (seconds)",
			endTime:  testStartTime.Add(30 * time.Second),
			contains: "30.0s",
		},
		{
			name:     "medium duration (minutes)",
			endTime:  testStartTime.Add(5*time.Minute + 30*time.Second),
			contains: "5m 30s",
		},
		{
			name:     "long duration (hours)",
			endTime:  testStartTime.Add(2*time.Hour + 15*time.Minute),
			contains: "2h 15m",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			endPtr := tt.endTime
			builder := NewReportBuilder().
				WithToolVersion("1.0.0").
				WithTimeRange(testStartTime, &endPtr)

			report := builder.Build()
			if !strings.Contains(report.Duration, tt.contains) {
				t.Errorf("Duration = %v, want to contain '%s'", report.Duration, tt.contains)
			}
		})
	}
}

func TestSortedKeysForDeterministicOutput(t *testing.T) {
	// Create report with parameters in arbitrary order
	report := NewReportBuilder().
		WithToolVersion("1.0.0").
		WithTimeRange(testStartTime, nil).
		WithParameter("zebra", "last").
		WithParameter("alpha", "first").
		WithParameter("middle", "middle").
		Build()

	markdown := report.ToMarkdown()

	// Find positions of parameters in output
	alphaPos := strings.Index(markdown, "alpha")
	middlePos := strings.Index(markdown, "middle")
	zebraPos := strings.Index(markdown, "zebra")

	if alphaPos == -1 || middlePos == -1 || zebraPos == -1 {
		t.Fatal("All parameters should be in output")
	}

	if !(alphaPos < middlePos && middlePos < zebraPos) {
		t.Errorf("Parameters should be sorted alphabetically: alpha=%d, middle=%d, zebra=%d",
			alphaPos, middlePos, zebraPos)
	}
}

func TestReportIncludesAllRequiredSections(t *testing.T) {
	// Create a comprehensive report
	report := NewReportBuilder().
		WithTitle("Comprehensive Test Report").
		WithAuthor("Test Author").
		WithSessionID("session-abc123").
		WithSessionName("Comprehensive Test").
		WithSessionDescription("A comprehensive test of all features").
		WithToolVersion("2.0.0-beta").
		WithMeasurementMode("triggered").
		WithMeasurementCount(100).
		WithConversationCount(25).
		WithTimeRange(testStartTime, testEndTimePtr()).
		WithAgent(AgentConfig{
			ID:       "claude-agent",
			Name:     "Claude",
			Type:     "llm",
			Model:    "claude-3-opus",
			Provider: "anthropic",
			Parameters: map[string]string{
				"temperature": "0.7",
			},
		}).
		WithDataSource(DataSource{
			Name:        "training_data.csv",
			Type:        "file",
			Path:        "/data/training.csv",
			Description: "Training dataset",
			Hash:        "sha256:abc123",
			RecordCount: 10000,
		}).
		WithParameter("learning_rate", "0.001").
		WithParameter("batch_size", "32").
		Build()

	markdown := report.ToMarkdown()

	// List of all sections that should be present
	requiredSections := []string{
		// Title and metadata
		"Comprehensive Test Report",
		"Test Author",
		"Generated:",

		// Session information
		"Session Information",
		"session-abc123",
		"Comprehensive Test",
		"A comprehensive test of all features",

		// Experiment hash
		"Experiment Hash",
		"Algorithm: SHA-256",

		// Tool information
		"Tool Information",
		"2.0.0-beta",

		// Timing
		"Experiment Timing",
		"Start Time",
		"End Time",
		"Duration",

		// Configuration
		"Configuration",
		"Measurement Mode",
		"triggered",
		"Measurement Count | 100",
		"Conversation Count | 25",

		// Agents
		"Agent Configurations",
		"Claude",
		"claude-3-opus",
		"anthropic",
		"temperature | 0.7",

		// Data sources
		"Data Sources",
		"training_data.csv",
		"Training dataset",
		"sha256:abc123",
		"Record Count | 10000",

		// Parameters
		"Experiment Parameters",
		"learning_rate | 0.001",
		"batch_size | 32",

		// Verification
		"Verification",
		"Hash Verification",
		"SHA-256",
		"Recomputation",
	}

	for _, section := range requiredSections {
		if !strings.Contains(markdown, section) {
			t.Errorf("Report should contain '%s'", section)
		}
	}
}

func TestReportHashIncludesVersion(t *testing.T) {
	report1 := NewReportBuilder().
		WithToolVersion("1.0.0").
		WithMeasurementMode("active").
		WithMeasurementCount(10).
		WithConversationCount(5).
		WithTimeRange(testStartTime, testEndTimePtr()).
		Build()

	report2 := NewReportBuilder().
		WithToolVersion("2.0.0"). // Different version
		WithMeasurementMode("active").
		WithMeasurementCount(10).
		WithConversationCount(5).
		WithTimeRange(testStartTime, testEndTimePtr()).
		Build()

	if report1.ExperimentHash.Hash == report2.ExperimentHash.Hash {
		t.Error("Different versions should produce different hashes")
	}
}

func TestReportHashIncludesConfig(t *testing.T) {
	report1 := NewReportBuilder().
		WithToolVersion("1.0.0").
		WithMeasurementMode("active").
		WithMeasurementCount(10).
		WithConversationCount(5).
		WithTimeRange(testStartTime, testEndTimePtr()).
		Build()

	report2 := NewReportBuilder().
		WithToolVersion("1.0.0").
		WithMeasurementMode("passive"). // Different mode
		WithMeasurementCount(10).
		WithConversationCount(5).
		WithTimeRange(testStartTime, testEndTimePtr()).
		Build()

	if report1.ExperimentHash.Hash == report2.ExperimentHash.Hash {
		t.Error("Different measurement modes should produce different hashes")
	}
}

func TestReportHashIncludesAllParameters(t *testing.T) {
	report1 := NewReportBuilder().
		WithToolVersion("1.0.0").
		WithMeasurementMode("active").
		WithMeasurementCount(10).
		WithConversationCount(5).
		WithTimeRange(testStartTime, testEndTimePtr()).
		WithParameter("alpha", "0.5").
		Build()

	report2 := NewReportBuilder().
		WithToolVersion("1.0.0").
		WithMeasurementMode("active").
		WithMeasurementCount(10).
		WithConversationCount(5).
		WithTimeRange(testStartTime, testEndTimePtr()).
		WithParameter("alpha", "0.6"). // Different value
		Build()

	if report1.ExperimentHash.Hash == report2.ExperimentHash.Hash {
		t.Error("Different parameter values should produce different hashes")
	}
}

func TestReportBuilderWithConfig(t *testing.T) {
	config := &ReproducibilityConfig{
		Format:                          FormatLaTeX,
		IncludeAgentConfigs:             false,
		IncludeDataSources:              false,
		IncludeParameters:               true,
		IncludeVerificationInstructions: false,
		Title:                           "Custom Title",
		Author:                          "Custom Author",
	}

	builder := NewReportBuilder().
		WithConfig(config).
		WithToolVersion("1.0.0").
		WithTimeRange(testStartTime, nil)

	// Verify config was applied
	if builder.config.Format != FormatLaTeX {
		t.Errorf("Format = %v, want %v", builder.config.Format, FormatLaTeX)
	}
	if builder.config.IncludeAgentConfigs {
		t.Error("IncludeAgentConfigs should be false")
	}
}

func TestReportBuilderWithFormat(t *testing.T) {
	builder := NewReportBuilder().
		WithFormat(FormatJSON).
		WithToolVersion("1.0.0").
		WithTimeRange(testStartTime, nil)

	if builder.config.Format != FormatJSON {
		t.Errorf("Format = %v, want %v", builder.config.Format, FormatJSON)
	}
}

func TestEmptyReportHandling(t *testing.T) {
	// Minimal report with no data
	report := NewReportBuilder().Build()

	// Should still generate valid output
	markdown := report.ToMarkdown()
	if markdown == "" {
		t.Error("Empty report should still generate output")
	}

	// Should have default title
	if !strings.Contains(markdown, "Experiment Reproducibility Report") {
		t.Error("Should have default title")
	}
}

func TestNilConfigHandling(t *testing.T) {
	report := NewReportBuilder().
		WithToolVersion("1.0.0").
		WithTimeRange(testStartTime, nil).
		Build()

	// Should not panic with nil config
	output := report.Format(nil)
	if output == "" {
		t.Error("Should generate output with nil config")
	}
}
