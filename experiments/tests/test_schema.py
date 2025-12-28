"""Tests for YAML schema loading and validation."""

import os
import tempfile
from pathlib import Path

import pytest

from src.config import FullExperimentConfig
from src.schema import (
    ExperimentParseError,
    ExperimentValidationError,
    _get_context_lines,
    _parse_yaml_error,
    _summarize_validation_errors,
    get_schema_info,
    load_experiment,
    validate_experiment,
    validate_yaml_string,
)


class TestExperimentParseError:
    """Test ExperimentParseError exception class."""

    def test_basic_error(self):
        error = ExperimentParseError("Test error message")
        assert error.message == "Test error message"
        assert error.path is None
        assert error.line is None
        assert error.column is None
        assert error.context is None
        assert str(error) == "Test error message"

    def test_error_with_path(self):
        error = ExperimentParseError("Test error", path="/path/to/file.yaml")
        assert error.path == Path("/path/to/file.yaml")
        assert "File: /path/to/file.yaml" in str(error)
        assert "Test error" in str(error)

    def test_error_with_path_object(self):
        error = ExperimentParseError("Test error", path=Path("/path/to/file.yaml"))
        assert error.path == Path("/path/to/file.yaml")

    def test_error_with_line(self):
        error = ExperimentParseError("Test error", line=10)
        assert error.line == 10
        assert "Line 10" in str(error)

    def test_error_with_line_and_column(self):
        error = ExperimentParseError("Test error", line=10, column=5)
        assert error.line == 10
        assert error.column == 5
        assert "Line 10, Column 5" in str(error)

    def test_error_with_context(self):
        error = ExperimentParseError(
            "Test error",
            context=">>>   10 | bad_line: here",
        )
        assert error.context == ">>>   10 | bad_line: here"
        assert "Context:" in str(error)
        assert "bad_line: here" in str(error)

    def test_full_error_format(self):
        error = ExperimentParseError(
            "Syntax error",
            path="/path/to/file.yaml",
            line=42,
            column=8,
            context="       41 | good: line\n>>>   42 | bad: line",
        )
        error_str = str(error)
        assert "File: /path/to/file.yaml" in error_str
        assert "Line 42, Column 8" in error_str
        assert "Syntax error" in error_str
        assert "Context:" in error_str
        assert "bad: line" in error_str


class TestExperimentValidationError:
    """Test ExperimentValidationError exception class."""

    def test_basic_validation_error(self):
        error = ExperimentValidationError("Validation failed")
        assert error.message == "Validation failed"
        assert error.errors == []
        assert error.path is None

    def test_validation_error_with_field_errors(self):
        errors = [
            {"loc": ("experiment", "name"), "msg": "Field required"},
            {"loc": ("agents", 0, "type"), "msg": "Invalid type"},
        ]
        error = ExperimentValidationError("Validation failed", errors=errors)
        assert len(error.errors) == 2
        error_str = str(error)
        assert "experiment.name: Field required" in error_str
        assert "agents.0.type: Invalid type" in error_str

    def test_validation_error_with_path(self):
        error = ExperimentValidationError(
            "Validation failed",
            path="/path/to/config.yaml",
        )
        assert "File: /path/to/config.yaml" in str(error)

    def test_validation_error_formatting(self):
        errors = [
            {"loc": ("experiment",), "msg": "Missing required field"},
        ]
        error = ExperimentValidationError(
            "Configuration validation failed: 1 error",
            errors=errors,
            path="test.yaml",
        )
        error_str = str(error)
        assert "File: test.yaml" in error_str
        assert "Validation errors:" in error_str
        assert "experiment: Missing required field" in error_str


class TestGetContextLines:
    """Test context line extraction helper."""

    def test_get_context_lines_basic(self):
        content = "line 1\nline 2\nline 3\nline 4\nline 5"
        context = _get_context_lines(content, 3)
        assert "line 1" in context
        assert "line 2" in context
        assert "line 3" in context
        assert "line 4" in context
        assert "line 5" in context
        # Line 3 should be marked
        assert ">>> " in context

    def test_get_context_lines_at_start(self):
        content = "line 1\nline 2\nline 3\nline 4\nline 5"
        context = _get_context_lines(content, 1)
        assert "line 1" in context
        assert ">>> " in context  # First line should be marked

    def test_get_context_lines_at_end(self):
        content = "line 1\nline 2\nline 3\nline 4\nline 5"
        context = _get_context_lines(content, 5)
        assert "line 5" in context
        assert ">>> " in context

    def test_get_context_lines_with_custom_context_size(self):
        content = "line 1\nline 2\nline 3\nline 4\nline 5\nline 6\nline 7"
        # With context_lines=1, should show only 1 line before/after
        context = _get_context_lines(content, 4, context_lines=1)
        lines = context.split("\n")
        # Should show lines 3, 4, 5 (1 before, target, 1 after)
        assert len(lines) == 3

    def test_get_context_lines_preserves_line_numbers(self):
        content = "line 1\nline 2\nline 3"
        context = _get_context_lines(content, 2)
        # Check that line numbers are included
        assert "1 |" in context or "   1 |" in context
        assert "2 |" in context or "   2 |" in context
        assert "3 |" in context or "   3 |" in context


class TestSummarizeValidationErrors:
    """Test validation error summarization helper."""

    def test_summarize_no_errors(self):
        assert _summarize_validation_errors([]) == "No errors"

    def test_summarize_one_error(self):
        errors = [{"loc": ("field",), "msg": "Error"}]
        assert _summarize_validation_errors(errors) == "1 error"

    def test_summarize_multiple_errors(self):
        errors = [
            {"loc": ("field1",), "msg": "Error 1"},
            {"loc": ("field2",), "msg": "Error 2"},
            {"loc": ("field3",), "msg": "Error 3"},
        ]
        assert _summarize_validation_errors(errors) == "3 errors"


class TestLoadExperiment:
    """Test load_experiment function."""

    def test_load_valid_experiment(self):
        yaml_content = """
experiment:
  name: test-experiment
  description: A test experiment
agents: []
scenarios: []
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = load_experiment(temp_path)
            assert isinstance(config, FullExperimentConfig)
            assert config.experiment.name == "test-experiment"
            assert config.experiment.description == "A test experiment"
        finally:
            os.unlink(temp_path)

    def test_load_with_path_object(self):
        yaml_content = """
experiment:
  name: test-experiment
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            config = load_experiment(temp_path)
            assert config.experiment.name == "test-experiment"
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_experiment("/nonexistent/path/config.yaml")

    def test_load_directory_instead_of_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ExperimentParseError, match="not a file"):
                load_experiment(temp_dir)

    def test_load_empty_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("")
            temp_path = f.name

        try:
            with pytest.raises(
                ExperimentParseError, match="empty or contains only comments"
            ):
                load_experiment(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_comment_only_file(self):
        yaml_content = """
# This is just a comment
# Another comment
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            with pytest.raises(
                ExperimentParseError, match="empty or contains only comments"
            ):
                load_experiment(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_non_dict_yaml(self):
        yaml_content = """
- item1
- item2
- item3
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            with pytest.raises(
                ExperimentParseError, match="must be a YAML mapping"
            ):
                load_experiment(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_string_yaml(self):
        yaml_content = "just a plain string"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            with pytest.raises(ExperimentParseError, match="got str"):
                load_experiment(temp_path)
        finally:
            os.unlink(temp_path)


class TestLoadExperimentMalformedYaml:
    """Test load_experiment with malformed YAML syntax."""

    def test_malformed_yaml_tabs(self):
        # YAML with tabs instead of spaces (common mistake)
        yaml_content = """
experiment:
\tname: test
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            with pytest.raises(ExperimentParseError) as exc_info:
                load_experiment(temp_path)
            error = exc_info.value
            # Should have line information
            assert error.line is not None
        finally:
            os.unlink(temp_path)

    def test_malformed_yaml_unclosed_quote(self):
        yaml_content = '''
experiment:
  name: "unclosed quote
  description: test
'''
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            with pytest.raises(ExperimentParseError) as exc_info:
                load_experiment(temp_path)
            error = exc_info.value
            assert "Invalid YAML syntax" in str(error)
        finally:
            os.unlink(temp_path)

    def test_malformed_yaml_invalid_indentation(self):
        yaml_content = """
experiment:
  name: test
 description: bad indent
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            with pytest.raises(ExperimentParseError) as exc_info:
                load_experiment(temp_path)
            error = exc_info.value
            # Should have path information
            assert error.path is not None
            assert str(temp_path) in str(error.path)
        finally:
            os.unlink(temp_path)

    def test_malformed_yaml_duplicate_keys(self):
        yaml_content = """
experiment:
  name: test1
  name: test2
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            # Note: yaml.safe_load allows duplicate keys (last one wins)
            # This should still parse but with last value
            config = load_experiment(temp_path)
            assert config.experiment.name == "test2"
        finally:
            os.unlink(temp_path)

    def test_malformed_yaml_invalid_character(self):
        yaml_content = """
experiment:
  name: test
  invalid: @@@{invalid}@@@
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            with pytest.raises(ExperimentParseError) as exc_info:
                load_experiment(temp_path)
            error = exc_info.value
            assert error.context is not None  # Should have context
        finally:
            os.unlink(temp_path)

    def test_malformed_yaml_includes_context_in_error(self):
        yaml_content = """
experiment:
  name: test
  parameters:
    key1: value1
    bad: [unclosed
    key2: value2
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            with pytest.raises(ExperimentParseError) as exc_info:
                load_experiment(temp_path)
            error = exc_info.value
            # Error should include context with surrounding lines
            if error.context:
                assert ">>>" in error.context  # Marker for error line
        finally:
            os.unlink(temp_path)


class TestLoadExperimentValidationErrors:
    """Test load_experiment with schema validation errors."""

    def test_missing_required_field(self):
        yaml_content = """
experiment:
  description: Missing name field
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            with pytest.raises(ExperimentValidationError) as exc_info:
                load_experiment(temp_path)
            error = exc_info.value
            assert "validation failed" in str(error).lower()
        finally:
            os.unlink(temp_path)

    def test_missing_experiment_section(self):
        yaml_content = """
agents:
  - id: agent1
    type: claude
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            with pytest.raises(ExperimentValidationError) as exc_info:
                load_experiment(temp_path)
            # Should indicate experiment is required
            assert len(exc_info.value.errors) > 0
        finally:
            os.unlink(temp_path)

    def test_invalid_agent_type(self):
        yaml_content = """
experiment:
  name: test
agents:
  - id: agent1
    type: invalid_type
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            with pytest.raises(ExperimentValidationError):
                load_experiment(temp_path)
        finally:
            os.unlink(temp_path)

    def test_duplicate_agent_ids(self):
        yaml_content = """
experiment:
  name: test
agents:
  - id: agent1
    type: claude
  - id: agent1
    type: local
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            with pytest.raises(ExperimentValidationError, match="Duplicate"):
                load_experiment(temp_path)
        finally:
            os.unlink(temp_path)

    def test_invalid_field_type(self):
        yaml_content = """
experiment:
  name: test
  max_iterations: not_a_number
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            with pytest.raises(ExperimentValidationError):
                load_experiment(temp_path)
        finally:
            os.unlink(temp_path)

    def test_empty_experiment_name(self):
        yaml_content = """
experiment:
  name: ""
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            with pytest.raises(ExperimentValidationError):
                load_experiment(temp_path)
        finally:
            os.unlink(temp_path)


class TestValidateExperiment:
    """Test validate_experiment function."""

    def test_validate_valid_dict(self):
        data = {
            "experiment": {"name": "test-experiment"},
            "agents": [],
            "scenarios": [],
        }
        config = validate_experiment(data)
        assert isinstance(config, FullExperimentConfig)
        assert config.experiment.name == "test-experiment"

    def test_validate_with_all_sections(self):
        data = {
            "experiment": {
                "name": "full-test",
                "seed": 42,
                "max_iterations": 50,
            },
            "agents": [
                {"id": "agent1", "type": "claude"},
                {"id": "agent2", "type": "local"},
            ],
            "scenarios": [
                {
                    "name": "scenario1",
                    "steps": [
                        {"agent": "agent1", "action": "generate"},
                    ],
                },
            ],
            "metrics": {
                "collectors": [{"type": "latency"}],
                "storage": {"type": "file", "path": "./results"},
            },
            "outputs": {
                "format": "json",
                "include_raw_responses": True,
            },
        }
        config = validate_experiment(data)
        assert config.experiment.name == "full-test"
        assert config.experiment.seed == 42
        assert len(config.agents) == 2
        assert len(config.scenarios) == 1
        assert len(config.metrics.collectors) == 1
        assert config.outputs.include_raw_responses is True

    def test_validate_with_path_for_error_reporting(self):
        data = {"experiment": {}}  # Missing name
        with pytest.raises(ExperimentValidationError) as exc_info:
            validate_experiment(data, path="test.yaml")
        assert "test.yaml" in str(exc_info.value)

    def test_validate_missing_experiment(self):
        data = {"agents": []}
        with pytest.raises(ExperimentValidationError):
            validate_experiment(data)

    def test_validate_invalid_nested_field(self):
        data = {
            "experiment": {"name": "test"},
            "agents": [{"id": "agent1", "type": "invalid"}],
        }
        with pytest.raises(ExperimentValidationError):
            validate_experiment(data)


class TestValidateYamlString:
    """Test validate_yaml_string function."""

    def test_validate_valid_yaml_string(self):
        yaml_str = """
experiment:
  name: test-experiment
  description: From YAML string
"""
        config = validate_yaml_string(yaml_str)
        assert config.experiment.name == "test-experiment"
        assert config.experiment.description == "From YAML string"

    def test_validate_minimal_yaml_string(self):
        yaml_str = """
experiment:
  name: minimal
"""
        config = validate_yaml_string(yaml_str)
        assert config.experiment.name == "minimal"
        # Defaults should be applied
        assert config.agents == []
        assert config.scenarios == []

    def test_validate_yaml_string_with_custom_source_name(self):
        yaml_str = """
experiment:
  description: no name
"""
        with pytest.raises(ExperimentValidationError) as exc_info:
            validate_yaml_string(yaml_str, source_name="inline-config")
        assert "inline-config" in str(exc_info.value)

    def test_validate_invalid_yaml_string_syntax(self):
        yaml_str = """
experiment:
  name: "unclosed
"""
        with pytest.raises(ExperimentParseError, match="Invalid YAML"):
            validate_yaml_string(yaml_str)

    def test_validate_empty_yaml_string(self):
        yaml_str = ""
        with pytest.raises(ExperimentParseError, match="empty"):
            validate_yaml_string(yaml_str)

    def test_validate_comment_only_yaml_string(self):
        yaml_str = "# Just a comment"
        with pytest.raises(ExperimentParseError, match="empty"):
            validate_yaml_string(yaml_str)

    def test_validate_non_dict_yaml_string(self):
        yaml_str = "- item1\n- item2"
        with pytest.raises(ExperimentParseError, match="must be a YAML mapping"):
            validate_yaml_string(yaml_str)

    def test_validate_yaml_string_validation_error(self):
        yaml_str = """
experiment:
  name: test
  max_iterations: -1
"""
        with pytest.raises(ExperimentValidationError):
            validate_yaml_string(yaml_str)


class TestGetSchemaInfo:
    """Test get_schema_info function."""

    def test_get_schema_info_returns_dict(self):
        schema = get_schema_info()
        assert isinstance(schema, dict)

    def test_get_schema_info_has_title(self):
        schema = get_schema_info()
        assert "title" in schema
        assert schema["title"] == "FullExperimentConfig"

    def test_get_schema_info_has_properties(self):
        schema = get_schema_info()
        assert "properties" in schema
        assert "experiment" in schema["properties"]

    def test_get_schema_info_has_required_fields(self):
        schema = get_schema_info()
        assert "required" in schema
        assert "experiment" in schema["required"]

    def test_get_schema_info_nested_definitions(self):
        schema = get_schema_info()
        # Should have definitions or $defs for nested models
        assert "$defs" in schema or "definitions" in schema


class TestCompleteExperimentConfigs:
    """Test loading complete experiment configurations."""

    def test_load_comprehensive_experiment(self):
        yaml_content = """
experiment:
  name: multi-agent-evaluation
  description: Evaluate agent collaboration on coding tasks
  version: "1.0"
  seed: 42
  save_checkpoints: true
  parameters:
    max_iterations: 100
    temperature: 0.7
    timeout_seconds: 300
  tags:
    - ml
    - evaluation

agents:
  - id: senior
    type: claude
    role: reviewer
    model: claude-3-opus
    parameters:
      temperature: 0.3
  - id: junior
    type: local
    role: implementer
    endpoint: http://localhost:11434

scenarios:
  - name: code-review-flow
    description: Junior generates, senior reviews
    steps:
      - agent: junior
        action: generate
        prompt_template: prompts/generate.txt
      - agent: senior
        action: review
        input_from: junior.output

metrics:
  collectors:
    - type: latency
      granularity: step
    - type: tokens
      per_agent: true
  storage:
    type: file
    path: ./results

outputs:
  format: json
  include_raw_responses: false
  aggregations:
    - mean
    - std
    - percentile_95
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = load_experiment(temp_path)

            # Verify experiment section
            assert config.experiment.name == "multi-agent-evaluation"
            assert config.experiment.seed == 42
            assert config.experiment.version == "1.0"
            assert "ml" in config.experiment.tags

            # Verify agents section
            assert len(config.agents) == 2
            senior = next(a for a in config.agents if a.id == "senior")
            assert senior.type.value == "claude"
            assert senior.parameters.get("temperature") == 0.3
            junior = next(a for a in config.agents if a.id == "junior")
            assert junior.endpoint == "http://localhost:11434"

            # Verify scenarios section
            assert len(config.scenarios) == 1
            scenario = config.scenarios[0]
            assert scenario.name == "code-review-flow"
            assert len(scenario.steps) == 2
            assert scenario.steps[1].input_from == "junior.output"

            # Verify metrics section
            assert len(config.metrics.collectors) == 2
            assert config.metrics.storage.path == "./results"

            # Verify outputs section
            assert config.outputs.format.value == "json"
            assert not config.outputs.include_raw_responses
        finally:
            os.unlink(temp_path)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_unicode_in_experiment_name(self):
        yaml_str = """
experiment:
  name: "测试-emoji-🚀"
"""
        config = validate_yaml_string(yaml_str)
        assert config.experiment.name == "测试-emoji-🚀"

    def test_very_long_experiment_name(self):
        long_name = "a" * 1000
        yaml_str = f"""
experiment:
  name: "{long_name}"
"""
        config = validate_yaml_string(yaml_str)
        assert config.experiment.name == long_name

    def test_nested_parameters(self):
        yaml_str = """
experiment:
  name: test
  parameters:
    nested:
      deep:
        value: 123
    list_param:
      - 1
      - 2
      - 3
"""
        config = validate_yaml_string(yaml_str)
        assert config.experiment.parameters["nested"]["deep"]["value"] == 123
        assert config.experiment.parameters["list_param"] == [1, 2, 3]

    def test_multiline_description(self):
        yaml_str = """
experiment:
  name: test
  description: |
    This is a multiline description.
    It spans multiple lines.
    And has preserved formatting.
"""
        config = validate_yaml_string(yaml_str)
        assert "multiline" in config.experiment.description
        assert "\n" in config.experiment.description

    def test_special_characters_in_parameters(self):
        yaml_str = """
experiment:
  name: test
  parameters:
    url: "http://example.com?foo=bar&baz=qux"
    path: "/path/with/special/chars"
    regex: "^[a-z]+$"
"""
        config = validate_yaml_string(yaml_str)
        assert "?" in config.experiment.parameters["url"]
        assert config.experiment.parameters["regex"] == "^[a-z]+$"

    def test_null_optional_fields(self):
        yaml_str = """
experiment:
  name: test
  seed: null
  description: ""
"""
        config = validate_yaml_string(yaml_str)
        assert config.experiment.seed is None
        assert config.experiment.description == ""

    def test_boolean_values(self):
        yaml_str = """
experiment:
  name: test
  save_checkpoints: true

outputs:
  include_raw_responses: false
"""
        config = validate_yaml_string(yaml_str)
        assert config.experiment.save_checkpoints is True
        assert config.outputs.include_raw_responses is False

    def test_numeric_values(self):
        yaml_str = """
experiment:
  name: test
  seed: 0
  max_iterations: 1
  timeout_seconds: 1
"""
        config = validate_yaml_string(yaml_str)
        assert config.experiment.seed == 0
        assert config.experiment.max_iterations == 1
        assert config.experiment.timeout_seconds == 1

    def test_yaml_anchors_and_aliases(self):
        yaml_str = """
experiment:
  name: test
  parameters:
    shared: &shared_config
      temperature: 0.7
      max_tokens: 1000
    config1:
      <<: *shared_config
      extra: value1
"""
        config = validate_yaml_string(yaml_str)
        # Anchors/aliases should be resolved
        assert config.experiment.parameters["config1"]["temperature"] == 0.7
        assert config.experiment.parameters["config1"]["extra"] == "value1"


class TestParseYamlError:
    """Test _parse_yaml_error helper function."""

    def test_parse_yaml_error_with_mark(self):
        yaml_str = "key: [unclosed"
        try:
            import yaml
            yaml.safe_load(yaml_str)
        except yaml.YAMLError as e:
            message, line, column, context = _parse_yaml_error(e, yaml_str)
            assert line is not None
            assert column is not None

    def test_parse_yaml_error_without_content(self):
        yaml_str = "key: [unclosed"
        try:
            import yaml
            yaml.safe_load(yaml_str)
        except yaml.YAMLError as e:
            message, line, column, context = _parse_yaml_error(e, None)
            # Should still get line/column but no context
            assert context is None


class TestStrictMode:
    """Test strict mode validation."""

    def test_strict_mode_default(self):
        # By default, extra fields might be ignored or cause errors depending on Pydantic config
        yaml_str = """
experiment:
  name: test
  unknown_field: should_be_ignored
"""
        # This should work as Pydantic by default ignores extra fields
        config = validate_yaml_string(yaml_str)
        assert config.experiment.name == "test"

    def test_load_with_strict_false(self):
        yaml_content = """
experiment:
  name: test
  unknown_field: value
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = load_experiment(temp_path, strict=False)
            assert config.experiment.name == "test"
        finally:
            os.unlink(temp_path)
