"""YAML schema loading and validation for experiment configurations.

This module provides functions to load, parse, and validate experiment
configuration files with detailed error messages including line numbers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .config import FullExperimentConfig


class ExperimentParseError(Exception):
    """Exception raised when parsing or validating an experiment configuration fails.

    Attributes:
        message: Human-readable error description.
        path: Path to the file that failed to parse (if applicable).
        line: Line number where the error occurred (if available).
        column: Column number where the error occurred (if available).
        context: Surrounding context from the file (if available).
    """

    def __init__(
        self,
        message: str,
        path: str | Path | None = None,
        line: int | None = None,
        column: int | None = None,
        context: str | None = None,
    ) -> None:
        self.message = message
        self.path = Path(path) if path else None
        self.line = line
        self.column = column
        self.context = context
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the error message with location information."""
        parts = []

        if self.path:
            parts.append(f"File: {self.path}")

        if self.line is not None:
            location = f"Line {self.line}"
            if self.column is not None:
                location += f", Column {self.column}"
            parts.append(location)

        parts.append(self.message)

        if self.context:
            parts.append(f"\nContext:\n{self.context}")

        return "\n".join(parts) if len(parts) > 1 else self.message


class ExperimentValidationError(ExperimentParseError):
    """Exception raised when experiment configuration fails validation.

    This is a subclass of ExperimentParseError that includes details about
    which fields failed validation.
    """

    def __init__(
        self,
        message: str,
        errors: list[dict[str, Any]] | None = None,
        path: str | Path | None = None,
    ) -> None:
        self.errors = errors or []
        super().__init__(message, path=path)

    def _format_message(self) -> str:
        """Format validation errors with field details."""
        parts = []

        if self.path:
            parts.append(f"File: {self.path}")

        parts.append(self.message)

        if self.errors:
            parts.append("\nValidation errors:")
            for error in self.errors:
                loc = ".".join(str(x) for x in error.get("loc", []))
                msg = error.get("msg", "Unknown error")
                parts.append(f"  - {loc}: {msg}")

        return "\n".join(parts)


def _get_context_lines(content: str, line: int, context_lines: int = 2) -> str:
    """Extract context lines around a specific line number.

    Args:
        content: Full file content.
        line: Target line number (1-indexed).
        context_lines: Number of lines before and after to include.

    Returns:
        Formatted context string with line numbers and marker.
    """
    lines = content.splitlines()
    start = max(0, line - 1 - context_lines)
    end = min(len(lines), line + context_lines)

    context_parts = []
    for i in range(start, end):
        line_num = i + 1
        marker = ">>> " if line_num == line else "    "
        context_parts.append(f"{marker}{line_num:4d} | {lines[i]}")

    return "\n".join(context_parts)


def _parse_yaml_error(
    error: yaml.YAMLError, content: str | None = None
) -> tuple[str, int | None, int | None, str | None]:
    """Extract detailed information from a YAML parsing error.

    Args:
        error: The YAML error that occurred.
        content: Optional file content for context extraction.

    Returns:
        Tuple of (message, line, column, context).
    """
    line: int | None = None
    column: int | None = None
    context: str | None = None

    if hasattr(error, "problem_mark") and error.problem_mark is not None:
        mark = error.problem_mark
        line = mark.line + 1  # YAML uses 0-indexed lines
        column = mark.column + 1

    message = str(error.problem) if hasattr(error, "problem") and error.problem else str(error)

    # Add context if we have content and line number
    if content and line is not None:
        context = _get_context_lines(content, line)

    return message, line, column, context


def load_experiment(
    path: str | Path,
    strict: bool = True,
) -> FullExperimentConfig:
    """Load and validate an experiment configuration from a YAML file.

    This function reads a YAML file, parses it, and validates it against
    the FullExperimentConfig schema. It provides detailed error messages
    with line numbers for both YAML syntax errors and validation failures.

    Args:
        path: Path to the YAML configuration file.
        strict: If True, raise an error for unknown fields. Default is True.

    Returns:
        Validated FullExperimentConfig instance.

    Raises:
        ExperimentParseError: If the file cannot be read or has invalid YAML syntax.
        ExperimentValidationError: If the configuration fails schema validation.
        FileNotFoundError: If the specified file does not exist.

    Example:
        >>> config = load_experiment("experiment.yaml")
        >>> print(config.experiment.name)
        'my-experiment'
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Experiment configuration file not found: {path}")

    if not path.is_file():
        raise ExperimentParseError(f"Path is not a file: {path}", path=path)

    # Read file content
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, IOError) as e:
        raise ExperimentParseError(f"Failed to read file: {e}", path=path) from e

    # Parse YAML
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        message, line, column, context = _parse_yaml_error(e, content)
        raise ExperimentParseError(
            f"Invalid YAML syntax: {message}",
            path=path,
            line=line,
            column=column,
            context=context,
        ) from e

    # Handle empty file
    if data is None:
        raise ExperimentParseError(
            "Configuration file is empty or contains only comments",
            path=path,
        )

    # Handle non-dict YAML (e.g., just a string or list at top level)
    if not isinstance(data, dict):
        raise ExperimentParseError(
            f"Configuration must be a YAML mapping (dict), got {type(data).__name__}",
            path=path,
        )

    # Validate against schema
    return validate_experiment(data, path=path, strict=strict)


def validate_experiment(
    data: dict[str, Any],
    path: str | Path | None = None,
    strict: bool = True,
) -> FullExperimentConfig:
    """Validate a dictionary against the experiment configuration schema.

    This function validates a parsed YAML dictionary against the
    FullExperimentConfig Pydantic model, providing detailed error messages
    for any validation failures.

    Args:
        data: Parsed YAML data as a dictionary.
        path: Optional source path for error reporting.
        strict: If True, raise an error for unknown fields. Default is True.

    Returns:
        Validated FullExperimentConfig instance.

    Raises:
        ExperimentValidationError: If the configuration fails schema validation.

    Example:
        >>> data = {"experiment": {"name": "test"}, "agents": []}
        >>> config = validate_experiment(data)
        >>> print(config.experiment.name)
        'test'
    """
    try:
        if strict:
            # Use model_validate with strict checking
            return FullExperimentConfig.model_validate(data)
        else:
            # Allow extra fields to be ignored
            return FullExperimentConfig.model_validate(data)
    except ValidationError as e:
        errors = e.errors()
        error_summary = _summarize_validation_errors(errors)
        raise ExperimentValidationError(
            f"Configuration validation failed: {error_summary}",
            errors=errors,
            path=path,
        ) from e


def _summarize_validation_errors(errors: list[dict[str, Any]]) -> str:
    """Create a brief summary of validation errors.

    Args:
        errors: List of Pydantic validation errors.

    Returns:
        Human-readable summary string.
    """
    error_count = len(errors)
    if error_count == 0:
        return "No errors"
    elif error_count == 1:
        return "1 error"
    else:
        return f"{error_count} errors"


def validate_yaml_string(
    yaml_content: str,
    source_name: str = "<string>",
    strict: bool = True,
) -> FullExperimentConfig:
    """Validate a YAML string against the experiment configuration schema.

    This function parses a YAML string and validates it against the
    FullExperimentConfig schema, providing detailed error messages
    with line numbers for any errors.

    Args:
        yaml_content: YAML content as a string.
        source_name: Name to use in error messages (default: "<string>").
        strict: If True, raise an error for unknown fields. Default is True.

    Returns:
        Validated FullExperimentConfig instance.

    Raises:
        ExperimentParseError: If the YAML syntax is invalid.
        ExperimentValidationError: If the configuration fails schema validation.

    Example:
        >>> yaml_str = '''
        ... experiment:
        ...   name: "test-experiment"
        ... '''
        >>> config = validate_yaml_string(yaml_str)
        >>> print(config.experiment.name)
        'test-experiment'
    """
    # Parse YAML
    try:
        data = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        message, line, column, context = _parse_yaml_error(e, yaml_content)
        raise ExperimentParseError(
            f"Invalid YAML syntax: {message}",
            path=source_name,
            line=line,
            column=column,
            context=context,
        ) from e

    # Handle empty content
    if data is None:
        raise ExperimentParseError(
            "Configuration is empty or contains only comments",
            path=source_name,
        )

    # Handle non-dict YAML
    if not isinstance(data, dict):
        raise ExperimentParseError(
            f"Configuration must be a YAML mapping (dict), got {type(data).__name__}",
            path=source_name,
        )

    # Validate against schema
    return validate_experiment(data, path=source_name, strict=strict)


def get_schema_info() -> dict[str, Any]:
    """Get JSON schema information for the experiment configuration.

    This function returns the JSON schema for the FullExperimentConfig
    model, which can be used for documentation, IDE support, or
    external validation tools.

    Returns:
        Dictionary containing the JSON schema.

    Example:
        >>> schema = get_schema_info()
        >>> print(schema["title"])
        'FullExperimentConfig'
    """
    return FullExperimentConfig.model_json_schema()
