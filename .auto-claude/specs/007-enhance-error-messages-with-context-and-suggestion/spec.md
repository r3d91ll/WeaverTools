# Enhance error messages with context and suggestions

## Overview

Transform generic 'Error: %v' messages into structured, actionable error displays that include: what went wrong, why it happened, and how to fix it. Use consistent color coding and formatting.

## Rationale

Current error handling shows raw error values without context or remediation guidance. Well-designed CLI tools (like rustc, gh, kubectl) provide rich error messages that help users self-diagnose and resolve issues without external documentation.

---
*This spec was created from ideation and is pending detailed specification.*
