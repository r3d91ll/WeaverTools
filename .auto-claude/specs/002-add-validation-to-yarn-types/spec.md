# Add Validation to Yarn Types

## Overview

Extend the Wool validation pattern (Validate() method returning ValidationError) to Yarn types: Message, Conversation, Session, Measurement. This ensures data integrity before storage or transmission.

## Rationale

Wool/agent.go has a mature validation pattern with Agent.Validate() that checks required fields, valid ranges, and consistency rules, returning a ValidationError with Field and Message. The same pattern should apply to Yarn data structures which currently have no validation.

---
*This spec was created from ideation and is pending detailed specification.*
