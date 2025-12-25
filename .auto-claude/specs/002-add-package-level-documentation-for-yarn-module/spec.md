# Add package-level documentation for Yarn module

## Overview

The Yarn package (conversation, message, measurement, session) has individual function comments but lacks package-level documentation explaining how these types work together. New developers must read all 4 source files to understand the data model for research sessions and conveyance measurements.

## Rationale

Yarn is a core data model package used throughout the ecosystem. Without package documentation, developers struggle to understand: (1) the relationship between Session, Conversation, and Message, (2) how Measurements integrate with hidden state analysis, (3) the purpose of each type in the 'yarn as thread' metaphor. The existing godoc comments describe individual functions but not the overall architecture.

---
*This spec was created from ideation and is pending detailed specification.*
