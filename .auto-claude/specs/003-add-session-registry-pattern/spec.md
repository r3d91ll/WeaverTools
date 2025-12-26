# Add Session Registry Pattern

## Overview

Create a SessionRegistry mirroring the Backend Registry pattern - manages multiple research sessions with thread-safe access, listing, and status retrieval. Enables running multiple concurrent experiments.

## Rationale

The Backend Registry in backend/registry.go provides a clean pattern for managing named resources: Register(), Get(), List(), Status(), with sync.RWMutex for concurrency. Sessions are currently standalone, but research workflows would benefit from a registry to manage multiple concurrent sessions.

---
*This spec was created from ideation and is pending detailed specification.*
