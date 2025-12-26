# Add Real Streaming to Loom Backend

## Overview

Implement true streaming support in the Loom backend ChatStream method using Server-Sent Events (SSE), replacing the current stub that buffers the entire response.

## Rationale

The Loom backend's ChatStream() currently calls Chat() and returns the response as a single chunk. The ClaudeCode backend shows the proper streaming implementation pattern using bufio.Scanner and JSON event parsing. Loom's HTTP client infrastructure supports streaming.

---
*This spec was created from ideation and is pending detailed specification.*
