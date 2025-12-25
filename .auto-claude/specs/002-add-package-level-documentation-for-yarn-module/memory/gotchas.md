# Gotchas & Pitfalls

Things to watch out for in this codebase.

## [2025-12-25 18:56]
Go is not available in the sandbox environment - cannot run 'go doc', 'go build', 'go test', etc.

_Context: When verifying Go package documentation (subtask 3.2), discovered that Go toolchain is not installed in the sandbox. Manual verification of doc.go conventions (package comment format, section headers, tab-indented code blocks, package declaration) can substitute for `go doc` verification._
