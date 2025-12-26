# Gotchas & Pitfalls

Things to watch out for in this codebase.

## [2025-12-26 04:56]
The `go` command is blocked in this project environment - cannot run go test, go vet, or go build directly

_Context: When verifying Go code (subtask 5.2), the go command is blocked by project configuration. Manual verification is required outside the automated environment. Document verification commands in build-progress.txt for users to run manually._
