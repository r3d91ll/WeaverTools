# Gotchas & Pitfalls

Things to watch out for in this codebase.

## [2025-12-26 04:33]
The `go` command is not available in this sandbox environment - cannot run go test, go vet, or go build directly

_Context: Testing subtask 2.2 - needed to run `go test ./Yarn/...` but command was blocked. Workaround: perform thorough code review and document recommendation to run tests outside sandbox._

## [2025-12-26 04:38]
Race detector verification (go test -race) requires running outside sandbox. Code review can verify patterns but actual race detection needs `go test -race ./Yarn/...` to be executed in an unrestricted environment.

_Context: Subtask 3.2 - Race detector verification. The sandbox blocks go commands, so manual pattern review was performed instead._
