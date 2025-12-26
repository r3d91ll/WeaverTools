# Gotchas & Pitfalls

Things to watch out for in this codebase.

## [2025-12-26 17:15]
Spinner tests that check for color codes (colorGreen, colorRed) must explicitly set IsTTY: boolPtr(true) in Config, because bytes.Buffer is not a TTY and auto-detection will disable colors

_Context: Weaver/pkg/spinner/spinner_test.go - TestSuccess and TestFail tests were failing because they expected color codes but bytes.Buffer triggered non-TTY mode_
