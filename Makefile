# Makefile for WeaverTools
# Multi-module Go monorepo with Weaver (CLI), Wool (types), and Yarn (utilities)

# ==============================================================================
# Module Directories
# ==============================================================================
WEAVER_DIR := Weaver
WOOL_DIR := Wool
YARN_DIR := Yarn
ALL_GO_DIRS := $(WEAVER_DIR) $(WOOL_DIR) $(YARN_DIR)

# ==============================================================================
# Go Settings
# ==============================================================================
GOCMD := go
GOBUILD := $(GOCMD) build
GOTEST := $(GOCMD) test
GOVET := $(GOCMD) vet
GOFMT := gofmt
GOIMPORTS := goimports
GOMOD := $(GOCMD) mod
GOCLEAN := $(GOCMD) clean

# Linter
GOLANGCI_LINT := golangci-lint

# Changelog Generator
GIT_CHGLOG := git-chglog

# ==============================================================================
# Common Flags
# ==============================================================================
# Build flags
GOFLAGS ?=
LDFLAGS ?= -s -w
BUILD_FLAGS := $(GOFLAGS) -ldflags="$(LDFLAGS)"

# Test flags
TESTFLAGS ?= -v
COVERFLAGS := -coverprofile=coverage.out -covermode=atomic

# Vet flags
VETFLAGS := -all

# Format flags
FMTFLAGS := -s -w

# ==============================================================================
# Output Paths
# ==============================================================================
BINARY_NAME := weaver
BINARY_PATH := $(WEAVER_DIR)/$(BINARY_NAME)
COVERAGE_FILE := coverage.out

# ==============================================================================
# Go Version (from go.mod)
# ==============================================================================
GO_VERSION := 1.23.4

# ==============================================================================
# Platform Detection
# ==============================================================================
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_S),Linux)
    OS := linux
endif
ifeq ($(UNAME_S),Darwin)
    OS := darwin
endif
ifeq ($(UNAME_M),x86_64)
    ARCH := amd64
endif
ifeq ($(UNAME_M),arm64)
    ARCH := arm64
endif

# Default target
.DEFAULT_GOAL := help

# ==============================================================================
# Build Targets
# ==============================================================================

.PHONY: build
build: ## Build all modules (compile check)
	@echo "Building all modules..."
	@for dir in $(ALL_GO_DIRS); do \
		echo "  Building $$dir..."; \
		(cd $$dir && $(GOBUILD) $(BUILD_FLAGS) ./...) || exit 1; \
	done
	@echo "All modules built successfully."

.PHONY: build-weaver
build-weaver: ## Build the weaver binary
	@echo "Building weaver binary..."
	@(cd $(WEAVER_DIR) && $(GOBUILD) $(BUILD_FLAGS) -o $(BINARY_NAME) ./cmd/weaver)
	@echo "Binary created: $(BINARY_PATH)"

# ==============================================================================
# Test Targets
# ==============================================================================

.PHONY: test
test: ## Run tests for all modules
	@echo "Running tests for all modules..."
	@for dir in $(ALL_GO_DIRS); do \
		echo "  Testing $$dir..."; \
		(cd $$dir && $(GOTEST) ./...) || exit 1; \
	done
	@echo "All tests passed."

.PHONY: test-verbose
test-verbose: ## Run tests with verbose output
	@echo "Running tests (verbose) for all modules..."
	@for dir in $(ALL_GO_DIRS); do \
		echo "  Testing $$dir..."; \
		(cd $$dir && $(GOTEST) $(TESTFLAGS) ./...) || exit 1; \
	done
	@echo "All tests passed."

.PHONY: test-coverage
test-coverage: ## Run tests with coverage report
	@echo "Running tests with coverage..."
	@echo "mode: atomic" > $(COVERAGE_FILE)
	@for dir in $(ALL_GO_DIRS); do \
		echo "  Testing $$dir with coverage..."; \
		(cd $$dir && $(GOTEST) $(COVERFLAGS) ./... && \
		if [ -f coverage.out ]; then \
			tail -n +2 coverage.out >> ../$(COVERAGE_FILE); \
			rm coverage.out; \
		fi) || exit 1; \
	done
	@echo "Coverage report generated: $(COVERAGE_FILE)"
	@$(GOCMD) tool cover -func=$(COVERAGE_FILE) | tail -1

# ==============================================================================
# Lint Targets
# ==============================================================================

# Check if golangci-lint is installed
.PHONY: check-lint
check-lint:
	@which $(GOLANGCI_LINT) > /dev/null 2>&1 || { \
		echo "Error: golangci-lint is not installed."; \
		echo "Install it with: go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest"; \
		echo "Or see: https://golangci-lint.run/welcome/install/"; \
		exit 1; \
	}

.PHONY: lint
lint: check-lint ## Run golangci-lint on all modules
	@echo "Running golangci-lint on all modules..."
	@for dir in $(ALL_GO_DIRS); do \
		echo "  Linting $$dir..."; \
		(cd $$dir && $(GOLANGCI_LINT) run ./...) || exit 1; \
	done
	@echo "All modules passed linting."

.PHONY: lint-fix
lint-fix: check-lint ## Run golangci-lint with auto-fix on all modules
	@echo "Running golangci-lint with auto-fix on all modules..."
	@for dir in $(ALL_GO_DIRS); do \
		echo "  Fixing $$dir..."; \
		(cd $$dir && $(GOLANGCI_LINT) run --fix ./...) || exit 1; \
	done
	@echo "Auto-fix complete."

.PHONY: vet
vet: ## Run go vet on all modules
	@echo "Running go vet on all modules..."
	@for dir in $(ALL_GO_DIRS); do \
		echo "  Vetting $$dir..."; \
		(cd $$dir && $(GOVET) $(VETFLAGS) ./...) || exit 1; \
	done
	@echo "All modules passed vet."

.PHONY: fmt
fmt: ## Run gofmt on all modules
	@echo "Running gofmt on all modules..."
	@for dir in $(ALL_GO_DIRS); do \
		echo "  Formatting $$dir..."; \
		(cd $$dir && $(GOFMT) $(FMTFLAGS) .) || exit 1; \
	done
	@echo "Formatting complete."

.PHONY: fmt-check
fmt-check: ## Check if code is formatted (no changes)
	@echo "Checking code formatting..."
	@UNFORMATTED=$$(for dir in $(ALL_GO_DIRS); do \
		(cd $$dir && $(GOFMT) -l .); \
	done); \
	if [ -n "$$UNFORMATTED" ]; then \
		echo "The following files are not formatted:"; \
		echo "$$UNFORMATTED"; \
		echo "Run 'make fmt' to fix."; \
		exit 1; \
	fi
	@echo "All files are properly formatted."

# ==============================================================================
# Utility Targets
# ==============================================================================

.PHONY: clean
clean: ## Remove build artifacts and generated files
	@echo "Cleaning build artifacts..."
	@rm -f $(BINARY_PATH)
	@rm -f $(COVERAGE_FILE)
	@for dir in $(ALL_GO_DIRS); do \
		echo "  Cleaning $$dir..."; \
		(cd $$dir && $(GOCLEAN) ./...) || exit 1; \
	done
	@echo "Clean complete."

.PHONY: deps
deps: ## Download and verify dependencies for all modules
	@echo "Downloading dependencies for all modules..."
	@for dir in $(ALL_GO_DIRS); do \
		echo "  Downloading dependencies for $$dir..."; \
		(cd $$dir && $(GOMOD) download) || exit 1; \
	done
	@echo "Verifying dependencies..."
	@for dir in $(ALL_GO_DIRS); do \
		echo "  Verifying $$dir..."; \
		(cd $$dir && $(GOMOD) verify) || exit 1; \
	done
	@echo "All dependencies downloaded and verified."

.PHONY: deps-tidy
deps-tidy: ## Tidy dependencies for all modules
	@echo "Tidying dependencies for all modules..."
	@for dir in $(ALL_GO_DIRS); do \
		echo "  Tidying $$dir..."; \
		(cd $$dir && $(GOMOD) tidy) || exit 1; \
	done
	@echo "Dependencies tidied."

.PHONY: check
check: fmt-check vet lint test ## Run all quality checks (format, vet, lint, test)
	@echo "All quality checks passed!"

# ==============================================================================
# Changelog Targets
# ==============================================================================

# Check if git-chglog is installed
.PHONY: check-chglog
check-chglog:
	@which $(GIT_CHGLOG) > /dev/null 2>&1 || { \
		echo "Error: git-chglog is not installed."; \
		echo "Install it with: go install github.com/git-chglog/git-chglog/cmd/git-chglog@latest"; \
		echo "Or see: https://github.com/git-chglog/git-chglog"; \
		exit 1; \
	}

.PHONY: install-chglog
install-chglog: ## Install git-chglog changelog generator
	@echo "Installing git-chglog..."
	@$(GOCMD) install github.com/git-chglog/git-chglog/cmd/git-chglog@latest
	@echo "git-chglog installed successfully."

.PHONY: changelog
changelog: check-chglog ## Generate changelog from git history
	@echo "Generating changelog..."
	@$(GIT_CHGLOG) -o CHANGELOG.md
	@echo "Changelog generated: CHANGELOG.md"

.PHONY: changelog-preview
changelog-preview: check-chglog ## Preview changelog without writing to file
	@echo "Changelog preview:"
	@echo "---"
	@$(GIT_CHGLOG)

.PHONY: help
help: ## Show this help message
	@echo "WeaverTools Makefile"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'