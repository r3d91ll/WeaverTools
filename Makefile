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
GOMOD := $(GOCMD) mod
GOCLEAN := $(GOCMD) clean

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
