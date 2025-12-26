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
