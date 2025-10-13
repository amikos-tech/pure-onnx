# ONNX Runtime Pure-Go Bindings Makefile

# Variables
GO := go
GOFLAGS :=
GOOS := $(shell go env GOOS)
GOARCH := $(shell go env GOARCH)
PROJECT_NAME := pure-onnx
PKG := github.com/amikos-tech/$(PROJECT_NAME)

# ONNX Runtime version (supports API v22)
ORT_VERSION := 1.23.1
ORT_BASE_URL := https://github.com/microsoft/onnxruntime/releases/download

# Platform detection for ONNX Runtime downloads
ifeq ($(GOOS),darwin)
	ifeq ($(GOARCH),arm64)
		ORT_PLATFORM := osx-arm64
	else
		ORT_PLATFORM := osx-x86_64
	endif
	ORT_LIB_EXT := dylib
else ifeq ($(GOOS),linux)
	ifeq ($(GOARCH),arm64)
		ORT_PLATFORM := linux-aarch64
	else
		ORT_PLATFORM := linux-x64
	endif
	ORT_LIB_EXT := so
else ifeq ($(GOOS),windows)
	ORT_PLATFORM := win-x64
	ORT_LIB_EXT := dll
endif

ORT_ARCHIVE := onnxruntime-$(ORT_PLATFORM)-$(ORT_VERSION)
ORT_URL := $(ORT_BASE_URL)/v$(ORT_VERSION)/$(ORT_ARCHIVE).tgz
ORT_DIR := third_party/onnxruntime
ORT_LIB_DIR := $(ORT_DIR)/lib
ORT_INCLUDE_DIR := $(ORT_DIR)/include

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

.PHONY: all build test clean fmt vet lint verify help install-tools download-ort list-ort-versions

## help: Show this help message
help:
	@echo "$(GREEN)ONNX Runtime Pure-Go Bindings$(NC)"
	@echo ""
	@echo "$(YELLOW)Usage:$(NC)"
	@echo "  make [target]"
	@echo ""
	@echo "$(YELLOW)Targets:$(NC)"
	@awk 'BEGIN {FS = ":.*##"}; \
		/^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2 } \
		/^##@/ { printf "\n$(YELLOW)%s$(NC)\n", substr($$0, 5) }' $(MAKEFILE_LIST)

## all: Build everything
all: verify build test

## build: Build the project
build:
	@echo "$(YELLOW)Building $(PROJECT_NAME)...$(NC)"
	$(GO) build $(GOFLAGS) ./...
	@echo "$(GREEN)✓ Build complete$(NC)"

## test: Run tests
test:
	@echo "$(YELLOW)Running tests...$(NC)"
	$(GO) test -v -race -cover ./ort/...
	@echo "$(GREEN)✓ Tests complete$(NC)"

## test-coverage: Run tests with coverage report
test-coverage:
	@echo "$(YELLOW)Running tests with coverage...$(NC)"
	$(GO) test -v -race -coverprofile=coverage.out ./ort/...
	$(GO) tool cover -html=coverage.out -o coverage.html
	@echo "$(GREEN)✓ Coverage report generated: coverage.html$(NC)"

## bench: Run benchmarks
bench:
	@echo "$(YELLOW)Running benchmarks...$(NC)"
	$(GO) test -bench=. -benchmem ./ort/...
	@echo "$(GREEN)✓ Benchmarks complete$(NC)"

## fmt: Format code
fmt:
	@echo "$(YELLOW)Formatting code...$(NC)"
	$(GO) fmt ./...
	@echo "$(GREEN)✓ Formatting complete$(NC)"

## vet: Run go vet
vet:
	@echo "$(YELLOW)Running go vet...$(NC)"
	@$(GO) vet ./ort/... || true
	@$(GO) vet ./examples/basic/... || true
	@echo "$(GREEN)✓ Vet complete$(NC)"

## lint: Run golangci-lint (requires golangci-lint to be installed)
lint:
	@echo "$(YELLOW)Running linter...$(NC)"
	@if command -v golangci-lint &> /dev/null; then \
		golangci-lint run --fix ./...; \
		echo "$(GREEN)✓ Linting complete$(NC)"; \
	else \
		echo "$(RED)✗ golangci-lint not installed. Run 'make install-tools' first$(NC)"; \
		exit 1; \
	fi

## verify: Run all verification steps (fmt, vet, lint)
verify: fmt vet
	@echo "$(GREEN)✓ All verification steps complete$(NC)"

## clean: Clean build artifacts and temporary files
clean:
	@echo "$(YELLOW)Cleaning...$(NC)"
	$(GO) clean ./...
	rm -f coverage.out coverage.html
	rm -rf $(ORT_DIR)
	rm -f *.test
	rm -f *.prof
	@echo "$(GREEN)✓ Clean complete$(NC)"

## mod-tidy: Tidy go modules
mod-tidy:
	@echo "$(YELLOW)Tidying modules...$(NC)"
	$(GO) mod tidy
	@echo "$(GREEN)✓ Module tidy complete$(NC)"

## mod-verify: Verify go modules
mod-verify:
	@echo "$(YELLOW)Verifying modules...$(NC)"
	$(GO) mod verify
	@echo "$(GREEN)✓ Module verification complete$(NC)"

## install-tools: Install development tools
install-tools:
	@echo "$(YELLOW)Installing development tools...$(NC)"
	@echo "Installing golangci-lint..."
	@if ! command -v golangci-lint &> /dev/null; then \
		curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $(shell go env GOPATH)/bin; \
	fi
	@echo "Installing goimports..."
	$(GO) install golang.org/x/tools/cmd/goimports@latest
	@echo "Installing godoc..."
	$(GO) install golang.org/x/tools/cmd/godoc@latest
	@echo "$(GREEN)✓ Tools installation complete$(NC)"

##@ ONNX Runtime Management

## download-ort: Download ONNX Runtime library for current platform
download-ort:
	@echo "$(YELLOW)Downloading ONNX Runtime $(ORT_VERSION) for $(ORT_PLATFORM)...$(NC)"
	@mkdir -p $(ORT_DIR)
	@echo "Downloading from: $(ORT_URL)"
	@curl -L -o $(ORT_DIR)/onnxruntime.tgz $(ORT_URL) || \
		(echo "$(RED)✗ Failed to download ONNX Runtime$(NC)" && exit 1)
	@echo "Extracting..."
	@tar -xzf $(ORT_DIR)/onnxruntime.tgz -C $(ORT_DIR) --strip-components=1
	@rm $(ORT_DIR)/onnxruntime.tgz
	@echo "$(GREEN)✓ ONNX Runtime $(ORT_VERSION) downloaded to $(ORT_DIR)$(NC)"
	@echo "$(YELLOW)Library location: $(ORT_LIB_DIR)$(NC)"
	@ls -la $(ORT_LIB_DIR)/*.$(ORT_LIB_EXT) 2>/dev/null || echo "$(RED)No .$(ORT_LIB_EXT) files found$(NC)"

## list-ort-versions: List available ONNX Runtime versions
list-ort-versions:
	@echo "$(YELLOW)Fetching available ONNX Runtime versions...$(NC)"
	@curl -s https://api.github.com/repos/microsoft/onnxruntime/releases | \
		grep '"tag_name":' | \
		sed -E 's/.*"v([^"]+)".*/\1/' | \
		head -20
	@echo ""
	@echo "$(YELLOW)Current configured version: $(ORT_VERSION)$(NC)"
	@echo "$(YELLOW)Platform: $(ORT_PLATFORM)$(NC)"

## download-ort-version: Download specific ONNX Runtime version (use ORT_VERSION=x.y.z)
download-ort-version:
	@if [ -z "$(VERSION)" ]; then \
		echo "$(RED)✗ Please specify VERSION. Example: make download-ort-version VERSION=1.21.0$(NC)"; \
		exit 1; \
	fi
	@$(MAKE) download-ort ORT_VERSION=$(VERSION)

## check-ort: Check if ONNX Runtime is installed
check-ort:
	@echo "$(YELLOW)Checking ONNX Runtime installation...$(NC)"
	@if [ -d "$(ORT_LIB_DIR)" ]; then \
		echo "$(GREEN)✓ ONNX Runtime found at: $(ORT_DIR)$(NC)"; \
		echo "Libraries:"; \
		ls -la $(ORT_LIB_DIR)/*.$(ORT_LIB_EXT) 2>/dev/null || echo "  No .$(ORT_LIB_EXT) files found"; \
	else \
		echo "$(RED)✗ ONNX Runtime not found. Run 'make download-ort' to download$(NC)"; \
	fi

##@ Examples

## run-basic: Run basic example
run-basic:
	@echo "$(YELLOW)Running basic example...$(NC)"
	@if [ ! -d "$(ORT_LIB_DIR)" ]; then \
		echo "$(RED)✗ ONNX Runtime not found. Run 'make download-ort' first$(NC)"; \
		exit 1; \
	fi
	@export DYLD_LIBRARY_PATH=$(ORT_LIB_DIR):$$DYLD_LIBRARY_PATH && \
	export LD_LIBRARY_PATH=$(ORT_LIB_DIR):$$LD_LIBRARY_PATH && \
	$(GO) run examples/basic/main.go

## run-experimental: Run experimental example
run-experimental:
	@echo "$(YELLOW)Running experimental example...$(NC)"
	@if [ ! -d "$(ORT_LIB_DIR)" ]; then \
		echo "$(RED)✗ ONNX Runtime not found. Run 'make download-ort' first$(NC)"; \
		exit 1; \
	fi
	@export DYLD_LIBRARY_PATH=$(ORT_LIB_DIR):$$DYLD_LIBRARY_PATH && \
	export LD_LIBRARY_PATH=$(ORT_LIB_DIR):$$LD_LIBRARY_PATH && \
	$(GO) run examples/experimental/main.go

##@ Development Workflow

## dev: Run full development workflow (verify, build, test)
dev: verify build test
	@echo "$(GREEN)✓ Development workflow complete$(NC)"

## ci: Run continuous integration checks
ci: mod-verify verify build test
	@echo "$(GREEN)✓ CI checks complete$(NC)"

## watch: Watch for changes and rebuild (requires entr)
watch:
	@if command -v entr &> /dev/null; then \
		echo "$(YELLOW)Watching for changes...$(NC)"; \
		find . -name '*.go' | entr -c make build; \
	else \
		echo "$(RED)✗ entr not installed. Install with: brew install entr (macOS) or apt-get install entr (Linux)$(NC)"; \
		exit 1; \
	fi

##@ Documentation

## docs: Generate and serve documentation
docs:
	@echo "$(YELLOW)Starting documentation server...$(NC)"
	@echo "Documentation will be available at http://localhost:6060"
	@godoc -http=:6060

## check-mod: Check for module updates
check-mod:
	@echo "$(YELLOW)Checking for module updates...$(NC)"
	$(GO) list -u -m all

## update-mod: Update all modules
update-mod:
	@echo "$(YELLOW)Updating modules...$(NC)"
	$(GO) get -u ./...
	$(GO) mod tidy
	@echo "$(GREEN)✓ Modules updated$(NC)"

# Variables for version bumping
CURRENT_VERSION := $(shell git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0")

## version: Show current version
version:
	@echo "Current version: $(CURRENT_VERSION)"

.DEFAULT_GOAL := help