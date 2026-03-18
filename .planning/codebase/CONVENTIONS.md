# Coding Conventions

**Analysis Date:** 2026-03-18

## Naming Patterns

**Files:**
- lower_snake_case for implementation files: `environment.go`, `bootstrap_lock_unix.go`, `golden_dataset_parity_test.go`
- `*_test.go` for all test files, often with specialized suffixes like `*_integration_test.go` or `*_benchmark_test.go`
- generated code is named explicitly with `_generated`: `ort/ortapi_generated.go`

**Functions:**
- Exported APIs use Go-standard PascalCase: `InitializeEnvironment`, `EnsureDefaultAssets`, `WithSequenceLength`
- Unexported helpers use lowerCamelCase: `resolveRuntimeArtifact`, `validateFilePath`, `setupORTTestEnvironment`
- Functional option constructors consistently start with `With` / `Without`

**Variables:**
- lowerCamelCase for locals and fields
- package-level constants are mostly PascalCase when exported and lowerCamelCase when internal
- all-caps names are reserved for compatibility constants or env-oriented identifiers such as `ORT_API_VERSION`

**Types:**
- PascalCase for structs, interfaces, and enums: `AdvancedSession`, `MemoryInfo`, `PoolingStrategy`
- Package names stay short and lowercase: `ort`, `minilm`, `splade`, `openclip`

## Code Style

**Formatting:**
- `gofmt` and `goimports` are the formatting source of truth (`.golangci.yml`, `Makefile`)
- Standard Go formatting conventions apply: tabs, grouped imports, no manual alignment
- Error strings are generally lowercase and sentence-fragment style

**Linting:**
- `go vet`, `golangci-lint`, and `gosec` are the main static checks
- `precommit` in `Makefile` mirrors CI blockers, with opt-out env vars for local workflows
- `.golangci.yml` keeps the rule set small and targeted instead of enabling everything

## Import Organization

**Order:**
1. Go standard library imports
2. Repository-local imports such as `github.com/amikos-tech/pure-onnx/ort`
3. Third-party imports such as `github.com/ebitengine/purego`

**Grouping:**
- One blank line between import groups
- No path aliases beyond short clarity-driven aliases like `tokenizers`
- Side-effect imports are only used when required, for example image codecs in `examples/openclip/main.go`

**Path Aliases:**
- None; imports use full module paths

## Error Handling

**Patterns:**
- Validate aggressively at function entry and return early on bad inputs
- Wrap underlying failures with `fmt.Errorf(...: %w)`
- Join cleanup failures with `errors.Join` where multiple destroy steps can fail
- Native-handle wrappers usually treat nil receivers as safe no-ops on `Destroy()`

**Error Types:**
- Most code returns plain `error` rather than custom error structs
- A few package-private sentinel or wrapper types exist where retry/permanence matters, such as `permanentBootstrapError` in `ort/bootstrap.go`
- Tests assert on message content frequently, so wording changes can be user-visible

## Logging

**Framework:**
- Standard library `log` package only

**Patterns:**
- Library code logs sparingly, mainly for warnings or bootstrap path visibility
- Examples use `log.Fatal` / `log.Printf` for CLI-style behavior
- There is no structured logging abstraction

## Comments

**When to Comment:**
- Explain FFI safety assumptions, lock ordering, and lifetime rules
- Document why `unsafe` usage is acceptable in specific places
- Keep obvious code uncommented

**Doc Comments:**
- Exported types, constants, and functions are usually documented
- Internal helpers receive comments when concurrency, lifecycle, or security behavior is non-obvious

**Security/Lint Markers:**
- `#nosec` annotations are used narrowly to justify intentional `unsafe` pointer conversions or checksum-like constants

## Function Design

**Size:**
- Public APIs tend to be moderate-sized with early validation and deferred cleanup
- The longest functions are concentrated in bootstrap/download code and embedder setup paths

**Parameters:**
- Constructors prefer explicit required parameters plus functional options
- Helpers avoid large configuration structs at call sites unless state needs to persist

**Return Values:**
- APIs nearly always return `(value, error)` or `error`
- Cleanup methods return `error` rather than panicking

## Module Design

**Exports:**
- Packages expose focused public APIs and keep helpers unexported
- `internal/` is used only where cross-package reuse should remain private

**Generated Code:**
- `ort/ortapi_generated.go` is treated as generated output; changes should come from `tools/gen_ortapi.go` and header inputs, not hand edits

---

*Convention analysis: 2026-03-18*
*Update when patterns change*
