# Testing Patterns

**Analysis Date:** 2026-03-18

## Test Framework

**Runner:**
- Go `testing` package
- No custom test harness; package-local helpers live in normal `*_test.go` files

**Assertion Library:**
- Primarily `testing` with `t.Fatalf`, `t.Errorf`, `t.Skip`, and `t.Helper`
- `testify` is available in `go.mod` for richer assertions where useful

**Run Commands:**
```bash
go test ./...                              # Full package test run
go test -v ./ort/...                       # Core ORT package with verbose output
go test -v ./embeddings/minilm             # One embedding package
go test -race ./ort -run 'TestAdvancedSessionRunConcurrent'  # Targeted race subset
go test -run '^$' -bench=. -benchmem ./ort/...               # Benchmarks
make test                                  # Repo-level test target
```

## Test File Organization

**Location:**
- Tests are colocated with implementation files in each package
- There is no separate top-level `tests/` directory

**Naming:**
- Unit tests: `*_test.go`
- Integration tests: `*_integration_test.go`
- Benchmarks: `*_benchmark_test.go` or benchmark functions inside standard test files

**Structure:**
```text
ort/
  environment.go
  environment_test.go
  session.go
  session_test.go
  session_benchmark_test.go
embeddings/minilm/
  embedder.go
  embedder_test.go
  embedder_integration_test.go
```

## Test Structure

**Suite Organization:**
```go
func TestResolveRuntimeArtifact(t *testing.T) {
    tests := []struct {
        name string
        // inputs and expectations...
    }{ /* ... */ }

    for _, tc := range tests {
        t.Run(tc.name, func(t *testing.T) {
            // arrange
            // act
            // assert
        })
    }
}
```

**Patterns:**
- Table-driven tests are common for validation and platform matrices
- `t.TempDir()` is the default for filesystem-heavy tests
- Helpers such as `setupORTTestEnvironment` centralize ORT init/cleanup in integration suites
- Concurrency tests use `sync.WaitGroup`, channels, and atomic counters rather than custom frameworks

## Mocking

**Framework:**
- Mostly hand-rolled fakes and standard-library test servers
- `httptest.Server` is used for download/bootstrap scenarios

**Patterns:**
- Mock HTTP sources by spinning up local servers in tests like `ort/bootstrap_test.go`
- Override env vars to steer download URLs, cache dirs, and runtime library paths
- Use temp files and synthetic archives instead of mocking filesystem packages

**What to Mock:**
- External downloads and HTTP status handling
- Local cache directories and archive contents
- Environment-variable-driven configuration

**What NOT to Mock:**
- Core in-memory tensor/session logic when unit tests can run without a real runtime
- Real ORT integration paths once `ONNXRUNTIME_LIB_PATH` is available

## Fixtures and Factories

**Test Data:**
- OpenCLIP ships committed image fixtures in `examples/openclip/assets/`
- Integration tests download or reuse cached model artifacts under a configurable cache root
- Golden regression datasets for SPLADE and OpenCLIP are validated against hosted JSONL assets

**Location:**
- Small helpers and factories stay beside the tests (`embeddings/openclip/test_helpers_test.go`, `embeddings/splade/test_helpers_test.go`)
- Larger fixture assets live under `examples/openclip/assets/` or user cache directories during test runs

## Coverage

**Requirements:**
- No explicit numeric coverage gate is enforced in the repo
- CI still uploads coverage from the main test matrix to Codecov

**Configuration:**
- Coverage is generated with `go test -coverprofile=coverage.out`
- Some race coverage is intentionally partial because purego/unsafe interop conflicts with full `-race` + checkptr usage

**View Coverage:**
```bash
make test-coverage
open coverage.html
```

## Test Types

**Unit Tests:**
- Heavy in `ort/` for validation, shape parsing, memory lifecycle, and bootstrap security logic
- Usually avoid requiring a real ONNX Runtime library

**Integration Tests:**
- Enabled when `ONNXRUNTIME_LIB_PATH` is set or when package-specific bootstrap logic can fetch assets
- Cover end-to-end model execution in `ort/`, `embeddings/minilm`, `embeddings/splade`, and `embeddings/openclip`

**Benchmark Tests:**
- Concentrated in `ort/session_benchmark_test.go` and `embeddings/splade/benchmark_test.go`
- Focus on warm-run throughput and session creation overhead

## Common Patterns

**Async/Concurrency Testing:**
```go
var wg sync.WaitGroup
for i := 0; i < workers; i++ {
    wg.Add(1)
    go func() {
        defer wg.Done()
        // invoke session/bootstrap path
    }()
}
wg.Wait()
```

**Error Testing:**
```go
if err == nil {
    t.Fatalf("expected error, got nil")
}
if !strings.Contains(err.Error(), "checksum") {
    t.Fatalf("unexpected error: %v", err)
}
```

**Snapshot Testing:**
- Not used
- Golden/parity testing is value-based rather than textual snapshot-based

---

*Testing analysis: 2026-03-18*
*Update when test patterns change*
