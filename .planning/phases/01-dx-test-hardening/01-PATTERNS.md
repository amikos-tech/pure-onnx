# Phase 1: DX & Test Hardening - Pattern Map

**Mapped:** 2026-07-21
**Files analyzed:** 9
**Analogs found:** 9 / 9

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|--------------------|------|-----------|-----------------|----------------|
| `ort/bootstrap.go` (add `ErrUnsupportedPlatform` sentinel + wrap at `resolveRuntimeArtifact`) | utility (error definitions) | request-response (error propagation) | `ort/bootstrap.go:49-50,463-465` (`errBootstrapRedirectPolicy` / `isBootstrapRedirectPolicyError`) | exact — same file, same idiom |
| `ort/bootstrap_test.go` (extend `TestResolveRuntimeArtifact`'s `unsupported` case) | test | request-response | `ort/bootstrap_test.go:26-120` (`TestResolveRuntimeArtifact`) | exact — modifying the exact existing table test |
| `examples/inference/main.go` (`initializeOrtEnvironment` / `main` error branch + extracted `diagnosticFor` helper) | controller (CLI entry point) | request-response | same file, `initializeOrtEnvironment` (lines 157-177) and `main`'s error-handling style (lines 41-89) | exact — same file, same function |
| `examples/inference/main_test.go` (NEW) | test | request-response | No existing test file in `examples/inference/`; closest analog is `ort/bootstrap_test.go`'s table-test style for error-path unit tests | role-match |
| `ort/session_test.go` — `TestAdvancedSessionRunAndDestroyConcurrent` (line 679) | test | event-driven (goroutine rendezvous) | `ort/session_test.go:679-751` (itself — modify in place) + reuse pattern from `TestAdvancedSessionRunConcurrentAcrossSessionsSharingTensor` (line 599, channel-based rendezvous) | exact — self + established rendezvous idiom |
| `ort/session_test.go` — `TestTensorDestroyWaitsForInFlightRun` (line 832) | test | event-driven | `ort/session_test.go:832-896` (itself) + `TestAdvancedSessionRunAndDestroyConcurrent` (structurally identical shape) | exact |
| `ort/session_test.go` — `TestAdvancedSessionDestroyDoesNotBlockUnrelatedRun` (line 753) | test | event-driven | `ort/session_test.go:753-830` (itself) + `TestTensorDestroyDoesNotBlockUnrelatedRun` (line 898, the already-deterministic sibling test using `require.Eventually` differently — not in scope but shows the "doesn't block" shape) | exact |
| `ort/environment_stress_test.go` (NEW) | test | event-driven / batch | `ort/environment_test.go:243-308` (`TestConcurrentInitialization`, `TestConcurrentDestroy`) | exact — explicitly named as the foundation in CONTEXT.md D-07 |
| `.github/workflows/ci.yml` (new dedicated stress-test job) | config (CI workflow) | batch | `.github/workflows/ci.yml:108-126` (`test-race-ort-concurrency` job) | exact — sibling job, same structure |
| `TESTING.md` (new "Running stress tests" section) | config (docs) | N/A | `TESTING.md:247-301` (`## Continuous Integration` / `### Local Pre-commit Checks` sections) | role-match |
| `Makefile` (`test`, `precommit` targets — add `-short`) — **flag for planner confirmation, see Open Question below** | config | N/A | `Makefile:104-114` (`test`, `test-race` targets), `Makefile:347-368` (`precommit`) | exact |

## Pattern Assignments

### `ort/bootstrap.go` (utility, request-response) — add `ErrUnsupportedPlatform`

**Analog:** same file, `errBootstrapRedirectPolicy` / `isBootstrapRedirectPolicyError` sentinel pattern

**Sentinel + detector pattern** (`ort/bootstrap.go:49-50, 463-465`):
```go
var errSharedLibraryNotFound = errors.New("ONNX Runtime shared library not found")
var errBootstrapRedirectPolicy = errors.New("bootstrap redirect policy rejection")
...
func isBootstrapRedirectPolicyError(err error) bool {
	return errors.Is(err, errBootstrapRedirectPolicy)
}
```

**Wrap-at-origin pattern** (`ort/bootstrap.go:445-461`, `rejectHTTPSDowngradeRedirect`):
```go
func rejectHTTPSDowngradeRedirect(req *http.Request, via []*http.Request) error {
	if len(via) >= 10 {
		return fmt.Errorf("%w: stopped after 10 redirects", errBootstrapRedirectPolicy)
	}
	...
}
```

**Current unsupported-platform error to replace** (`ort/bootstrap.go:467-523`, `resolveRuntimeArtifact`):
```go
func resolveRuntimeArtifact(goos, goarch string) (runtimeArtifact, error) {
	switch goos {
	case "darwin": ... case "linux": ... case "windows": ...
	}
	return runtimeArtifact{}, fmt.Errorf("unsupported platform for ONNX Runtime bootstrap: GOOS=%s GOARCH=%s", goos, goarch)
}
```

**Apply this way (per D-02, D-08 discretion note):** add a new, separate, **exported** sentinel next to the existing unexported ones (do not touch/reuse `errBootstrapRedirectPolicy`):
```go
var ErrUnsupportedPlatform = errors.New("unsupported platform for ONNX Runtime bootstrap")

func IsUnsupportedPlatformError(err error) bool {
	return errors.Is(err, ErrUnsupportedPlatform)
}
```
And change the return in `resolveRuntimeArtifact` to wrap it with `%w` (keep GOOS/GOARCH in the message text, same as today, so it stays a drop-in replacement for anything checking the message):
```go
return runtimeArtifact{}, fmt.Errorf("%w: GOOS=%s GOARCH=%s", ErrUnsupportedPlatform, goos, goarch)
```

**Note:** `errSharedLibraryNotFound` and `errBootstrapRedirectPolicy` are unexported (package-private, checked only inside `ort`); `ErrUnsupportedPlatform` must be exported (capitalized) because `examples/inference` — a different package — needs `errors.Is(err, ort.ErrUnsupportedPlatform)`. This is the one place this phase's pattern deviates from the letter of the existing idiom, and it's an explicit, intentional deviation called out in CONTEXT.md D-02/RESEARCH.md.

---

### `ort/bootstrap_test.go` (test, request-response) — extend `TestResolveRuntimeArtifact`

**Analog:** `ort/bootstrap_test.go:26-120` — the existing table test itself

**Existing table + assertion shape** (`ort/bootstrap_test.go:26-33, 100-120`):
```go
tests := []struct {
	name    string
	goos    string
	goarch  string
	want    runtimeArtifact
	wantErr bool
}{
	...
	{
		name:    "unsupported",
		goos:    "linux",
		goarch:  "386",
		wantErr: true,
	},
}

for _, tc := range tests {
	t.Run(tc.name, func(t *testing.T) {
		got, err := resolveRuntimeArtifact(tc.goos, tc.goarch)
		if tc.wantErr {
			if err == nil {
				t.Fatalf("expected error, got nil")
			}
			return
		}
		...
	})
}
```

**Apply this way:** in the `tc.wantErr` branch, add an additional assertion (only for the `"unsupported"` case, or for all `wantErr` cases since all `wantErr=true` cases in this table are the unsupported-platform path today):
```go
if tc.wantErr {
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
	if !errors.Is(err, ErrUnsupportedPlatform) {
		t.Fatalf("expected ErrUnsupportedPlatform, got: %v", err)
	}
	return
}
```
`errors` is already imported in this file (line 10).

---

### `examples/inference/main.go` (controller, request-response) — DX-01 hint

**Analog:** same file — `initializeOrtEnvironment` (lines 157-177) and `main`'s `log.Fatalf` error-handling convention (lines 41-89)

**Current error site to change** (`examples/inference/main.go:41-43`):
```go
if err := initializeOrtEnvironment(); err != nil {
	log.Fatalf("failed to initialize ONNX Runtime: %v", err)
}
```

**Existing imports** (`examples/inference/main.go:1-11`):
```go
package main

import (
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"

	"github.com/amikos-tech/pure-onnx/ort"
)
```
Add `"runtime"` for `runtime.GOOS`/`runtime.GOARCH` (per RESEARCH.md Standard Stack — `runtime` is stdlib, already used the same way in `ort/bootstrap.go`).

**Apply this way (per D-01, extracting a testable helper so `examples/inference/main_test.go` doesn't need to exec the binary):**
```go
func diagnosticFor(err error, goos, goarch string) string {
	if ort.IsUnsupportedPlatformError(err) {
		return fmt.Sprintf(
			"failed to initialize ONNX Runtime: %v\n"+
				"GOOS=%s GOARCH=%s is not supported by automatic bootstrap; "+
				"set ONNXRUNTIME_LIB_PATH to a prebuilt ONNX Runtime shared library for this platform.",
			err, goos, goarch,
		)
	}
	return fmt.Sprintf("failed to initialize ONNX Runtime: %v", err)
}

// in main():
if err := initializeOrtEnvironment(); err != nil {
	log.Fatal(diagnosticFor(err, runtime.GOOS, runtime.GOARCH))
}
```

**Error handling pattern to keep consistent with rest of file** (every other error site in this file uses `log.Fatalf("...: %v", err)` — see lines 24, 28, 33, 38, 52, 62, 79, 88): the new branch should still ultimately funnel through a single `log.Fatal`/`log.Fatalf` call so the file's error-handling style stays uniform. D-01 explicitly requires *other* bootstrap failures to keep their existing, unchanged message — `diagnosticFor`'s fallback branch reproduces the exact current string format for that reason.

---

### `examples/inference/main_test.go` (NEW — test, request-response)

**Analog:** `ort/bootstrap_test.go`'s table-test style for pure error-path assertions (no existing test file in `examples/inference/` to copy structurally — this is a new package-level test file)

**Pattern to follow (table-driven, stdlib `testing`, no testify needed since this is a simple string-content check):**
```go
package main

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/amikos-tech/pure-onnx/ort"
)

func TestDiagnosticForUnsupportedPlatform(t *testing.T) {
	err := fmt.Errorf("%w: GOOS=%s GOARCH=%s", ort.ErrUnsupportedPlatform, "plan9", "386")
	got := diagnosticFor(err, "plan9", "386")
	if !strings.Contains(got, "GOOS=plan9") || !strings.Contains(got, "GOARCH=386") {
		t.Fatalf("expected GOOS/GOARCH in diagnostic, got: %q", got)
	}
	if !strings.Contains(got, "ONNXRUNTIME_LIB_PATH") {
		t.Fatalf("expected ONNXRUNTIME_LIB_PATH hint, got: %q", got)
	}
}

func TestDiagnosticForOtherBootstrapFailureUnchanged(t *testing.T) {
	err := errors.New("checksum mismatch")
	got := diagnosticFor(err, "linux", "amd64")
	if strings.Contains(got, "ONNXRUNTIME_LIB_PATH") {
		t.Fatalf("expected no misleading hint for non-platform failures, got: %q", got)
	}
}
```
Note: this package is `main` (not `ort`), so it must import `ort` as an external package (`github.com/amikos-tech/pure-onnx/ort`) exactly as `examples/inference/main.go` already does — confirms why `ErrUnsupportedPlatform`/`IsUnsupportedPlatformError` must be exported.

---

### `ort/session_test.go` — 3 named tests (test, event-driven)

**Analog:** each test modifies itself in place; the channel-rendezvous *style* to extend is borrowed from `TestAdvancedSessionRunConcurrentAcrossSessionsSharingTensor` (lines 599-677), which already uses buffered channels + `close()` handshakes (just not yet event-order recording).

**Current structure common to all 3 tests** (e.g. `TestAdvancedSessionRunAndDestroyConcurrent`, `ort/session_test.go:679-751`):
```go
runStarted := make(chan struct{})
allowRunReturn := make(chan struct{})
var closeRunStarted sync.Once
...
runSessionFunc = func(...) uintptr {
	closeRunStarted.Do(func() { close(runStarted) })
	<-allowRunReturn
	return 0
}
releaseSessionFunc = func(handle uintptr) {
	atomic.AddInt32(&releasedCount, 1)
	releasedHandle.Store(handle)
}
...
go func() { runErrCh <- session.Run() }()
<-runStarted
go func() { destroyErrCh <- session.Destroy() }()

require.Never(t, func() bool {
	select {
	case <-destroyErrCh:
		return true
	default:
		return false
	}
}, 500*time.Millisecond, 50*time.Millisecond, "destroy returned before run completed")

close(allowRunReturn)
if err := <-runErrCh; err != nil { t.Fatalf("run failed: %v", err) }
if err := <-destroyErrCh; err != nil { t.Fatalf("destroy failed: %v", err) }
```

**Replace the `require.Never(...)` block with (per D-04, RESEARCH.md Pattern 2 — event-order recording + single-shot watchdog on the main test goroutine only):**
```go
var eventsMu sync.Mutex
var events []string
record := func(e string) {
	eventsMu.Lock()
	events = append(events, e)
	eventsMu.Unlock()
}
// inside runSessionFunc, after <-allowRunReturn:
//   record("run-returned")
// inside releaseSessionFunc:
//   record("destroy-released")

select {
case err := <-destroyErrCh:
	t.Fatalf("destroy returned before run completed (err=%v) -- deadlock-safety-net fired unexpectedly early", err)
case <-time.After(500 * time.Millisecond):
	// expected: destroy is still blocked; continue.
}

close(allowRunReturn)
if err := <-runErrCh; err != nil { t.Fatalf("run failed: %v", err) }
if err := <-destroyErrCh; err != nil { t.Fatalf("destroy failed: %v", err) }

eventsMu.Lock()
got := append([]string(nil), events...)
eventsMu.Unlock()
want := []string{"run-returned", "destroy-released"}
if !slices.Equal(got, want) {
	t.Fatalf("expected event order %v, got %v", want, got)
}
```
`slices` (stdlib, Go 1.21+) will need to be added to this file's import block if not already present — check current imports before adding.

**Apply the same transform to `TestTensorDestroyWaitsForInFlightRun`** (`ort/session_test.go:832-896`, same shape, `releaseValueFunc` instead of `releaseSessionFunc`) **and `TestAdvancedSessionDestroyDoesNotBlockUnrelatedRun`** (`ort/session_test.go:753-830`, which already uses `require.Eventually` rather than `require.Never` — its transform is the mirror image: assert the "unrelated destroy is NOT blocked" case via event ordering / a done-channel receive without a blocking watchdog needed, since this test proves the *absence* of blocking, not presence).

**Explicitly do not touch** (per D-05): `TestAdvancedSessionRunConcurrent` (`ort/session_test.go:530-597`, its `time.Sleep(1*time.Millisecond)` at line 556 is a mock work-duration simulator, not a correctness assertion) and `TestTensorDestroyDoesNotBlockUnrelatedRun` (`ort/session_test.go:898+`, already uses `require.Eventually` similarly to the not-in-scope `TestAdvancedSessionDestroyDoesNotBlockUnrelatedRun` shape — re-verify against the exact 3 names in D-03 before editing any test not named there).

---

### `ort/environment_stress_test.go` (NEW — test, event-driven/batch)

**Analog:** `ort/environment_test.go:243-308` (`TestConcurrentInitialization`, `TestConcurrentDestroy`) — explicitly named in CONTEXT.md D-07 as the foundation to build on. Also reuse `resetEnvironmentState()` (`ort/environment_test.go:11-32`).

**Foundation pattern** (`ort/environment_test.go:243-277`, `TestConcurrentInitialization`):
```go
func TestConcurrentInitialization(t *testing.T) {
	resetEnvironmentState()

	if err := SetSharedLibraryPath("/nonexistent/path.so"); err != nil {
		t.Fatalf("unexpected error setting library path: %v", err)
	}

	var wg sync.WaitGroup
	concurrency := 10

	mu.Lock()
	refCount = 1
	mu.Unlock()

	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = InitializeEnvironment()
		}()
	}
	wg.Wait()

	mu.Lock()
	expectedCount := 1 + concurrency
	if refCount != expectedCount {
		t.Errorf("expected refCount to be %d after concurrent inits, got %d", expectedCount, refCount)
	}
	mu.Unlock()

	resetEnvironmentState()
}
```

**Apply this way — new file, package `ort`, gated behind `testing.Short()` (per D-06):**
```go
package ort

import (
	"sync"
	"testing"
)

func TestStressConcurrentInitDestroy(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping stress test in short mode")
	}
	resetEnvironmentState()
	defer resetEnvironmentState()

	const goroutines = 100
	const iterations = 1000

	if err := SetSharedLibraryPath("/nonexistent/path.so"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var wg sync.WaitGroup
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				_ = InitializeEnvironment()
				_ = DestroyEnvironment()
			}
		}()
	}
	wg.Wait()

	mu.Lock()
	got := refCount
	mu.Unlock()
	if got != 0 {
		t.Fatalf("expected refCount == 0 after balanced init/destroy cycles, got %d", got)
	}
}
```
The 2+ additional stress tests (D-06 discretion — "Mixed Operations Under Load" per issue #24) should follow the exact same shape: `resetEnvironmentState()` → spawn goroutines calling a **fixed** number of iterations (not wall-clock duration — see RESEARCH.md Pitfall #1) of a mix of `InitializeEnvironment`/`DestroyEnvironment`/`IsInitialized`/`GetVersionString`/`SetSharedLibraryPath`/`SetLogLevel` → `wg.Wait()` → assert `refCount` invariants and no panic. `-race` safety comes from the same source as `TestConcurrentInitialization`/`TestConcurrentDestroy`: no real FFI/unsafe calls, just refcount/mutex logic against a nonexistent library path.

---

### `.github/workflows/ci.yml` (config, batch) — new stress-test job

**Analog:** `.github/workflows/ci.yml:108-126` (`test-race-ort-concurrency`)

**Existing sibling job to copy structure from:**
```yaml
  test-race-ort-concurrency:
    name: Test Race (ORT concurrency subset)
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6

      - name: Setup Go
        uses: actions/setup-go@4a3601121dd01d1626a1e23e37211e3254c1c06c # v6
        with:
          go-version: ${{ env.GO_VERSION_STABLE }}
          cache: true

      - name: Get dependencies
        run: go mod download

      - name: Run race detector on ORT concurrency tests
        run: |
          go test -race ./ort -run '...pipe-separated test names...'
```

**Apply this way — new, separate job (per D-08: do NOT merge into `test-race-ort-concurrency`'s regex):**
```yaml
  test-race-ort-stress:
    name: Test Race (ORT stress suite)
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - name: Checkout code
        uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6

      - name: Setup Go
        uses: actions/setup-go@4a3601121dd01d1626a1e23e37211e3254c1c06c # v6
        with:
          go-version: ${{ env.GO_VERSION_STABLE }}
          cache: true

      - name: Get dependencies
        run: go mod download

      - name: Run stress tests under race detector
        run: go test -v -race -run=TestStress -count=50 -parallel=4 ./ort/...
```
`GO_VERSION_STABLE` is already defined at the workflow-level `env:` block (`.github/workflows/ci.yml:11`) — reuse it, do not hardcode a Go version. Pin the same `actions/checkout` and `actions/setup-go` SHAs already used elsewhere in this file for consistency.

---

### `TESTING.md` (docs) — "Running stress tests" section

**Analog:** `TESTING.md:258-285` (`### Local Pre-commit Checks`) for tone/structure; `TESTING.md:247-256` (`### GitHub Actions`) for where to note the new CI job.

**Structure to mirror** (existing section, prose + a single fenced shell block per step):
- Heading: `### Local Pre-commit Checks`
- One-line lead-in ("Install repo-managed hooks once:")
- A fenced `bash` block with the one-liner (e.g. `make install-hooks`)
- A second lead-in + fenced `bash` block (e.g. `make precommit`)
- A bullet list summarizing what the command runs

**Apply this way** — add a new `###` subsection titled `### Running Stress Tests`, placed right after `### Local Pre-commit Checks` and before `### Local CI Simulation`, following that same shape:
- Lead-in paragraph explaining `ort/environment_stress_test.go` contains `testing.Short()`-gated stress tests exercising many concurrent `InitializeEnvironment`/`DestroyEnvironment` cycles to catch refcount corruption, deadlocks, and panics under load, and that they are skipped by default (`go test ./...`), running only when explicitly targeted.
- A fenced `bash` block with the local invocation:
  ```
  go test -race -run TestStress -count=10 ./ort/...
  ```
- A closing note that CI runs these in a dedicated `test-race-ort-stress` job with `-count=50` and a 10-minute timeout, separate from the `test-race-ort-concurrency` job.

Also update the `### GitHub Actions` bullet list (`TESTING.md:251-256`) to add a line for the new stress job, following the existing bullet style (`- **Real-model Integration Job**: ...`).

---

### `Makefile` (config) — `-short` wiring for `test`/`precommit` targets

**Analog:** `Makefile:104-114` (`test`, `test-race` targets)

**Current `test` target:**
```makefile
## test: Run tests
test:
	@echo "$(YELLOW)Running tests...$(NC)"
	$(GO) test -v -cover ./...
	@echo "$(GREEN)✓ Tests complete$(NC)"
```

**Current `precommit`'s test step** (`Makefile:361`):
```makefile
	$(GO) test ./...
```

**Apply this way (only if the planner/user confirms — see Open Question below):** add `-short` to both, e.g. `$(GO) test -v -cover -short ./...` and `$(GO) test -short ./...`, so `testing.Short()` actually gates the new stress tests out of default runs per D-06's literal wording. Leave `test-race` (`Makefile:110-114`, targets the curated `-race` concurrency subset via a different mechanism) unchanged — it doesn't run `TestStress*` today and won't unless someone widens its scope separately.

## Shared Patterns

### Sentinel error + `errors.Is` detection
**Source:** `ort/bootstrap.go:49-50, 463-465` (`errBootstrapRedirectPolicy` / `isBootstrapRedirectPolicyError`)
**Apply to:** `ort/bootstrap.go` (new `ErrUnsupportedPlatform`), `examples/inference/main.go` (detection call site)
```go
var ErrUnsupportedPlatform = errors.New("unsupported platform for ONNX Runtime bootstrap")
func IsUnsupportedPlatformError(err error) bool { return errors.Is(err, ErrUnsupportedPlatform) }
```

### Test-mock function-variable injection for FFI boundary
**Source:** `ort/environment.go:39-50` (package-level `var runSessionFunc func(...) uintptr`, etc.) + `ort/session_test.go:690-701` (tests reassign these under `mu.Lock()`)
**Apply to:** `ort/session_test.go` (3 modified tests), `ort/environment_stress_test.go` (indirectly, via `resetEnvironmentState()` clearing these same globals)
```go
mu.Lock()
ortAPI = &OrtApi{}
runSessionFunc = func(...) uintptr { ... }
releaseSessionFunc = func(handle uintptr) { ... }
mu.Unlock()
```

### `resetEnvironmentState()` test fixture
**Source:** `ort/environment_test.go:10-32`
**Apply to:** every test in `ort/session_test.go` and the new `ort/environment_stress_test.go` — always call at test start (and via `defer` at test end for stress tests, matching `TestAdvancedSessionRunAndDestroyConcurrent`'s `defer resetEnvironmentState()` at line 681).

### Channel rendezvous (no sleep) for "goroutine reached this point" proof
**Source:** `ort/session_test.go:683-701` (`runStarted` + `sync.Once` + `close()`)
```go
runStarted := make(chan struct{})
var closeRunStarted sync.Once
runSessionFunc = func(...) uintptr {
	closeRunStarted.Do(func() { close(runStarted) })
	<-allowRunReturn
	return 0
}
...
<-runStarted // blocks until rendezvous fires, no polling
```
**Apply to:** all 3 modified tests already use this pattern for the "run started" side; it stays unchanged. Only the *destroy/tensor-destroy-not-yet-returned* assertion (currently `require.Never`/`require.Eventually`) is being replaced per D-04.

### Watchdog-as-hang-detector-only (never the assertion itself)
**Source:** RESEARCH.md Pattern 2/3, derived from stdlib `testing.T.FailNow` contract ("must be called from the goroutine running the test... not from other goroutines")
**Apply to:** all 3 modified `ort/session_test.go` tests
```go
select {
case err := <-destroyErrCh:
	t.Fatalf("destroy returned before run completed (err=%v)", err)
case <-time.After(500 * time.Millisecond):
	// expected: still blocked
}
```
Never spawn a separate watchdog goroutine that calls `t.Fatal`.

## No Analog Found

None. Every file in scope has at least a role-match analog in the existing codebase; this phase is a hardening pass entirely on top of existing, well-established internal patterns (no new architecture, no new packages, no new dependencies).

## Metadata

**Analog search scope:** `ort/*.go`, `ort/*_test.go`, `examples/inference/`, `.github/workflows/ci.yml`, `Makefile`, `TESTING.md`
**Files scanned:** `ort/bootstrap.go`, `ort/bootstrap_test.go`, `ort/environment.go`, `ort/environment_test.go`, `ort/session_test.go`, `examples/inference/main.go`, `.github/workflows/ci.yml`, `Makefile`, `TESTING.md`
**Pattern extraction date:** 2026-07-21

## Open Question for Planner

RESEARCH.md's Open Question #1 (Makefile/CI `-short` wiring) is unresolved in CONTEXT.md — D-08 enumerates CI/TESTING.md changes explicitly but is silent on `Makefile`'s `test`/`precommit` targets and the two CI unit-test steps. Without this wiring, `testing.Short()` gating in `ort/environment_stress_test.go` is cosmetic (stress tests would still run in every default `go test ./...`/`make precommit`). Planner should either fold this into a TST-02 plan task or explicitly flag it back to the user before implementation — this is a scope-boundary judgment call, not a pattern question, so it's called out here rather than silently assumed.
