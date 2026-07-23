# Phase 2: Core API — Errors & Values - Research

<user_constraints>
## User Constraints (from CONTEXT.md)

**Provenance:** The decisions, discretion, and deferred scope below are copied from `02-CONTEXT.md`. [VERIFIED: .planning/phases/02-core-api-errors-values/02-CONTEXT.md]

### Locked Decisions

#### AdvancedSession value flow
- **D-01:** Keep the existing constructor-bound `AdvancedSession.Run() error` API unchanged for backward compatibility and for the preallocated hot path used by the embedding packages.
- **D-02:** Add `RunWithValues(inputs, outputs []Value) error` as an additive per-call API. Input and output names remain fixed on the session; the supplied value counts must match those names.
- **D-03:** `RunWithValues` borrows all input and output values only for the duration of the call. Callers retain ownership and remain responsible for `Destroy()`.
- **D-04:** Outputs are caller-created and preallocated; ONNX Runtime writes into the supplied output values. Phase 2 does not add ORT-allocated output wrapping or a `RunValues(inputs) ([]Value, error)` path.
- **D-05:** `Run()` and `RunWithValues()` must share the same validation, handle-leasing, concurrency, and lock-order behavior rather than becoming separate implementations.

#### Value inspection and ownership
- **D-06:** Make `Value` package-sealed with an unexported marker because sessions already accept only values created by `ort`; do not expose raw native handles as an external extension mechanism.
- **D-07:** Keep the public `Value` method set minimal and add `IsTensor` plus an exact-type generic `AsTensor[T]` helper.
- **D-08:** `AsTensor[T]` performs checked extraction only. It must not coerce or copy numeric data between element types.
- **D-09:** Do not add a metadata-heavy `TensorValue` interface or a universal non-generic value wrapper in this phase.

#### Error taxonomy and wrapping
- **D-10:** Introduce a public typed `ORTError` for native ONNX Runtime failures. It must retain the native `ErrorCode`, the failed operation, and a Go-owned copy of the native message so callers can inspect it with `errors.As`.
- **D-11:** Centralize native status conversion in one helper. A zero status returns `nil`; for every non-zero status, install `defer ReleaseStatus(status)` before calling any status accessor, then capture the native `ErrorCode` and copy the native message into a Go-owned string. The helper owns exactly one release.
- **D-12:** Add a lean set of public sentinel categories for actionable validation and lifecycle conditions, including invalid arguments, uninitialized state, destroyed resources, and unsupported platform/library conditions. Callers inspect these with `errors.Is`.
- **D-13:** Preserve underlying OS, filesystem, network, and cleanup causes with `%w` and `errors.Join` where applicable. Do not replace useful lower-level causes with string-only errors.
- **D-14:** Apply the error model comprehensively across environment, memory, tensor, session, and bootstrap flows; native status failures must no longer be flattened with `%s`.

#### Error messages and diagnostic logging
- **D-15:** Preserve useful existing operation prefixes where practical, but do not make exact error text a compatibility contract. Machine handling belongs to `errors.Is` and `errors.As`.
- **D-16:** Error text should contain the operation, relevant identifiers, and native detail. Do not add source file names, line numbers, or stack traces.
- **D-17:** Expose `SetDiagnosticHandler(handler slog.Handler)` as the consumer configuration API. Passing `nil` restores silent behavior through `slog.DiscardHandler`; do not add project-owned logger, level, or field types.
- **D-18:** Use the private diagnostic emitter only for failures or notices that cannot be returned, such as finalizer cleanup failures, bootstrap fallback/cleanup notices, lock-wait information, and runtime-version warnings. Never automatically log an error that is also returned to the caller.
- **D-19:** Emit through an internal `*slog.Logger` with `Logger.LogAttrs`, using standard `slog.Level` and `slog.Attr` values. A slog consumer wires an existing logger with `logger.Handler()`; Zap and other consumers own their chosen `slog.Handler` bridge, so this module adds no third-party logging dependency.
- **D-20:** Handler configuration must be available before bootstrap or environment initialization so all Go-side `ort` diagnostics can use it. Store the process-wide logger in an atomically replaced configuration box initialized with `slog.New(slog.DiscardHandler)`. This does not redesign ONNX Runtime's native logging callback.

#### Spike-validated constraints
- **D-21:** Keep diagnostic emission private to `ort`; this is an observability hook, not a general logging facade. Use `slog.LevelInfo` and `slog.LevelWarn` unless an implementation need proves another level is necessary.
- **D-22:** The handler type cannot enforce the rule against logging returned errors. Planning and verification must include a call-site audit covering finalizers, bootstrap cleanup/fallbacks, lock waits, and runtime-version warnings, with tests proving returned failures do not also emit diagnostics.
- **D-23:** Verify native status conversion with two complementary test layers: instrumented callbacks under `go test -race` for exact release accounting and a real ONNX Runtime ABI round trip without `-race`. Do not disable checkptr to combine them; the repository's intentional `uintptr` purego boundary is incompatible with race-enabled checkptr.

### the agent's Discretion
- Exact private marker name used to seal `Value`.
- Whether `AsTensor[T]` returns `(*Tensor[T], bool)` or `(*Tensor[T], error)`, provided it remains an exact-type checked extraction.
- Exact exported sentinel names and how closely related lifecycle states are grouped, while keeping the public set lean and actionable.
- Internal `ORTError` constructors/accessors and helper organization.
- Private diagnostic helper/state type names and the exact structured attribute keys at each approved call site, subject to D-17 through D-23.
- Test-file organization and internal refactoring needed to share the `Run`/`RunWithValues` implementation without changing observable behavior.

### Deferred Ideas (OUT OF SCOPE)

None — alternatives such as runtime-allocated output values, numeric coercion, non-tensor ONNX value implementations, and a full logging framework were considered and explicitly excluded from Phase 2 rather than added as new roadmap work.
</user_constraints>

**Researched:** 2026-07-23  
**Domain:** Go public API compatibility, purego FFI status ownership, resource lifetime, and structured diagnostics. [VERIFIED: codebase grep; 02-CONTEXT.md]  
**Confidence:** HIGH

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| API-02 | The public API returns comprehensive, wrapped errors with actionable context across environment, tensor, session, and bootstrap flows (#7). | Central `ORTError` conversion, a lean sentinel taxonomy, `%w`/`errors.Join` rules, seven native-status migration points, and the diagnostic call-site audit provide an implementation and verification map. [VERIFIED: .planning/REQUIREMENTS.md; codebase grep] |
| API-03 | A `Value` interface enables polymorphic tensor handling for session inputs and outputs (#6). | A sealed `Value`, exact generic extraction, and an additive `RunWithValues` path reuse the existing session/value lease protocol without changing the bound-value hot path. [VERIFIED: .planning/REQUIREMENTS.md; 02-CONTEXT.md; codebase grep] |
</phase_requirements>

## Project Constraints (from AGENTS.md)

- Never place private repository information in commit messages, pull requests, or related artifacts; notify the user if a requested change would require it. [VERIFIED: user-provided AGENTS.md instructions]
- Use squash merges. [VERIFIED: user-provided AGENTS.md instructions]
- Prefer reuse, deletion, and standard-library facilities over adding code or dependencies. [VERIFIED: user-provided AGENTS.md instructions]
- Explain concepts in plain language with small, direct examples. [VERIFIED: user-provided AGENTS.md instructions]

## Summary

Phase 2 should be an additive hardening pass, not a new value system. The repository already has a public `Value` interface, generic `Tensor[T]`, session-bound `[]Value` fields, per-value read leases, repeated-value deduplication, and a documented lock hierarchy. The implementation should seal that interface, add exact tensor inspection, and route both the existing `Run()` and the new `RunWithValues()` through one internal run core. [VERIFIED: codebase grep; 02-CONTEXT.md]

Native errors are the largest correctness gap. Seven call sites currently read an ORT status message, release the status, and flatten the failure into `%s`: one environment call, one memory call, two tensor calls, and three session calls. The bundled header and official C API require every non-null status to be released and state that the message pointer belongs to the status, so conversion must copy the message before the status is released. The validated Phase 2 spike proves the required order, exact one-release behavior, concurrent fake-callback coverage under `-race`, and a separate real native ABI round trip. [VERIFIED: codebase grep; .planning/spikes/001-ort-status-lifetime/README.md] [CITED: https://onnxruntime.ai/docs/api/c/struct_ort_api.html]

Diagnostics should remain silent until a consumer supplies a standard `slog.Handler`. There are 14 direct `log.Printf` sites in `ort`—12 in bootstrap, one runtime-version warning, and one finalizer helper—plus finalizer callers in the tensor, session, and memory wrappers. Move only unreturnable notices to one private emitter; returned errors retain their normal return path and must not also be logged. [VERIFIED: codebase grep; .planning/spikes/002-b-slog-handler-sink/README.md]

**Primary recommendation:** Add `errors.go` and `diagnostics.go`, seal the existing interface in place, and make the smallest possible edits to current call sites and the session run path; do not add dependencies, wrappers, conversions, or a second concurrency protocol.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| `Value` sealing and tensor inspection | Go public API (`ort`) | Tensor resource wrapper | The interface and generic tensor type already live in `ort/types.go` and `ort/tensor.go`; callers should never handle raw native values. [VERIFIED: codebase grep; 02-CONTEXT.md] |
| Per-call `RunWithValues` | Session orchestration (`ort`) | Native ORT C API boundary | The session owns fixed names and serialization; the C API consumes borrowed `OrtValue*` arrays and writes into preallocated outputs. [VERIFIED: codebase grep; bundled C header] [CITED: https://onnxruntime.ai/docs/api/c/struct_ort_api.html] |
| Native status conversion | FFI boundary (`ort`) | Go error API | ORT owns the status and its message; the boundary must snapshot code/message, release once, and expose Go-owned error data. [VERIFIED: bundled C header; status-lifetime spike] [CITED: https://onnxruntime.ai/docs/api/c/struct_ort_api.html] |
| Validation/lifecycle sentinels | Go public API (`ort`) | Environment, tensor, memory, session, bootstrap call sites | These are caller-actionable categories shared across resource wrappers, while operation-specific detail remains in wrapping text. [VERIFIED: 02-CONTEXT.md; codebase grep] |
| Diagnostic configuration and emission | Process-wide `ort` infrastructure | Consumer-provided `slog.Handler` | Configuration must work before environment setup, while each existing unreturnable notice remains at its owning call site. [VERIFIED: 02-CONTEXT.md; slog spike] |
| Filesystem/network cause preservation | Bootstrap and environment implementation | Go standard error chain | Bootstrap already uses `%w` and `errors.Join`; Phase 2 extends that pattern rather than replacing it. [VERIFIED: codebase grep] [CITED: https://go.dev/blog/go1.13-errors] |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Go standard library: `errors`, `fmt` | Module baseline Go 1.25.0 | Sentinel matching, typed matching, wrapping, and multi-cause cleanup errors | Go's official compatibility guidance treats wrapped sentinels/types as API contracts and provides `errors.Is`, `errors.As`, and `errors.Join`; no error framework is needed. [VERIFIED: go.mod] [CITED: https://go.dev/blog/go1.13-errors] [CITED: https://pkg.go.dev/errors] |
| Go standard library: `log/slog` | Module baseline Go 1.25.0; `DiscardHandler` available since Go 1.24 | Consumer-wired structured diagnostics with a silent default | It supplies the handler contract, levels, attributes, logger frontend, and no-op handler required by D-17 through D-20. [VERIFIED: go.mod] [CITED: https://pkg.go.dev/log/slog] [CITED: https://go.dev/doc/go1.24] |
| Go standard library: `sync/atomic` | Module baseline Go 1.25.0 | Race-safe process-wide logger replacement | `atomic.Pointer[T]` supplies typed atomic `Load`/`Store`, matching the validated spike without adding the global runtime mutex to diagnostic emission. [VERIFIED: slog spike] [CITED: https://pkg.go.dev/sync/atomic] |
| Go standard library: `runtime` | Module baseline Go 1.25.0 | Preserve FFI backing memory through native calls | Existing session/tensor/environment paths use `runtime.KeepAlive` after purego calls that receive raw pointers; the shared run refactor must retain those barriers. [VERIFIED: codebase grep] [CITED: https://pkg.go.dev/runtime#KeepAlive] |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Existing `github.com/ebitengine/purego` | v0.10.1, unchanged | Register `GetErrorCode`, `GetErrorMessage`, `ReleaseStatus`, and `Run` function pointers without CGO | Reuse only at the existing FFI registration boundary; Phase 2 does not install or upgrade it. [VERIFIED: go.mod; go list -m -json] |
| Existing `CstringToGo` | Repository helper | Copy a native NUL-terminated error message into a Go string | Call from the production status converter before the deferred status release runs. [VERIFIED: ort/cstring.go; status-lifetime spike] |
| Existing `valuesToHandles` and `Tensor.lockForRun` | Repository helpers | Validate values and keep tensor handles alive against concurrent `Destroy()` | Reuse from the single shared run core for both bound and per-call values. [VERIFIED: ort/session.go; ort/tensor.go] |
| Go `testing` plus existing test seams | Module baseline Go 1.25.0 | Deterministic unit, race, and native integration coverage | Use fake status callbacks for exact accounting and the actual ORT library only for the non-race ABI test. [VERIFIED: existing tests; status-lifetime spike] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Standard `errors` chain | A third-party error package | Adds a vocabulary and dependency without improving `errors.Is`/`errors.As`; rejected by the locked lean-public-API decision. [VERIFIED: 02-CONTEXT.md] |
| `slog.Handler` | A project-owned logger interface or backend-specific logger | Expands the exported surface or couples `ort` to a backend; the comparison spike selected `slog.Handler`. [VERIFIED: .planning/spikes/002-logging-contract-comparison.md] |
| Exact `AsTensor[T]` assertion | Reflection, numeric coercion, or a universal wrapper | Adds conversion and ownership behavior explicitly excluded from this phase. [VERIFIED: 02-CONTEXT.md] |
| Caller-preallocated output values | ORT-allocated output wrapping | Requires new ownership and destruction semantics explicitly deferred by D-04. [VERIFIED: 02-CONTEXT.md] |
| One shared run core | Separate `Run` and `RunWithValues` implementations | Duplicates the lock and lease protocol and can make the two public paths diverge; D-05 forbids it. [VERIFIED: 02-CONTEXT.md] |

**Installation:**

```bash
# No installation or go.mod change is required for Phase 2.
```

The implementation must remain within the standard library and dependencies already present in `go.mod`. [VERIFIED: 02-CONTEXT.md; go.mod]

## Package Legitimacy Audit

Not applicable: Phase 2 installs no external package and changes no module requirement. The package-legitimacy gate is therefore not triggered. [VERIFIED: 02-CONTEXT.md; Standard Stack recommendation]

## Architecture Patterns

### System Architecture Diagram

```text
Caller
  ├─ Run()
  │    └─ select constructor-bound values while holding session runMu
  └─ RunWithValues(inputs, outputs)
       └─ borrow caller values for this call
                         │
                         ▼
                 one private run core
                         │
       validate receiver, session state, and fixed-name counts
                         │
                         ▼
       runMu → ortCallMu.RLock → global mu snapshot
                         │
                         ▼
       lease/deduplicate Value handles → build C name arrays
                         │
                         ▼
                    OrtApi.Run
                    /        \
             status == 0   status != 0
                  │             │
                  │       status converter
                  │       ├─ defer ReleaseStatus
                  │       ├─ read ErrorCode
                  │       └─ copy message
                  │             │
                  ▼             ▼
       caller output buffers   *ORTError
                  \             /
                   release all value leases

Unreturnable notice only
  → private emitDiagnostic
  → atomic *slog.Logger state
  → consumer slog.Handler or slog.DiscardHandler

Returned error ───────────────X──> diagnostic emitter
```

The diagram follows the established lock hierarchy and the two spike-validated boundaries: borrowed value handles during `Run`, and Go-owned error data after status release. [VERIFIED: ort/environment.go; ort/session.go; both Phase 2 spike READMEs]

### Recommended Project Structure

```text
ort/
├── errors.go                 # NEW: sentinels, ORTError, status conversion
├── errors_test.go            # NEW: Is/As, ordering, exact release, concurrency
├── errors_native_test.go     # NEW: real ORT status ABI round trip, non-race
├── diagnostics.go            # NEW: atomic logger state and private emitter
├── diagnostics_test.go       # NEW: silence, attrs, reconfiguration, no double-log
├── types.go                  # EDIT: seal Value; leave Status compatibility stubs scoped out
├── tensor.go                 # EDIT: marker plus status/sentinel migration
├── session.go                # EDIT: RunWithValues and one shared run core
├── memory.go                 # EDIT: status/sentinel migration
├── environment.go            # EDIT: GetErrorCode registration and diagnostics
├── bootstrap.go              # EDIT: sentinels, wrapping audit, structured notices
├── finalizer_log.go          # EDIT or REMOVE: route through diagnostics.go
└── *_test.go                 # EDIT: flow-specific behavior and call-site audit

.github/workflows/ci.yml      # EDIT: targeted race callback test + native ABI test
```

This layout isolates two cross-cutting policies while keeping resource-specific context at existing call sites. It does not move the FFI registration or session/value ownership code into a new abstraction layer. [RECOMMENDED]

### Pattern 1: Seal the Existing Value Interface

**What:** Add one unexported marker method to `Value` and implement it on `*Tensor[T]`. Keep `Destroy()` and `Type()` unchanged. Add package-level helpers rather than expanding every value implementation with generic behavior. [VERIFIED: 02-CONTEXT.md] [CITED: https://go.dev/blog/module-compatibility]

**When to use:** At the public API boundary where sessions accept values but only `ort` can safely provide native handles and lease behavior. [VERIFIED: codebase grep; 02-CONTEXT.md]

**Recommended shape:**

```go
// Source: https://go.dev/blog/module-compatibility
type Value interface {
	Destroy() error
	Type() ValueType
	ortValue()
}

func (*Tensor[T]) ortValue() {}

func IsTensor(v Value) bool {
	return v != nil && v.Type() == ValueTypeTensor
}

func AsTensor[T any](v Value) (*Tensor[T], bool) {
	tensor, ok := v.(*Tensor[T])
	return tensor, ok && tensor != nil
}
```

Choose the `(*Tensor[T], bool)` form. It mirrors an exact Go type assertion, adds no new error category, and makes a mismatch a normal branch rather than a runtime failure. `IsTensor` is a kind discriminator; `AsTensor` is the exact element-type and non-nil check. [RECOMMENDED]

In-package test doubles that currently implement `Value` must add the private marker. No external custom `Value` implementation was found in the repository or its checked sibling consumer; existing consumers pass package-created `*Tensor[T]` values. [VERIFIED: codebase grep]

### Pattern 2: One Run Core with Bound/Override Selection Inside `runMu`

**What:** Keep both public methods thin and route them to one private method that owns validation, lock acquisition, name conversion, handle leasing, ORT invocation, `KeepAlive`, and status conversion. The private method must acquire `runMu` before reading session-owned fields. [VERIFIED: 02-CONTEXT.md; ort/session.go]

**When to use:** For both constructor-bound values and caller-supplied values, so fixes and concurrency behavior cannot diverge. [VERIFIED: D-05]

**Recommended shape:**

```go
func (s *AdvancedSession) Run() error {
	return s.run(nil, nil, true)
}

func (s *AdvancedSession) RunWithValues(inputs, outputs []Value) error {
	return s.run(inputs, outputs, false)
}

func (s *AdvancedSession) run(inputs, outputs []Value, useBoundValues bool) error {
	if s == nil {
		return fmt.Errorf("run session: %w", ErrInvalidArgument)
	}

	s.runMu.Lock()
	defer s.runMu.Unlock()

	if useBoundValues {
		// Read these fields only after runMu is held; Destroy clears them under runMu.
		inputs = s.inputValues
		outputs = s.outputValues
	}

	// Validate counts, then preserve the existing
	// runMu → ortCallMu → mu → Tensor.runMu order.
	// Reuse valuesToHandles and every existing runtime.KeepAlive barrier.
	return s.runLocked(inputs, outputs)
}
```

The exact private names may differ, but `Run()` must not evaluate `s.inputValues` or `s.outputValues` before the shared core holds `runMu`; doing so races with `Destroy()`. [VERIFIED: ort/session.go lock ownership] [RECOMMENDED]

### Pattern 3: Status Conversion Owns the Native Lifetime

**What:** Register `GetErrorCode` beside the existing message/release functions, and replace all seven open-coded status blocks with a single converter. The converter installs the release defer before either accessor and returns only Go-owned fields. [VERIFIED: codebase grep; status-lifetime spike]

**When to use:** Immediately after every ORT call returning `uintptr` status. A zero handle returns `nil`. [VERIFIED: bundled C header] [CITED: https://onnxruntime.ai/docs/api/c/struct_ort_api.html]

**Recommended shape:**

```go
type ORTError struct {
	Operation string
	Code      ErrorCode
	Message   string
}

func (e *ORTError) Error() string {
	return fmt.Sprintf("%s: ORT code %d: %s", e.Operation, e.Code, e.Message)
}

type statusOps struct {
	getCode     func(uintptr) ErrorCode
	copyMessage func(uintptr) string
	release     func(uintptr)
}

func statusToErrorWithOps(status uintptr, operation string, ops statusOps) error {
	if status == 0 {
		return nil
	}
	defer ops.release(status)

	return &ORTError{
		Operation: operation,
		Code:      ops.getCode(status),
		Message:   ops.copyMessage(status),
	}
}

func statusToError(status uintptr, operation string) error {
	return statusToErrorWithOps(status, operation, statusOps{
		getCode: getErrorCodeFunc,
		copyMessage: func(status uintptr) string {
			return CstringToGo(getErrorMessageFunc(status))
		},
		release: releaseStatusFunc,
	})
}
```

The production helper may snapshot function pointers under the existing global lock before use, but it must not split ownership across accessors or call sites. Add `getErrorCodeFunc` to registration, global clearing, and test reset helpers wherever the message/release pointers are handled. [RECOMMENDED] [VERIFIED: ort/environment.go]

Do not implement `Unwrap` or map native codes automatically to Go sentinels. A native `ErrorCodeInvalidArgument` remains discoverable through `errors.As`; Go validation failures use `ErrInvalidArgument` through `errors.Is`. This keeps two different sources of failure unambiguous. [RECOMMENDED]

### Pattern 4: Lean Sentinel Categories with Contextual Wrapping

**What:** Export only categories that tell a caller what action is possible, and retain operation-specific detail in the wrapper. [VERIFIED: D-12 through D-16] [CITED: https://go.dev/blog/go1.13-errors]

**Recommended set:**

| Sentinel | Covers | Does not cover |
|----------|--------|----------------|
| `ErrInvalidArgument` | Nil/unsupported values, name/value count mismatches, empty required names/paths, invalid options | Native ORT validation codes, which remain `*ORTError`. [RECOMMENDED] |
| `ErrNotInitialized` | Environment/function state required for a call is unavailable | A resource that was valid and then destroyed. [RECOMMENDED] |
| `ErrDestroyed` | Session, tensor, memory-info, or supplied value has already been destroyed | Global runtime initialization failure. [RECOMMENDED] |
| `ErrSharedLibraryNotFound` | Promote the current private shared-library-not-found marker where a caller can select download/configuration fallback | Unsupported OS/architecture. [VERIFIED: ort/bootstrap.go] [RECOMMENDED] |
| Existing `ErrUnsupportedPlatform` | Unsupported `GOOS`/`GOARCH` bootstrap resolution | Library absence on an otherwise supported platform. [VERIFIED: ort/bootstrap.go] |

Keep retry-policy and internal bootstrap-control sentinels private. Use `%w` for the actionable category or underlying OS/network error, and use `errors.Join` only when independent cleanup failures must survive alongside the primary failure. [VERIFIED: current bootstrap pattern] [CITED: https://pkg.go.dev/errors]

### Pattern 5: Atomic, Silent-by-Default Diagnostics

**What:** Store a fully constructed `*slog.Logger` inside an atomically replaced state object. `nil` becomes `slog.DiscardHandler`, and all emission goes through `Logger.LogAttrs`. [VERIFIED: slog spike] [CITED: https://pkg.go.dev/log/slog]

**When to use:** Only when a failure or notice cannot be returned: finalizer cleanup failures, deferred cleanup/fallback notices, periodic lock-wait information, skipped archive entries, and runtime-version warnings. [VERIFIED: D-18; codebase grep]

**Recommended shape:**

```go
type diagnosticState struct {
	logger *slog.Logger
}

var diagnostics = newDiagnosticStore()

func newDiagnosticStore() *atomic.Pointer[diagnosticState] {
	store := &atomic.Pointer[diagnosticState]{}
	store.Store(&diagnosticState{logger: slog.New(slog.DiscardHandler)})
	return store
}

func SetDiagnosticHandler(handler slog.Handler) {
	if handler == nil {
		handler = slog.DiscardHandler
	}
	diagnostics.Store(&diagnosticState{logger: slog.New(handler)})
}

func emitDiagnostic(ctx context.Context, level slog.Level, message string, attrs ...slog.Attr) {
	if ctx == nil {
		ctx = context.Background()
	}
	diagnostics.Load().logger.LogAttrs(ctx, level, message, attrs...)
}
```

Use a small vocabulary such as `operation`, `resource`, `path`, `url`, `wait_duration`, `runtime_version`, `api_version`, and `error`. Pass URLs through their redacted form and never attach credential/environment values. [RECOMMENDED] [VERIFIED: current bootstrap URL-redaction pattern]

### Anti-Patterns to Avoid

- **Public raw-handle extension point:** External values cannot participate safely in private handle leases and the global lock order. Seal `Value`. [VERIFIED: codebase grep; D-06]
- **Two run implementations:** Duplicated validation and locks drift. Both APIs must enter one core. [VERIFIED: D-05]
- **Reading bound values before `runMu`:** `Destroy()` clears session fields under that lock, so an early read introduces a data race. [VERIFIED: ort/session.go]
- **Release after access without an installed defer:** An accessor failure or later early return can leak the native status. Install the defer first. [VERIFIED: status-lifetime spike]
- **Reading the message after release:** The C API says the message pointer is status-owned. Copy before release. [VERIFIED: bundled C header] [CITED: https://onnxruntime.ai/docs/api/c/struct_ort_api.html]
- **Calling `slog.Handler.Handle` directly:** Emit through `*slog.Logger`; the standard documentation explicitly assigns record construction and enablement to the logger frontend. [CITED: https://pkg.go.dev/log/slog]
- **Logging returned errors:** It creates duplicate reporting and violates D-18; tests must audit this behavior because a handler type cannot enforce it. [VERIFIED: D-18; D-22]
- **Using Go 1.26-only helpers:** The local toolchain is newer than the module/CI baseline. In particular, do not use `errors.AsType`, which was added in Go 1.26; use `errors.As`. [VERIFIED: local toolchain; go.mod] [CITED: https://pkg.go.dev/errors]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Typed and category error inspection | String parsing or a custom error registry | `errors.Is`, `errors.As`, `%w`, `errors.Join` | These are the Go compatibility mechanisms and already work through wrapping. [CITED: https://go.dev/blog/go1.13-errors] |
| Native status lifetime | Repeated message/release blocks at each ORT call | One `statusToError` owner | The message is status-owned and every non-null status requires exactly one release. [VERIFIED: bundled C header; status spike] |
| Structured logging API | Custom logger, level, field, or no-op types | `slog.Handler`, `slog.Logger`, `slog.DiscardHandler`, `slog.Attr` | The standard library supplies the complete contract and silent implementation. [CITED: https://pkg.go.dev/log/slog] |
| Logger reconfiguration synchronization | Another mutex in the ORT lock graph | `atomic.Pointer[diagnosticState]` | The logger is immutable after construction and can be swapped as one pointer. [VERIFIED: slog spike] [CITED: https://pkg.go.dev/sync/atomic] |
| Value conversion | Reflection-based element conversion or copying | Exact `v.(*Tensor[T])` assertion | Numeric coercion is explicitly out of scope and would obscure ownership/performance. [VERIFIED: D-08; deferred scope] |
| Value ownership protocol | New public raw handles or a second lease type | Existing `valueRunLockable` and `Tensor.lockForRun` | They already serialize `Destroy()` against in-flight native use. [VERIFIED: ort/session.go; ort/tensor.go] |
| Output allocation | Go wrappers for ORT-created output values | Caller-created `Tensor[T]` outputs | D-04 fixes preallocated caller ownership for this phase. [VERIFIED: D-04] |
| Cleanup error aggregation | A custom multi-error type | `errors.Join` | Bootstrap already uses it and callers retain `Is`/`As` traversal. [VERIFIED: codebase grep] [CITED: https://pkg.go.dev/errors] |

**Key insight:** This phase is mostly policy centralization. The safest implementation removes repeated status/logging decisions and reuses the value/lock machinery already proven by Phase 1. [VERIFIED: codebase grep; 02-CONTEXT.md]

## Common Pitfalls

### Pitfall 1: Releasing an ORT Status Too Late—or Twice

**What goes wrong:** One branch leaks the status, or both a helper and caller release it. [VERIFIED: status-lifetime spike]

**Why it happens:** Ownership is currently open-coded at seven call sites. [VERIFIED: codebase grep]

**How to avoid:** Make `statusToError` the sole owner and install `defer release(status)` before code/message access. Call sites only return the resulting error. [VERIFIED: D-11]

**Warning signs:** A call site mentions `getErrorMessage`, `getErrorCode`, or `releaseStatus` after migration; a fake status reports a release count other than one. [RECOMMENDED]

### Pitfall 2: Retaining Native Message Memory

**What goes wrong:** `ORTError.Message` changes or points to invalid memory after `ReleaseStatus`. [VERIFIED: bundled C header; status spike]

**Why it happens:** `GetErrorMessage` returns a pointer owned by `OrtStatus`. [VERIFIED: bundled C header] [CITED: https://onnxruntime.ai/docs/api/c/struct_ort_api.html]

**How to avoid:** Convert with `CstringToGo` while the status is live and store only the resulting Go string. [VERIFIED: ort/cstring.go; status spike]

**Warning signs:** An error stores `uintptr`, `unsafe.Pointer`, or a byte slice backed by native status memory. [RECOMMENDED]

### Pitfall 3: Testing the Real `uintptr` ABI Under `-race`

**What goes wrong:** Race-enabled checkptr rejects the intentional purego `uintptr` boundary before the lifetime behavior can be tested. [VERIFIED: status-lifetime spike]

**Why it happens:** The fake proof and native ABI proof exercise different risks. [VERIFIED: status-lifetime spike]

**How to avoid:** Keep instrumented callback tests under `-race`; run one real `CreateStatus` round trip without `-race`. Never disable checkptr. [VERIFIED: D-23]

**Warning signs:** CI adds `-gcflags=all=-d=checkptr=0`, combines native status ABI with the race job, or removes either test layer. [RECOMMENDED]

### Pitfall 4: Breaking the Session Lock Order

**What goes wrong:** `Run`, `RunWithValues`, `Destroy`, or environment teardown deadlocks. [VERIFIED: documented lock hierarchy]

**Why it happens:** Session fields, runtime function pointers, and tensor handles have different owners. [VERIFIED: ort/environment.go; ort/session.go; ort/tensor.go]

**How to avoid:** Preserve `AdvancedSession.runMu → ortCallMu → global mu → Tensor.runMu`; snapshot only the fields owned at each level. [VERIFIED: ort/environment.go]

**Warning signs:** A new path acquires `runMu` after `ortCallMu`, locks a value before `ortCallMu`, or reads session fields outside `runMu`. [RECOMMENDED]

### Pitfall 5: Losing Lease Deduplication or `KeepAlive`

**What goes wrong:** Repeated values can self-block around a queued writer, or Go backing arrays become collectible while native code still has their raw addresses. [VERIFIED: existing comments and concurrency tests]

**Why it happens:** A refactor that looks like slice plumbing also carries lifetime behavior. [VERIFIED: ort/session.go]

**How to avoid:** Reuse `valuesToHandles`, keep reverse-order release, preserve repeated comparable-value tests, and retain all name/backing/handle `runtime.KeepAlive` calls after `Run`. [VERIFIED: ort/session.go; session tests]

**Warning signs:** `RunWithValues` converts handles directly, omits a release closure, or moves a `KeepAlive` before the native call. [RECOMMENDED]

### Pitfall 6: Treating Exact Type Mismatch as Numeric Conversion

**What goes wrong:** `AsTensor[float32]` silently copies or converts a `*Tensor[int64]`, changing performance and ownership semantics. [VERIFIED: D-08]

**Why it happens:** The helper name can be mistaken for a conversion API. [RECOMMENDED]

**How to avoid:** Use only `v.(*Tensor[T])` and return `false` for a mismatch or typed-nil tensor. [RECOMMENDED]

**Warning signs:** Reflection over element types, new allocations, or numeric conversion loops appear in the helper. [RECOMMENDED]

### Pitfall 7: Making Native and Go Validation Categories Indistinguishable

**What goes wrong:** Callers cannot tell a local precondition failure from an ONNX Runtime failure with the same broad meaning. [RECOMMENDED]

**Why it happens:** It is tempting to make `ORTError` unwrap to `ErrInvalidArgument`. [RECOMMENDED]

**How to avoid:** Use `errors.Is` for local sentinels and `errors.As` plus `ORTError.Code` for native errors. Do not add automatic cross-mapping. [RECOMMENDED]

**Warning signs:** `errors.Is(nativeErr, ErrInvalidArgument)` succeeds without an explicit wrapping decision at a public boundary. [RECOMMENDED]

### Pitfall 8: Diagnostic Duplication and Sensitive Attributes

**What goes wrong:** A returned failure is also emitted, or logs expose credentials and unredacted URLs. [VERIFIED: D-18; existing bootstrap redaction tests]

**Why it happens:** Replacing every `log.Printf` mechanically ignores whether the call site is a returned error, a fallback, or a security-sensitive path. [VERIFIED: codebase grep]

**How to avoid:** Audit each of the 14 direct sites by behavior, emit only unreturnable notices, use redacted URLs, and attach errors as structured values. [VERIFIED: D-22; codebase grep]

**Warning signs:** An `emitDiagnostic` call immediately precedes `return err`, or attributes include tokens, authorization headers, or raw environment values. [RECOMMENDED]

### Pitfall 9: Misusing the Consumer Handler

**What goes wrong:** Pre-bound consumer attributes disappear, disabled records are constructed unnecessarily, or concurrent calls violate handler expectations. [VERIFIED: slog spike]

**Why it happens:** Calling `Handler.Handle` directly bypasses `slog.Logger`. [CITED: https://pkg.go.dev/log/slog]

**How to avoid:** Store `slog.New(handler)` and emit with `Logger.LogAttrs`; accept `logger.Handler()` for consumer wiring. [VERIFIED: slog spike]

**Warning signs:** Production code constructs `slog.Record` or invokes `Enabled`/`Handle` itself. [RECOMMENDED]

### Pitfall 10: Coupling Tests to Exact Error Text

**What goes wrong:** Harmless context improvements become breaking test failures. [VERIFIED: D-15]

**Why it happens:** Current errors are mostly strings, so tests may reach for equality or substring assertions. [VERIFIED: codebase grep]

**How to avoid:** Assert `errors.Is`, `errors.As`, `ORTError` fields, and required actionable fragments only. [CITED: https://go.dev/blog/go1.13-errors]

**Warning signs:** Tests compare a complete `err.Error()` value. [RECOMMENDED]

## Code Examples

Verified patterns from official sources and repository spikes:

### Inspect a Native ORT Failure

```go
// Source: https://go.dev/blog/go1.13-errors
if err := session.Run(); err != nil {
	var ortErr *ort.ORTError
	if errors.As(err, &ortErr) {
		fmt.Printf("operation=%s code=%d detail=%s\n",
			ortErr.Operation, ortErr.Code, ortErr.Message)
	}
}
```

`errors.As` traverses wrapped and joined errors, so resource-specific context may wrap `*ORTError` without losing typed access. [CITED: https://pkg.go.dev/errors]

### Inspect a Local Lifecycle Category

```go
// Source: https://go.dev/blog/go1.13-errors
if err := session.Run(); errors.Is(err, ort.ErrDestroyed) {
	// Recreate the session instead of parsing its message.
}
```

Sentinel matching is the compatibility contract; exact English text is not. [VERIFIED: D-15] [CITED: https://go.dev/blog/go1.13-errors]

### Extract an Exact Tensor Type

```go
value := getValue()

tensor, ok := ort.AsTensor[float32](value)
if !ok {
	return fmt.Errorf("expected float32 tensor")
}
_ = tensor
```

The assertion succeeds only for a non-nil `*Tensor[float32]`; another element type is not converted. [RECOMMENDED]

### Supply Per-Call Values Without Transferring Ownership

```go
if err := session.RunWithValues(
	[]ort.Value{inputIDs, attentionMask},
	[]ort.Value{embeddings},
); err != nil {
	return err
}

// The caller still owns and later destroys all three tensors.
```

Names remain those fixed at session construction, counts must match, and outputs are filled in place. [VERIFIED: D-02 through D-04]

### Configure Diagnostics Before Initialization

```go
handler := slog.NewJSONHandler(os.Stderr, nil)
ort.SetDiagnosticHandler(handler)
defer ort.SetDiagnosticHandler(nil)

// Configure before bootstrap/environment setup.
```

An existing consumer logger is wired as `ort.SetDiagnosticHandler(logger.Handler())`; passing `nil` restores `slog.DiscardHandler`. [VERIFIED: D-17; slog spike]

### Preserve a Primary and Cleanup Failure

```go
// Source: https://pkg.go.dev/errors#Join
if cleanupErr != nil {
	err = errors.Join(err, fmt.Errorf("clean temporary archive: %w", cleanupErr))
}
```

Use this only when both failures are independently useful; a single causal chain should use `%w`. [VERIFIED: current bootstrap pattern] [CITED: https://pkg.go.dev/errors]

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Flatten native errors with `fmt.Errorf("...: %s", message)` | Store operation, native code, and copied message in `*ORTError` | Go error inspection has supported wrapping and `Is`/`As` since Go 1.13 | Callers can branch without parsing text while retaining human context. [CITED: https://go.dev/blog/go1.13-errors] |
| Project-owned logging interfaces/no-op implementations | Accept standard `slog.Handler` and use `slog.DiscardHandler` | `log/slog` entered the standard library in Go 1.21; `DiscardHandler` was added in Go 1.24 | Phase 2 needs one exported configuration function and no logging dependency. [CITED: https://pkg.go.dev/log/slog] [CITED: https://go.dev/doc/go1.24] |
| Mutex-protected mutable global logger | Atomically replace an immutable logger state pointer | `atomic.Pointer[T]` is available in the project baseline | Emission does not enter the ORT runtime lock graph. [CITED: https://pkg.go.dev/sync/atomic] [VERIFIED: go.mod; slog spike] |
| Open package interface with private handle assertions later | Seal a package-owned interface at declaration | Official Go compatibility guidance recommends an unexported method when implementations must remain package-controlled | Unsupported external implementations fail at compile time rather than deep in `Run`. [CITED: https://go.dev/blog/module-compatibility] |
| Generic-looking status wrapper stubs | Return ordinary Go errors with an inspectable `*ORTError` | Locked Phase 2 design | No separate public native-status ownership API is needed. [VERIFIED: ort/types.go; D-10/D-11] |

**Deprecated/outdated for this phase:**

- The open-coded `getErrorMessage(status)`, `releaseStatus(status)`, `%s` sequence is replaced by the central converter. [VERIFIED: codebase grep; D-11/D-14]
- Direct `log.Printf` inside `ort` is replaced at approved unreturnable-notice sites by the private structured emitter. [VERIFIED: codebase grep; D-17 through D-22]
- The public `Status` type's placeholder accessors should not be expanded into a second error path during Phase 2: the type has no public non-zero producer, and D-10/D-11 select returned `*ORTError` instead. Preserve source compatibility and leave maturity documentation to the documentation phase. [VERIFIED: ort/types.go; roadmap traceability] [RECOMMENDED]
- `errors.AsType` is outside the Go 1.25 baseline even though it exists in the local Go 1.26 toolchain; use `errors.As`. [VERIFIED: go.mod; local toolchain] [CITED: https://pkg.go.dev/errors]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| — | None. All project-state claims were checked in the repository, locked in `CONTEXT.md`, proven by the committed spikes, or cited from official documentation. | — | — |

## Open Questions (RESOLVED)

1. **RESOLVED — Cross-role reuse of the same tensor remains at the current support level.**
   - What we know: `valuesToHandles` deduplicates repeated comparable values within one input or output slice, while current `Run` acquires input and output leases in separate calls. [VERIFIED: ort/session.go]
   - Resolution: Preserve current behavior and scope for both public methods. Add a focused regression test only if existing tests or a known consumer use the same tensor in both roles; do not invent in-place semantics during this phase. [RESOLVED: selected researcher recommendation]

2. **RESOLVED — The real native status test uses `ONNXRUNTIME_LIB_PATH`.**
   - What we know: Local research found an ONNX Runtime 1.23.1 dylib in the project bootstrap cache, while CI's integration job downloads 1.24.1 and exports `ONNXRUNTIME_LIB_PATH`. [VERIFIED: environment probe; .github/workflows/ci.yml]
   - Resolution: Make the test skip unless `ONNXRUNTIME_LIB_PATH` is set, then add its name to the existing non-race ORT integration job. Keep fake status ownership tests self-contained and always runnable. [RESOLVED: selected researcher recommendation]

Both questions are resolved by the selected recommendations above, and both resolutions preserve the locked scope. [RESOLVED]

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| Go toolchain | Build/unit/race tests | ✓ | Local 1.26.5; module baseline 1.25.0 | CI validates the declared 1.25 baseline; avoid local-only 1.26 APIs. [VERIFIED: environment probe; go.mod; CI workflow] |
| `golangci-lint` | Phase 2 changed-code/new-issues check | ✓ | 2.12.2 | Use the repository's `make precommit-lint-new` target; historical full-tree lint cleanup and enforcement remain Phase 5 / CLN-01. [VERIFIED: environment probe; Makefile; .github/workflows/ci.yml; ROADMAP.md] |
| ONNX Runtime shared library | Native status ABI and real run tests | ✓ via local cache; env var unset | Local cache 1.23.1 | Existing CI integration job downloads 1.24.1 and sets `ONNXRUNTIME_LIB_PATH`; native tests skip when unset. [VERIFIED: environment probe; .github/workflows/ci.yml] |
| `purego` module | Native function registration | ✓ | v0.10.1 in `go.mod` | No fallback needed; no upgrade planned. [VERIFIED: go.mod; go list -m -json] |
| Context7 CLI/MCP | Documentation lookup during research only | ✗ | — | Official Go, ONNX Runtime, and OWASP documentation was used directly. [VERIFIED: environment probe] |

**Missing dependencies with no fallback:**

- None. [VERIFIED: environment audit]

**Missing dependencies with fallback:**

- Context7 was unavailable for research; official primary documentation supplied the required evidence. [VERIFIED: environment probe]
- `ONNXRUNTIME_LIB_PATH` is not set in the local shell; unit/race tests remain available, the local cache can support an explicit native run, and CI supplies the variable. [VERIFIED: environment probe; CI workflow]

## Validation Architecture

`workflow.nyquist_validation` is enabled, so Phase 2 needs unit, targeted race, and native integration layers. [VERIFIED: .planning/config.json]

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Go `testing` with module baseline Go 1.25.0. [VERIFIED: go.mod; existing tests] |
| Config file | None; commands and CI selection live in `Makefile` and `.github/workflows/ci.yml`. [VERIFIED: codebase grep] |
| Quick run command | `go test -short ./ort -run 'Test(ORTError|StatusToError|Value|AdvancedSessionRunWithValues|Diagnostic)'` [RECOMMENDED] |
| Full suite command | `go test -short ./...` [VERIFIED: Makefile; .github/workflows/ci.yml] |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| API-02 | Zero status returns nil; nonzero status snapshots code/message and releases exactly once, including accessor failure and concurrent calls | unit + race | `go test -race ./ort -run '^(TestStatusToError|TestORTError)$'` | ❌ Wave 0: `ort/errors_test.go` |
| API-02 | Real `CreateStatus`/read/release ABI preserves code and message | native integration, non-race | `ONNXRUNTIME_LIB_PATH="$ONNXRUNTIME_LIB_PATH" go test ./ort -run '^TestNativeORTStatusRoundTrip$'` | ❌ Wave 0: `ort/errors_native_test.go` |
| API-02 | Local validation/lifecycle errors match lean sentinels; OS/network/cleanup causes remain reachable | unit | `go test -short ./ort -run 'Test(ErrorSentinel|Bootstrap.*Error|.*Destroyed|.*NotInitialized)'` | ⚠️ Existing flow files; new assertions needed |
| API-02 | Default diagnostics are silent; handler wiring, attributes, nil reset, and concurrent reconfiguration work | unit + race | `go test -race ./ort -run '^TestDiagnostic$'` | ❌ Wave 0: `ort/diagnostics_test.go` |
| API-02 | Every approved unreturnable site emits once and returned failures emit zero records | unit/call-site audit | `go test -short ./ort -run 'Test(DiagnosticCallSites|ReturnedErrorsDoNotEmit)'` | ❌ Wave 0 plus existing flow-test edits |
| API-02 | Bootstrap-created directories, installed TGZ/ZIP libraries, and lock files retain Unix-safe permissions | security regression | `go test -short ./ort -run '^TestBootstrapCreatedFilePermissions$'` | ⚠️ `ort/bootstrap_test.go` exists; exact Unix assertion plus Windows skip missing |
| API-03 | `Value` is package-sealed; `IsTensor` and exact `AsTensor[T]` handle match, mismatch, nil, and typed nil | compile/unit | `go test -short ./ort -run TestValue` | ❌ Wave 0: `ort/value_test.go`; existing in-package doubles need marker edits |
| API-03 | `RunWithValues` validates counts/state and uses supplied preallocated handles without changing bound `Run` | unit | `go test -short ./ort -run TestAdvancedSessionRunWithValues` | ⚠️ `ort/session_test.go` exists; cases missing |
| API-03 | Both run paths preserve serialization, destroy waiting, repeated-value leases, and lock order | race | `go test -race ./ort -run '^(TestAdvancedSessionRunWithValues|TestAdvancedSessionRunConcurrent|TestAdvancedSessionRunConcurrentAcrossSessionsSharingTensor|TestAdvancedSessionRunAndDestroyConcurrent|TestTensorDestroyWaitsForInFlightRun|TestValuesToHandlesDeduplicatesRepeatedLockableValue|TestValuesToHandlesReleasesPriorLeasesOnError)$'` | ⚠️ Existing concurrency helpers/cases; per-call variants missing |
| API-03 | Per-call tensors produce correct output against a real model | integration, non-race | `go test ./ort -run '^TestAdvancedSessionRunWithValuesRealModel$'` | ⚠️ Existing model fixture/path; new case missing |

### Sampling Rate

- **Per task commit:** `go test -short ./ort -run 'Test(ORTError|StatusToError|Value|AdvancedSessionRunWithValues|Diagnostic)'` [RECOMMENDED]
- **Per wave merge:** `go test -short ./...` [VERIFIED: established CI command]
- **Race-sensitive task:** `go test -race ./ort -run '^(TestStatusToError|TestORTError|TestDiagnostic|TestAdvancedSessionRunWithValues|TestAdvancedSessionErrorContracts|TestAdvancedSessionDiagnosticPolicy|TestValuesToHandlesDeduplicatesRepeatedLockableValue|TestValuesToHandlesReleasesPriorLeasesOnError|TestAdvancedSessionRunConcurrent|TestAdvancedSessionRunConcurrentAcrossSessionsSharingTensor|TestAdvancedSessionRunAndDestroyConcurrent|TestTensorDestroyWaitsForInFlightRun|TestTensorDestroyConcurrentCallsReleaseOnce|TestTensorStatusConversion|TestTensorDiagnosticPolicy|TestEnvironmentErrorChains|TestEnvironmentStatusConversion|TestConcurrentInitialization|TestConcurrentDestroy|TestDiagnosticRuntimeVersion|TestMemoryInfoStatusConversion|TestDiagnosticMemoryInfo|TestDiagnosticCallSites|TestReturnedErrorsDoNotEmit)$'` [RECOMMENDED]
- **Phase gate:** Full short, targeted race, compile-only, focused permission, vet, `make precommit-lint-new`, and native non-race status/real-model tests green before `$gsd-verify-work`; full-tree lint debt and enforcement remain Phase 5. [RECOMMENDED]

### Wave 0 Gaps

- [ ] `ort/errors_test.go` — fake status store, zero/nonzero behavior, accessor order, message copy, exact release, concurrent conversion, `errors.As`, and sentinel wrapping for API-02. [RECOMMENDED]
- [ ] `ort/errors_native_test.go` — real `CreateStatus` ABI round trip, gated by `ONNXRUNTIME_LIB_PATH`, for API-02. [RECOMMENDED]
- [ ] `ort/diagnostics_test.go` — silent default, standard attributes/levels, consumer-bound attrs, nil reset, concurrent emit/reconfigure, and returned-error zero-emission proof for API-02. [RECOMMENDED]
- [ ] `ort/value_test.go` — kind check plus exact generic extraction matrix for API-03. [RECOMMENDED]
- [ ] `ort/session_test.go` additions — count validation, supplied handle arrays, bound-path compatibility, borrow/Destroy synchronization, and per-call concurrency for API-03. [RECOMMENDED]
- [ ] Flow-test additions in `environment_test.go`, `memory_test.go`, `tensor_test.go`, `session_test.go`, and `bootstrap_test.go` — `Is`/`As`, preserved causes, and approved diagnostic call sites for API-02. [RECOMMENDED]
- [ ] `ort/bootstrap_test.go` — exact `TestBootstrapCreatedFilePermissions` regression covering Unix-safe bootstrap directory, installed-library, and lock-file modes with a Windows-safe POSIX-mode skip. [RECOMMENDED]
- [ ] `.github/workflows/ci.yml` — include fake status and diagnostic concurrency tests in the targeted race job; include native status and `RunWithValues` real-model cases in the existing integration job. [RECOMMENDED]
- [ ] No framework installation is needed. [VERIFIED: existing Go test infrastructure]

## Security Domain

OWASP ASVS 5.0.0 is the latest stable release, dated May 2025; its current chapter names differ from the older ASVS 4.x numbering shown in some templates, so this audit uses the versioned 5.0 chapter map. [CITED: https://github.com/OWASP/ASVS/releases/tag/v5.0.0_release] [CITED: https://github.com/OWASP/ASVS/tree/v5.0.0_release/5.0/en]

### Applicable ASVS Categories

| ASVS 5.0 Category | Applies | Standard Control |
|-------------------|---------|-----------------|
| V2 Validation and Business Logic | yes | Validate receiver/state/name-value counts before FFI; seal values; reject nil, destroyed, and unsupported values with inspectable local errors. [VERIFIED: phase decisions; existing validation] |
| V4 API and Web Service | no | This is an in-process Go library phase, not an HTTP/API service boundary. [VERIFIED: project structure] |
| V5 File Handling | existing bootstrap only | Preserve path containment, archive validation, and wrapped filesystem causes; Phase 2 does not redesign extraction. [VERIFIED: ort/bootstrap.go] |
| V6 Authentication | no | The library exposes no identity/authentication mechanism in this phase. [VERIFIED: phase scope; codebase grep] |
| V7 Session Management | no | `AdvancedSession` is an inference resource, not an authenticated user session. [VERIFIED: ort/session.go] |
| V8 Authorization | no | The phase adds no principal, permission, or access-control boundary. [VERIFIED: phase scope; codebase grep] |
| V11 Cryptography | existing bootstrap only | Preserve existing SHA-256 artifact-integrity verification; do not hand-roll cryptography. [VERIFIED: ort/bootstrap.go] |
| V15 Secure Coding and Architecture | yes | Preserve FFI lifetime, lock order, explicit resource ownership, and the no-CGO boundary. [VERIFIED: PROJECT.md; codebase comments; status spike] |
| V16 Security Logging and Error Handling | yes | Typed errors retain safe context; diagnostics are opt-in, structured, non-duplicative, and exclude credentials/unredacted URLs. [VERIFIED: D-10 through D-23] |

### Known Threat Patterns for Go + purego FFI

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Forged external `Value` implementation supplies an unsafe/native handle | Tampering / Elevation of Privilege | Seal `Value`; keep raw-handle and lease interfaces private. [VERIFIED: D-06; codebase design] |
| Name/value count mismatch crosses the FFI boundary | Tampering / Denial of Service | Validate both input and output counts against fixed names before pointer-array construction. [VERIFIED: D-02; current constructor/run validation] |
| Status message accessed after release or status released twice | Tampering / Denial of Service | One converter installs release first, copies message/code, and owns exactly one release; prove with instrumented callbacks. [VERIFIED: D-11; status spike] |
| Tensor/session handle used during concurrent destruction | Denial of Service | Preserve `runMu`, `ortCallMu`, global lock, and tensor read lease order; release leases after native return. [VERIFIED: lock hierarchy; concurrency tests] |
| Deadlock from inverted lifecycle locks | Denial of Service | Keep the documented order and reuse one run core for both APIs. [VERIFIED: ort/environment.go; D-05] |
| Diagnostic reconfiguration races | Denial of Service | Atomically replace immutable logger state; depend on the standard concurrent handler contract. [VERIFIED: slog spike] [CITED: https://pkg.go.dev/log/slog] |
| Logs disclose tokens, raw URLs, or sensitive environment values | Information Disclosure | Emit selected structured attrs, call URL redaction, and never attach credential/environment contents. [VERIFIED: existing bootstrap security tests; D-16/D-18] |
| Returned failures are logged again | Repudiation / Denial of Service | Audit call sites and assert zero diagnostic records on returned-error paths. [VERIFIED: D-18/D-22] |
| Local Go 1.26 build accidentally introduces APIs unavailable to Go 1.25 consumers | Denial of Service | Compile/test in CI against `go.mod` baseline and avoid `errors.AsType`. [VERIFIED: environment audit; go.mod] [CITED: https://pkg.go.dev/errors] |

## Sources

### Primary (HIGH confidence)

- `.planning/phases/02-core-api-errors-values/02-CONTEXT.md` — locked API, ownership, error, logging, and verification decisions. [VERIFIED: repository]
- `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md`, and `.planning/PROJECT.md` — API-02/API-03 scope, phase success criteria, and no-CGO constraint. [VERIFIED: repository]
- `ort/types.go`, `ort/session.go`, `ort/tensor.go`, `ort/environment.go`, `ort/memory.go`, `ort/bootstrap.go`, `ort/finalizer_log.go`, and tests — current API, locks, status call sites, causes, diagnostics, and validation. [VERIFIED: codebase grep]
- `internal/c_api/onnxruntime_c_api.h` — bundled ownership contracts and C API signatures. [VERIFIED: repository]
- `.planning/spikes/001-ort-status-lifetime/README.md` and tests — validated release/copy sequence, race seam, and native ABI split. [VERIFIED: repository spike rerun]
- `.planning/spikes/002-logging-contract-comparison.md` and `.planning/spikes/002-b-slog-handler-sink/README.md` — selected logging contract and race/no-op evidence. [VERIFIED: repository spike rerun]
- https://onnxruntime.ai/docs/api/c/struct_ort_api.html — `Run`, `GetErrorCode`, `GetErrorMessage`, and `ReleaseStatus` semantics. [CITED: official ONNX Runtime docs]
- https://go.dev/blog/go1.13-errors and https://pkg.go.dev/errors — wrapping, `Is`, `As`, `Join`, and baseline-safe APIs. [CITED: official Go docs]
- https://go.dev/blog/module-compatibility — additive API changes and package-sealed interfaces. [CITED: official Go docs]
- https://pkg.go.dev/log/slog and https://go.dev/doc/go1.24 — handler concurrency, logger emission, `LogAttrs`, and `DiscardHandler`. [CITED: official Go docs]
- https://pkg.go.dev/sync/atomic — typed atomic pointer semantics. [CITED: official Go docs]
- https://github.com/OWASP/ASVS/tree/v5.0.0_release/5.0/en — stable ASVS 5.0 chapter map used for the security audit. [CITED: official OWASP repository]

### Secondary (MEDIUM confidence)

- `go list -m -json` and local tool probes — installed module/tool versions and local runtime availability. [VERIFIED: environment commands]
- `.github/workflows/ci.yml`, `Makefile`, and `TESTING.md` — current quick, race, integration, and lint execution paths. [VERIFIED: repository]
- Checked sibling consumer source referenced by `CONTEXT.md` — existing consumers provide package-created tensors and standard logger handlers rather than custom `Value` implementations. [VERIFIED: codebase grep]

### Tertiary (LOW confidence)

- None. Search-discovered claims were either verified against official documentation or omitted. [VERIFIED: research log]

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH — the plan uses only the repository's Go 1.25 baseline, standard library, existing purego boundary, and validated helpers. [VERIFIED: go.mod; codebase; official Go docs]
- Architecture: HIGH — the locked context, documented lock hierarchy, current resource wrappers, and both Phase 2 spikes agree on the boundaries. [VERIFIED: codebase; context; spikes]
- Pitfalls: HIGH — status ownership, checkptr separation, handler concurrency, call-site counts, and current locking were directly inspected or spike-tested. [VERIFIED: codebase; spike reruns; official docs]
- Security: HIGH for applicable FFI/error/logging controls; web authentication/session/authorization categories are non-applicable to this in-process library phase. [VERIFIED: phase scope; ASVS 5.0 chapter map]

**Research date:** 2026-07-23  
**Valid until:** 2026-08-22 — locked project decisions are stable, but toolchain and ONNX Runtime integration details should be rechecked after 30 days. [RECOMMENDED]
