# Phase 2: Core API — Errors & Values - Context

**Gathered:** 2026-07-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 2 completes the existing partial polymorphic-value work in `ort/` and makes failures across environment, tensor, session, and bootstrap flows actionable and programmatically inspectable. It adds a backward-compatible per-call value path using caller-provided tensors, a package-sealed `Value` surface with typed inspection helpers, structured native ONNX Runtime errors, lean error sentinels, and silent-by-default opt-in diagnostic logging.

Existing constructor-bound `AdvancedSession.Run()`, typed `Tensor[T]` use, explicit resource ownership, the no-CGO constraint, and the established lock hierarchy remain intact. Runtime-allocated outputs, automatic numeric conversion, and implementations for sequence/map/optional ONNX values are not part of this phase.

</domain>

<decisions>
## Implementation Decisions

### AdvancedSession value flow
- **D-01:** Keep the existing constructor-bound `AdvancedSession.Run() error` API unchanged for backward compatibility and for the preallocated hot path used by the embedding packages.
- **D-02:** Add `RunWithValues(inputs, outputs []Value) error` as an additive per-call API. Input and output names remain fixed on the session; the supplied value counts must match those names.
- **D-03:** `RunWithValues` borrows all input and output values only for the duration of the call. Callers retain ownership and remain responsible for `Destroy()`.
- **D-04:** Outputs are caller-created and preallocated; ONNX Runtime writes into the supplied output values. Phase 2 does not add ORT-allocated output wrapping or a `RunValues(inputs) ([]Value, error)` path.
- **D-05:** `Run()` and `RunWithValues()` must share the same validation, handle-leasing, concurrency, and lock-order behavior rather than becoming separate implementations.

### Value inspection and ownership
- **D-06:** Make `Value` package-sealed with an unexported marker because sessions already accept only values created by `ort`; do not expose raw native handles as an external extension mechanism.
- **D-07:** Keep the public `Value` method set minimal and add `IsTensor` plus an exact-type generic `AsTensor[T]` helper.
- **D-08:** `AsTensor[T]` performs checked extraction only. It must not coerce or copy numeric data between element types.
- **D-09:** Do not add a metadata-heavy `TensorValue` interface or a universal non-generic value wrapper in this phase.

### Error taxonomy and wrapping
- **D-10:** Introduce a public typed `ORTError` for native ONNX Runtime failures. It must retain the native `ErrorCode`, the failed operation, and a Go-owned copy of the native message so callers can inspect it with `errors.As`.
- **D-11:** Centralize native status conversion in one helper. A zero status returns `nil`; for every non-zero status, install `defer ReleaseStatus(status)` before calling any status accessor, then capture the native `ErrorCode` and copy the native message into a Go-owned string. The helper owns exactly one release.
- **D-12:** Add a lean set of public sentinel categories for actionable validation and lifecycle conditions, including invalid arguments, uninitialized state, destroyed resources, and unsupported platform/library conditions. Callers inspect these with `errors.Is`.
- **D-13:** Preserve underlying OS, filesystem, network, and cleanup causes with `%w` and `errors.Join` where applicable. Do not replace useful lower-level causes with string-only errors.
- **D-14:** Apply the error model comprehensively across environment, memory, tensor, session, and bootstrap flows; native status failures must no longer be flattened with `%s`.

### Error messages and diagnostic logging
- **D-15:** Preserve useful existing operation prefixes where practical, but do not make exact error text a compatibility contract. Machine handling belongs to `errors.Is` and `errors.As`.
- **D-16:** Error text should contain the operation, relevant identifiers, and native detail. Do not add source file names, line numbers, or stack traces.
- **D-17:** Expose `SetDiagnosticHandler(handler slog.Handler)` as the consumer configuration API. Passing `nil` restores silent behavior through `slog.DiscardHandler`; do not add project-owned logger, level, or field types.
- **D-18:** Use the private diagnostic emitter only for failures or notices that cannot be returned, such as finalizer cleanup failures, bootstrap fallback/cleanup notices, lock-wait information, and runtime-version warnings. Never automatically log an error that is also returned to the caller.
- **D-19:** Emit through an internal `*slog.Logger` with `Logger.LogAttrs`, using standard `slog.Level` and `slog.Attr` values. A slog consumer wires an existing logger with `logger.Handler()`; Zap and other consumers own their chosen `slog.Handler` bridge, so this module adds no third-party logging dependency.
- **D-20:** Handler configuration must be available before bootstrap or environment initialization so all Go-side `ort` diagnostics can use it. Store the process-wide logger in an atomically replaced configuration box initialized with `slog.New(slog.DiscardHandler)`. This does not redesign ONNX Runtime's native logging callback.

### Spike-validated constraints
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

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project scope and requirements
- `.planning/PROJECT.md` — core value, no-CGO constraint, platform compatibility, and public-API milestone boundaries.
- `.planning/REQUIREMENTS.md` — API-02 and API-03 requirement text and Phase 2 traceability.
- `.planning/ROADMAP.md` §Phase 2 — phase goal and literal success criteria.
- `.planning/phases/01-dx-test-hardening/01-CONTEXT.md` — prior decision to use sentinel errors with `errors.Is` and the lock/lifetime constraints carried into this phase.
- `https://github.com/amikos-tech/onnx-purego/issues/6` — original `Value` interface scope, including heterogeneous tensor arrays, type checks, and future non-tensor values.
- `https://github.com/amikos-tech/onnx-purego/issues/7` — original comprehensive-error-handling scope, including native status mapping and structured logging support.

### Existing value and session API
- `ort/types.go` — current partial `Value` interface, `ValueType`, and stubbed `Status` accessors.
- `ort/session.go` — constructor-bound values, `Run()`, handle leasing, value validation, and session locking.
- `ort/tensor.go` — `Tensor[T]`, explicit ownership, native handle lease, typed data, and destroy synchronization.
- `examples/inference/main.go` — direct public `AdvancedSession` usage that must continue to compile.
- `embeddings/minilm/embedder.go` — cached preallocated tensor/session hot path that motivates preserving `Run()`.
- `embeddings/splade/embedder.go` — heterogeneous input tensors and reusable output buffers.
- `embeddings/openclip/embedder.go` — text and vision sessions using the existing bound-value API.

### Native status and error plumbing
- `.planning/spikes/001-ort-status-lifetime/README.md` — validated status ownership sequence, race/native verification split, commands, and evidence.
- `internal/c_api/onnxruntime_c_api.h` — exact bundled C API contract for `Run`, `GetErrorCode`, `GetErrorMessage`, `ReleaseStatus`, value inspection, and output ownership.
- `ort/constants.go` — existing Go `ErrorCode` values corresponding to native ORT error codes.
- `ort/environment.go` — current status-message extraction/release helpers, runtime initialization, and hard-coded warnings.
- `ort/memory.go` — native status conversion and resource-lifecycle failures.
- `ort/bootstrap.go` — existing `%w`, `errors.Join`, `ErrUnsupportedPlatform`, and hard-coded diagnostic logging patterns.
- `ort/finalizer_log.go` — current finalizer-only logging helper.
- `https://onnxruntime.ai/docs/api/c/struct_ort_api.html` — official ownership and status semantics for ORT C API calls.
- `https://go.dev/blog/go1.13-errors` — official guidance for wrapping, `errors.Is`, `errors.As`, and public error contracts.
- `https://go.dev/blog/module-compatibility` — official guidance to add rather than change exported function signatures and to seal package-owned interfaces.

### Consumer-wired logging reference
- `.planning/spikes/002-logging-contract-comparison.md` — validated comparison and selected `slog.Handler` planning contract.
- `.planning/spikes/002-b-slog-handler-sink/README.md` — recommended variant's API shape, concurrency proof, and zero-allocation no-op evidence.
- `.planning/spikes/MANIFEST.md` — spike requirements, verdicts, and consolidated planning outcomes.
- `.planning/spikes/CONVENTIONS.md` — required verification conventions for purego FFI and comparison spikes.
- `../chroma-go/pkg/logger/logger.go` — user-selected reference for a consumer-supplied structured logger interface.
- `../chroma-go/pkg/logger/noop_logger.go` — no-op default behavior to emulate in a narrower form.
- `../chroma-go/pkg/api/v2/client.go` — `WithLogger` wiring and default no-op installation; inspiration only, not a dependency.
- `https://pkg.go.dev/log/slog` — standard structured-logging concepts and adapter target.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `AdvancedSession.valuesToHandles` and `valueRunLockable`: reuse the existing per-value lease and deduplication machinery for `RunWithValues`.
- `Tensor[T].lockForRun` / `unlockForRun`: already protect value handles from concurrent `Destroy()` during inference.
- `ErrorCode`: already mirrors the native ORT status-code enumeration needed by `ORTError`.
- `getErrorMessage`, `releaseStatus`, and native function-pointer registration: validated foundation for one status-to-error helper that installs release first and copies all status-owned data.
- `ErrUnsupportedPlatform` plus existing `%w`/`errors.Join` usage in bootstrap: established patterns for inspectable sentinel and multi-cause errors.
- `logFinalizerWarning` and existing `log.Printf` call sites: concrete audit and migration points for the new silent-by-default diagnostic handler.

### Established Patterns
- Sessions fix input/output names and embedders cache preallocated tensors by batch size; per-call values should vary without discarding that reuse model.
- Resource wrappers are caller-owned, explicitly destroyed, and guarded by finalizers only as a safety net.
- Lock ordering is documented in `ort/environment.go` and must remain `AdvancedSession.runMu` → `ortCallMu` → global `mu` → value-local lock.
- Bootstrap already preserves many filesystem/network causes with standard wrapping; Phase 2 should extend that pattern rather than introduce a parallel error library.
- Process-global runtime settings are configured before initialization; the diagnostic handler follows the same model while using atomic replacement to remain race-safe.
- Purego ownership tests separate deterministic race-checked callbacks from real native ABI tests because race-enabled checkptr rejects the intentional `uintptr` FFI boundary.

### Integration Points
- `ort/session.go`: additive `RunWithValues`, shared run core, and sealed-value validation.
- `ort/types.go` / `ort/tensor.go`: sealed `Value`, tensor inspection helpers, and typed-error declarations as appropriate.
- `ort/environment.go`, `ort/memory.go`, `ort/tensor.go`, and `ort/session.go`: native status conversion to `ORTError`.
- `ort/bootstrap.go`: sentinel coverage, preserved causes, and migration of approved non-returnable diagnostics to the configured handler.
- `ort/finalizer_log.go`: route non-returnable finalizer failures through the private diagnostic emitter.
- Existing `ort/*_test.go`, integration tests, examples, and all three embedder packages: compatibility and new `errors.Is`/`errors.As` verification.

</code_context>

<specifics>
## Specific Ideas

- The new call shape is explicitly `RunWithValues(inputs, outputs []Value) error`; outputs are filled in place.
- The intended type-inspection experience is an exact generic helper such as `tensor, ok := ort.AsTensor[float32](value)`.
- The logging model keeps `chroma-go`'s consumer-wired/no-op-default behavior without copying its logging abstraction: `ort.SetDiagnosticHandler(logger.Handler())` configures standard structured logging, while `nil` restores silence.

</specifics>

<deferred>
## Deferred Ideas

None — alternatives such as runtime-allocated output values, numeric coercion, non-tensor ONNX value implementations, and a full logging framework were considered and explicitly excluded from Phase 2 rather than added as new roadmap work.

</deferred>

---

*Phase: 2-Core API — Errors & Values*
*Context gathered: 2026-07-23*
