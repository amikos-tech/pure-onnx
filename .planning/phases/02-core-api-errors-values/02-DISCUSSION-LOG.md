# Phase 2: Core API — Errors & Values - Discussion Log

> Superseded by c7e58011: the shipped diagnostics default is a stderr TextHandler at
> LevelWarn, not a silent DiscardHandler.

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-23
**Phase:** 2-Core API — Errors & Values
**Areas discussed:** Session value flow, Value inspection and ownership, Error taxonomy, Error messages and logging, ORT status lifetime spike, Diagnostic logging contract spike

---

## Session value flow

| Option | Description | Selected |
|--------|-------------|----------|
| Add `RunWithValues(inputs, outputs []Value) error` | Preserve `Run()`, accept heterogeneous caller-owned values per call, and reuse preallocated outputs. | ✓ |
| Add `RunValues(inputs []Value) ([]Value, error)` | Let ONNX Runtime allocate outputs and return wrapped values to the caller. | |
| Support both output modes | Offer both preallocated and runtime-allocated outputs. | |
| Replace the existing constructor/`Run()` flow | Move entirely to call-bound values and migrate current users. | |

**User's choice:** Add `RunWithValues(inputs, outputs []Value) error`.
**Notes:** Existing `Run()` stays unchanged. Names remain fixed on the session, values are borrowed during the call, and outputs are filled in place. A separate `RunWithValues` spike was considered but intentionally skipped because `valuesToHandles`, per-value run leases, and existing concurrency tests already establish feasibility; a spike would duplicate Phase 2 implementation work.

---

## Value inspection and ownership

| Option | Description | Selected |
|--------|-------------|----------|
| Sealed minimal interface plus typed helpers | Seal `Value`, retain a small method set, and add exact-type `IsTensor`/`AsTensor[T]` helpers. | ✓ |
| Open interface plus typed helpers | Keep external implementations possible and expose enough plumbing for sessions to use them. | |
| Metadata-rich `TensorValue` interface | Add runtime shape and element-type inspection to a second interface. | |
| Universal non-generic value wrapper | Represent unknown runtime types and provide conversion/copy utilities. | |

**User's choice:** Sealed minimal interface plus typed helpers.
**Notes:** Values remain caller-owned. `AsTensor[T]` checks the exact tensor element type and does not perform numeric conversion.

---

## Error taxonomy

| Option | Description | Selected |
|--------|-------------|----------|
| Typed `ORTError` plus lean sentinels | Use `errors.As` for native details, `errors.Is` for actionable categories, and preserve wrapped system causes. | ✓ |
| One universal structured error type | Convert every validation, lifecycle, bootstrap, and native failure into one error schema. | |
| Sentinels for every condition | Represent each failure class with an exported sentinel. | |
| Typed errors only for native statuses | Structure ORT status failures but leave validation and lifecycle errors string-only. | |

**User's choice:** Typed `ORTError` plus lean sentinels.
**Notes:** Exact sentinel names and grouping are left to planning, provided the set remains small and actionable.

---

## Error messages

| Option | Description | Selected |
|--------|-------------|----------|
| Preserve useful prefixes, not exact text | Retain recognizable operation wording where practical while making `errors.Is`/`errors.As` the stable contract. | ✓ |
| Guarantee exact existing strings | Treat every current error string as compatibility-sensitive. | |
| Rewrite everything into one format | Replace current messages with a new canonical schema. | |

**User's choice:** Preserve useful prefixes, not exact text.
**Notes:** Messages include operation and relevant native detail without file/line or stack traces.

---

## Diagnostic logging

| Option | Description | Selected |
|--------|-------------|----------|
| Return errors only | Add no logging integration; callers manually log returned errors. | |
| `slog.LogValuer` on `ORTError` | Make typed errors render structured fields through `slog`, without package-emitted logs. | |
| Optional injected logger | Allow the package to emit opt-in structured diagnostics. | |
| Automatic default logging | Emit errors through the process default logger. | |
| Narrow opt-in sink with default no-op | Adapt `chroma-go`'s consumer-wired pattern to a smaller interface and log only non-returnable diagnostics. | ✓ |
| Full `chroma-go`-style abstraction | Copy its multi-level, context-aware, `With`/`Sync` logger surface. | |

**User's choice:** Narrow opt-in structured logging sink with a default no-op.
**Notes:** The user referenced `../chroma-go/pkg/logger/` as the design inspiration. The chosen variant stays smaller, supports consumer adapters, and never logs a failure that is also returned.

---

## Pre-planning spike: ORT status lifetime

| Approach | Description | Selected |
|----------|-------------|----------|
| Instrumented callbacks only | Prove exact release count, ordering, copied-message lifetime, and concurrency without a native runtime. | |
| Real native round trip only | Exercise `CreateStatus`, status accessors, and `ReleaseStatus` through purego against ONNX Runtime. | |
| Combined callback and native proof | Use race-checked callbacks for exact accounting and a separate real ABI round trip for native compatibility. | ✓ |
| Public inference failure only | Trigger an error through the current public API without access to native code or release accounting. | |

**User's choice:** Run the status-lifetime spike before planning.
**Spike verdict:** VALIDATED.
**Notes:** The helper must treat zero as success, install `defer ReleaseStatus` before any accessor, and copy code/message into Go-owned values. The callback suite passed under `-race`; the real ABI round trip passed separately. Combining the real purego path with `-race` is invalid because race-enabled checkptr rejects this repository's intentional `uintptr` FFI boundary.

---

## Pre-planning spike: diagnostic logging contract

| Option | Description | Selected |
|--------|-------------|----------|
| Custom `DiagnosticSink` | One project-owned method plus custom `Level` and `Field` types; simplest direct adapter for non-slog consumers. | |
| Consumer-supplied `slog.Handler` | Standard interface, levels, fields, concurrency rules, and `DiscardHandler`; configure an internal `slog.Logger`. | ✓ |
| Consumer-supplied `*slog.Logger` | Most direct slog wiring, but exposes a concrete frontend instead of the requested sink interface. | |

**User's choice:** Run all logging variants and fold the validated result into Phase 2 context.
**Spike verdict:** All three variants were viable; `slog.Handler` is RECOMMENDED.
**Notes:** `slog.Handler` preserves the consumer-wired interface boundary with no project-owned logging abstraction and a zero-allocation silent path. The custom sink required public logging types, adapter translation, and one 64-byte allocation in the measured no-op path. `*slog.Logger` saved only one `Handler()` call while losing the interface seam. The production contract is `SetDiagnosticHandler(handler slog.Handler)`; `nil` restores `slog.DiscardHandler`, emission stays private, and non-slog adapters remain consumer-owned.

---

## the agent's Discretion

- Private marker name for the sealed `Value` interface.
- Exact `AsTensor[T]` return convention.
- Exact sentinel names and grouping.
- Internal `ORTError` helper organization.
- Private diagnostic helper/state names and exact structured attribute keys at approved non-returnable call sites.
- Internal refactoring and test organization.

## Deferred Ideas

None. Runtime-allocated outputs, numeric coercion, non-tensor value implementations, and a full logging framework were considered and explicitly excluded from Phase 2.
