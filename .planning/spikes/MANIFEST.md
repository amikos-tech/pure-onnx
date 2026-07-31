# Spike Manifest

## Idea

De-risk milestone decisions whose public API, FFI ownership, or integration shape would be costly to reverse. Phase 2 covered native `OrtStatus` ownership and consumer-wired diagnostics; Phase 3 adds a compile-only proof for the generalized dense/sparse embedder contract before planning.

## Requirements

- Native error conversion must copy the ORT code and message before `ReleaseStatus`.
- Every non-zero status must be released exactly once, including concurrent failure paths.
- Returned errors are never logged automatically.
- Diagnostic logging is opt-in and silent by default.
- The logging contract must carry structured fields without adding a third-party logging dependency.
- The logging abstraction must remain substantially smaller than the referenced `chroma-go` logger.
- The Phase 3 embedder contract must be generic and additive: existing constructors, methods, and result types remain unchanged.
- The common text method set is `EmbedDocuments`, `EmbedQuery`, and `Close`.
- OpenCLIP may gain forwarding methods, but its existing text and image methods must remain available.
- The root `embeddings` package must remain a dependency-light contract package, not a facade or factory.

## Spikes

| # | Name | Type | Validates | Verdict | Tags |
|---|------|------|-----------|---------|------|
| 001 | ort-status-lifetime | standard | Given a native ORT status, when it is converted and released, then its code/operation/message remain inspectable and release occurs exactly once under repeated and concurrent failures | VALIDATED | ffi, errors, ownership, concurrency |
| 002a | custom-diagnostic-sink | comparison | Given a narrow custom sink with a no-op default, when bootstrap/finalizer diagnostics are emitted concurrently, then structured fields arrive without output or races by default | VALIDATED (NOT RECOMMENDED) | logging, api, noop |
| 002b | slog-handler-sink | comparison | Given a consumer-supplied `slog.Handler`, when diagnostics are emitted, then standard structured output and silent defaults work without a custom logging vocabulary | VALIDATED (RECOMMENDED) | logging, slog, api |
| 002c | slog-logger-sink | comparison | Given a consumer-supplied `*slog.Logger`, when diagnostics are emitted, then wiring is ergonomic, structured, silent by default, and race-safe | VALIDATED (NOT RECOMMENDED) | logging, slog, api |
| 003 | generic-embedder-contract-proof | standard | Given the existing MiniLM, SPLADE, and OpenCLIP APIs, when the proposed root generic contract and additive OpenCLIP forwarding methods are compiled in isolation, then all three conform without changing current constructors, methods, result types, or creating import cycles | VALIDATED | embeddings, generics, api, compatibility |

## Planning Outcomes

- [Spike 001](001-ort-status-lifetime/) validates a single status-conversion
  helper that installs `defer ReleaseStatus` first, then copies the native code
  and message into a Go-owned `ORTError`.
- [Spike 002 comparison](002-logging-contract-comparison.md) recommends a
  consumer-supplied `slog.Handler`, stored internally as a `*slog.Logger` and
  defaulting to `slog.DiscardHandler`.
- Logging emission remains private. Planning must audit migrated call sites so
  only non-returnable diagnostics are emitted.
- [Spike 003](003-generic-embedder-contract-proof/) validates a dependency-free
  root `Embedder[T]` contract and exactly two additive OpenCLIP forwarding
  methods. All existing constructor/method signatures, package dependencies,
  and the complete short repository suite remain compatible under the overlay.

## Considered Without a Spike

`RunWithValues` did not need a throwaway experiment. The existing
`AdvancedSession.valuesToHandles` implementation, per-value run leases, and
concurrency tests already establish feasibility; a separate spike would
duplicate Phase 2 implementation work rather than reduce uncertainty.
