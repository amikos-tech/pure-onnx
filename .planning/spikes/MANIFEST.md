# Spike Manifest

## Idea

De-risk the two Phase 2 decisions with the highest irreversible cost before planning: native `OrtStatus` ownership when constructing typed Go errors, and the public contract for a silent-by-default consumer-wired diagnostic logger.

## Requirements

- Native error conversion must copy the ORT code and message before `ReleaseStatus`.
- Every non-zero status must be released exactly once, including concurrent failure paths.
- Returned errors are never logged automatically.
- Diagnostic logging is opt-in and silent by default.
- The logging contract must carry structured fields without adding a third-party logging dependency.
- The logging abstraction must remain substantially smaller than the referenced `chroma-go` logger.

## Spikes

| # | Name | Type | Validates | Verdict | Tags |
|---|------|------|-----------|---------|------|
| 001 | ort-status-lifetime | standard | Given a native ORT status, when it is converted and released, then its code/operation/message remain inspectable and release occurs exactly once under repeated and concurrent failures | VALIDATED | ffi, errors, ownership, concurrency |
| 002a | custom-diagnostic-sink | comparison | Given a narrow custom sink with a no-op default, when bootstrap/finalizer diagnostics are emitted concurrently, then structured fields arrive without output or races by default | VALIDATED (NOT RECOMMENDED) | logging, api, noop |
| 002b | slog-handler-sink | comparison | Given a consumer-supplied `slog.Handler`, when diagnostics are emitted, then standard structured output and silent defaults work without a custom logging vocabulary | VALIDATED (RECOMMENDED) | logging, slog, api |
| 002c | slog-logger-sink | comparison | Given a consumer-supplied `*slog.Logger`, when diagnostics are emitted, then wiring is ergonomic, structured, silent by default, and race-safe | VALIDATED (NOT RECOMMENDED) | logging, slog, api |

## Planning Outcomes

- [Spike 001](001-ort-status-lifetime/) validates a single status-conversion
  helper that installs `defer ReleaseStatus` first, then copies the native code
  and message into a Go-owned `ORTError`.
- [Spike 002 comparison](002-logging-contract-comparison.md) recommends a
  consumer-supplied `slog.Handler`, stored internally as a `*slog.Logger` and
  defaulting to `slog.DiscardHandler`.
- Logging emission remains private. Planning must audit migrated call sites so
  only non-returnable diagnostics are emitted.

## Considered Without a Spike

`RunWithValues` did not need a throwaway experiment. The existing
`AdvancedSession.valuesToHandles` implementation, per-value run leases, and
concurrency tests already establish feasibility; a separate spike would
duplicate Phase 2 implementation work rather than reduce uncertainty.
