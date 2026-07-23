---
spike: 002b
name: slog-handler-sink
type: comparison
validates: "Given a consumer-supplied slog.Handler, when diagnostics are emitted, then standard structured output and silent defaults work without a custom logging vocabulary"
verdict: VALIDATED
related: [002a, 002c]
tags: [logging, slog, api]
---

# Spike 002b: `slog.Handler` Sink

## What This Validates

A consumer-supplied standard-library `slog.Handler` can provide:

- a process-wide silent default before runtime initialization;
- structured fields and levels with no project-owned logging types;
- race-safe concurrent emission and reconfiguration;
- preservation of attributes already attached to a consumer `*slog.Logger`;
- no automatic logging path for errors returned to callers.

## Research

Go splits structured logging into a `Logger` frontend and `Handler` backend.
The documentation says handler methods may be called concurrently and tells
users to emit through `Logger`, not by calling `Handler` directly. The
prototype therefore stores a `*slog.Logger` built from the configured handler.

`slog.DiscardHandler`, available since Go 1.24 and therefore inside the
project's Go 1.25 baseline, is never enabled and supplies the no-op default
without project code.

| Property | `slog.Handler` sink |
|----------|---------------------|
| Project-owned exported interface methods | 0 |
| Additional project-owned value types | 0 |
| Configuration surface | one function accepting `slog.Handler` |
| Default | `slog.DiscardHandler` |
| Consumer `slog` wiring | `SetDiagnosticHandler(logger.Handler())` |
| Consumer Zap wiring | a complete four-method `slog.Handler` adapter |
| Concurrency | atomic configuration plus the standard handler contract |

Primary references:

- https://go.dev/src/log/slog/handler.go
- https://go.dev/blog/slog
- https://go.dev/doc/go1.24
- https://pkg.go.dev/log/slog

## How to Run

```bash
go test -race ./.planning/spikes/002-b-slog-handler-sink
go test -run '^$' -bench BenchmarkDiscardedDiagnostic -benchmem ./.planning/spikes/002-b-slog-handler-sink
```

## Investigation Trail

1. Used the standard `slog.Attr` and `slog.Level` vocabulary at diagnostic call
   sites.
2. Used `slog.DiscardHandler` rather than writing another no-op implementation.
3. Wrapped the configured handler in `slog.Logger` so `Enabled`, record
   construction, and handler invocation follow the standard contract.
4. Extracted a handler from a logger with pre-bound attributes to prove those
   attributes survive consumer wiring.
5. Implemented a minimal counting handler to expose the four-method adapter
   burden for consumers whose logging backend is not already slog-compatible.

## Results

**Verdict: VALIDATED as a viable design; comparison winner is not yet chosen.**

The race suite passed:

```text
ok  github.com/amikos-tech/pure-onnx/.planning/spikes/002-b-slog-handler-sink  1.419s
```

The silent path benchmark measured:

```text
BenchmarkDiscardedDiagnostic-16  10.51 ns/op  0 B/op  0 allocs/op
```

The handler design has the smallest project-owned public surface: one
configuration function, with levels, fields, concurrency rules, and the no-op
implementation supplied by the standard library. A consumer's logger-level
attributes survive `logger.Handler()` extraction.

Its main cost is adapter complexity outside the slog ecosystem. A direct
backend adapter must correctly implement all four `slog.Handler` methods, not
just one diagnostic method. That burden is acceptable when an existing slog
handler adapter is already available, but heavier for a small custom or Zap
integration.
