---
spike: 002c
name: slog-logger-sink
type: comparison
validates: "Given a consumer-supplied *slog.Logger, when diagnostics are emitted, then wiring is ergonomic, structured, silent by default, and race-safe"
verdict: VALIDATED
related: [002a, 002b]
tags: [logging, slog, api]
---

# Spike 002c: `*slog.Logger` Sink

## What This Validates

A consumer-supplied standard-library logger can provide:

- a process-wide silent default before runtime initialization;
- structured fields and levels with no project-owned logging types;
- race-safe concurrent emission and reconfiguration;
- direct preservation of consumer configuration and bound attributes;
- no automatic logging path for errors returned to callers.

## Research

`slog.Logger` is the standard frontend that creates records and passes them to
its handler. `LogAttrs` is the efficient structured emission method. A logger
created with `slog.DiscardHandler` supplies the silent default inside the
project's Go 1.25 baseline.

Accepting the logger itself removes `logger.Handler()` from slog consumer
wiring, but it commits the public configuration API to one concrete logging
frontend rather than a sink interface.

| Property | `*slog.Logger` sink |
|----------|---------------------|
| Project-owned exported interface methods | 0 |
| Additional project-owned value types | 0 |
| Configuration surface | one function accepting `*slog.Logger` |
| Default | `slog.New(slog.DiscardHandler)` |
| Consumer `slog` wiring | `SetDiagnosticLogger(logger)` |
| Consumer Zap wiring | build or obtain a `slog.Handler`, then wrap it with `slog.New` |
| Concurrency | atomic configuration plus the logger's handler contract |

Primary references:

- https://pkg.go.dev/log/slog
- https://go.dev/blog/slog
- https://go.dev/src/log/slog/logger.go

## How to Run

```bash
go test -race ./.planning/spikes/002-c-slog-logger-sink
go test -run '^$' -bench BenchmarkDiscardedDiagnostic -benchmem ./.planning/spikes/002-c-slog-logger-sink
```

## Investigation Trail

1. Accepted a configured `*slog.Logger` directly.
2. Used `slog.DiscardHandler` for a silent default with no custom no-op type.
3. Used `LogAttrs` so disabled diagnostics can remain allocation-free.
4. Passed a logger with pre-bound attributes to prove configuration survives
   without handler extraction.
5. Wrapped a non-slog backend handler with `slog.New` to expose the extra
   consumer step outside the slog ecosystem.

## Results

**Verdict: VALIDATED as a viable design, but not recommended.**

The race suite passed:

```text
ok  github.com/amikos-tech/pure-onnx/.planning/spikes/002-c-slog-logger-sink  1.435s
```

The silent path benchmark measured:

```text
BenchmarkDiscardedDiagnostic-16  10.48 ns/op  0 B/op  0 allocs/op
```

Direct logger injection is the most convenient option for a consumer already
using slog. It preserves bound attributes and matches the handler variant's
zero-allocation no-op path.

The convenience difference is only `logger` versus `logger.Handler()` at
configuration time. In exchange, this option exposes a concrete logging
frontend where the phase decision calls for a consumer-wired sink interface.
Non-slog consumers still need a complete handler adapter plus `slog.New`, so
the concrete logger does not improve their integration.
