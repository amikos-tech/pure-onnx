---
spike: 002a
name: custom-diagnostic-sink
type: comparison
validates: "Given a narrow custom sink with a no-op default, when bootstrap/finalizer diagnostics are emitted concurrently, then structured fields arrive without output or races by default"
verdict: VALIDATED
related: [002b, 002c]
tags: [logging, api, noop]
---

# Spike 002a: Custom Diagnostic Sink

## What This Validates

A project-owned one-method interface can provide:

- a process-wide silent default before runtime initialization;
- structured info and warning diagnostics;
- race-safe concurrent emission and reconfiguration;
- consumer adapters without a bundled third-party logging dependency;
- no automatic logging path for errors returned to callers.

## Research

The referenced `chroma-go` design proves the consumer-wired/no-op-default
pattern, but its general-purpose `Logger` has eleven methods plus its own field
helpers. Phase 2 only needs non-returnable library diagnostics.

Go's `slog` design separates the `Logger` frontend from the `Handler` backend.
Its standard field and level types reduce adapter work, while a custom
framework-neutral interface avoids making `slog` the public vocabulary.

| Property | Custom sink |
|----------|-------------|
| Exported interface methods | 1 |
| Additional exported value types | `Level`, `Field` |
| Default | private no-op implementation |
| Consumer `slog` wiring | one-method adapter |
| Consumer Zap wiring | one-method adapter; dependency remains consumer-side |
| Concurrency | atomic configuration; sink contract requires safe concurrent use |

Primary references:

- `../chroma-go/pkg/logger/logger.go`
- `../chroma-go/pkg/logger/noop_logger.go`
- https://go.dev/blog/slog
- https://pkg.go.dev/log/slog

## How to Run

```bash
go test -race ./.planning/spikes/002-a-custom-diagnostic-sink
go test -run '^$' -bench BenchmarkDiscardedDiagnostic -benchmem ./.planning/spikes/002-a-custom-diagnostic-sink
```

## Investigation Trail

1. Reduced the reference logger from eleven methods to one diagnostic method.
2. Kept the no-op implementation private so it adds no public API.
3. Stored the interface inside an atomically replaced concrete box; storing
   different interface implementations directly in `atomic.Value` would panic
   because `atomic.Value` requires one consistent concrete type.
4. Kept emission private so the package does not become a general logging
   facade.
5. Added a consumer-side `slog` adapter to measure the translation burden.

## Results

**Verdict: VALIDATED as a viable design; comparison winner is not yet chosen.**

The race suite passed:

```text
ok  github.com/amikos-tech/pure-onnx/.planning/spikes/002-a-custom-diagnostic-sink  1.440s
```

The silent path benchmark measured:

```text
BenchmarkDiscardedDiagnostic-16  18.37 ns/op  64 B/op  1 allocs/op
```

The one-method interface is substantially smaller than the reference logger
and gives non-slog consumers the simplest adapter seam. The cost is two
project-owned public value types and field conversion in every adapter.

The allocation is caused by the variadic `[]Field` escaping through an
interface call, even when the concrete implementation is the no-op sink. It is
unlikely to matter for rare cleanup/fallback diagnostics, but the standard
`slog` variants can now be compared against a measured baseline.
