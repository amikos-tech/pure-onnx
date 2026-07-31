---
spike: 001
name: ort-status-lifetime
type: standard
validates: "Given a native ORT status, when it is converted and released, then its code, operation, and message remain inspectable and release occurs exactly once under repeated and concurrent failures"
verdict: VALIDATED
related: []
tags: [ffi, errors, ownership, concurrency]
---

# Spike 001: ORT Status Lifetime

## What This Validates

Given an `OrtStatus` owned by ONNX Runtime, when Go snapshots it into a typed error and calls `ReleaseStatus`, then:

- the Go error retains its operation, native code, and message;
- the status is released exactly once;
- a null status remains a no-op success;
- independent concurrent failures remain race-free.

## Research

The bundled `internal/c_api/onnxruntime_c_api.h` and the official ONNX Runtime C API agree on the ownership contract:

- `GetErrorMessage` returns memory owned by the status and callers must not free it;
- a non-null `OrtStatus` returned by an ORT call must be freed with `ReleaseStatus`;
- `GetErrorCode` exposes the native machine-readable category.

`purego.RegisterFunc` binds raw function pointers but cannot verify that the Go signature matches the C ABI, so the spike includes a real native round trip in addition to deterministic fake callbacks.

| Approach | Tool/Library | Pros | Cons | Status |
|----------|--------------|------|------|--------|
| Instrumented fake callbacks | Go tests | Proves exact release count, release ordering, concurrency, and post-release copy semantics | Does not validate the native ABI | Included |
| Real `CreateStatus` round trip | ONNX Runtime C API + `purego` | Validates generated API layout and function signatures against a real runtime | Cannot instrument native `ReleaseStatus` internals | Included |
| Trigger a public inference failure | Existing `ort` API | Exercises a consumer-visible path | Current API flattens status code and cannot expose release accounting | Not sufficient alone |

**Chosen approach:** combine instrumented fake callbacks with a real native `CreateStatus`/`GetErrorCode`/`GetErrorMessage`/`ReleaseStatus` round trip.

Primary references:

- `internal/c_api/onnxruntime_c_api.h`
- `ort/ortapi_generated.go`
- `ort/cstring.go`
- https://onnxruntime.ai/docs/api/c/struct_ort_api.html
- https://pkg.go.dev/github.com/ebitengine/purego

## How to Run

Deterministic ownership and concurrency proof:

```bash
go test -race -run 'Test(Status|Zero|Concurrent)' ./.planning/spikes/001-ort-status-lifetime
```

Include a real ONNX Runtime status round trip:

```bash
ORT_SPIKE_NATIVE=1 go test -run TestNativeORTStatusRoundTrip ./.planning/spikes/001-ort-status-lifetime
```

The native run resolves the runtime through the project's normal bootstrap cache. It runs separately from `-race` because the race build enables checkptr, which is incompatible with this repository's intentional `uintptr`-based purego FFI boundary.

## What to Expect

- Fake statuses have a release count of exactly one.
- Messages remain unchanged after fake release overwrites the original backing memory.
- The real ORT error code and message survive `ReleaseStatus`.
- `go test -race` reports no data races.

## Investigation Trail

1. Defined the smallest planned typed-error shape: operation, `ort.ErrorCode`, and Go-owned message.
2. Installed `defer ReleaseStatus` before reading status fields so later helper changes cannot introduce an early-return leak.
3. Added a fake store whose release operation overwrites message memory, making an accidental alias observable.
4. Added 256 concurrent independent status conversions with exact release accounting.
5. Added a real C API round trip to validate `purego` signatures and the generated `OrtApi` layout.
6. The first race run exposed checkptr rejecting a Go allocation routed through the production `uintptr` C-string reader. Split deterministic race coverage from the real native ABI proof instead of weakening checkptr globally.

## Results

**Verdict: VALIDATED.**

The deterministic ownership suite passed under the race detector:

```text
ok  github.com/amikos-tech/pure-onnx/.planning/spikes/001-ort-status-lifetime  1.516s
```

The real ONNX Runtime round trip also passed:

```text
--- PASS: TestNativeORTStatusRoundTrip (0.06s)
PASS
```

The Phase 2 implementation can safely preserve native error details if one
central conversion helper owns this sequence:

1. Return `nil` for a zero status.
2. Install `defer ReleaseStatus(status)` immediately.
3. Read the native code.
4. copy the native message into a Go `string`.
5. Return an `ORTError` containing only Go-owned values.

The race suite should keep using instrumented callbacks for exact release
accounting. The real purego ABI test must remain a separate non-race test
because the repository's `uintptr` FFI boundary is intentionally incompatible
with checkptr.
