---
status: complete
phase: 02-core-api-errors-values
source: [02-01-SUMMARY.md, 02-02-SUMMARY.md, 02-03-SUMMARY.md, 02-04-SUMMARY.md, 02-05-SUMMARY.md, 02-06-SUMMARY.md, 02-07-SUMMARY.md, 02-08-SUMMARY.md]
started: 2026-07-29T11:13:39Z
updated: 2026-07-30T08:47:39Z
---

> Superseded by c7e58011: the shipped diagnostics default is a stderr TextHandler at
> LevelWarn, not a silent DiscardHandler.

## Current Test

[testing complete]

## Tests

### 1. Inspectable Error Contracts
expected: Local validation and lifecycle failures match their documented sentinel with errors.Is. Native failures can be extracted with errors.As as *ort.ORTError and expose a stable operation, native code, and copied message. Wrapped lower-level causes remain inspectable.
result: pass

### 2. Value and Tensor Inspection
expected: IsTensor recognizes tensor values, and AsTensor[T] returns the original non-nil *Tensor[T] only for the exact element type. A mismatched type or typed-nil tensor returns nil, false, while heterogeneous tensors can still be stored in []ort.Value.
result: pass

### 3. Opt-in Structured Diagnostics
expected: The library is silent by default. Installing a slog.Handler produces structured notices with standard levels and attributes; passing nil restores silence. Ordinary returned errors are not logged a second time.
result: pass

### 4. Per-call Session Values
expected: AdvancedSession.RunWithValues accepts caller-owned input and output values for one inference, produces the expected model output, and leaves those values usable and owned by the caller afterward. Constructor-bound Run continues to behave as before.
result: pass

### 5. Shape and Tensor Failure Handling
expected: Invalid ParseShape and ShapeElementCount calls match ort.ErrInvalidArgument and retain useful dimension or strconv details. Tensor validation, uninitialized-runtime, and destroyed-value failures match the appropriate public sentinel without losing context.
result: pass

### 6. Environment and MemoryInfo Lifecycle
expected: Missing or invalid environment configuration is classified with the public sentinels while loader, symbol, and cleanup causes remain discoverable. MemoryInfo creation and destruction remain safe when environment teardown happens concurrently.
result: pass

### 7. Bootstrap Safety and Error Categories
expected: A missing library on a supported platform matches ErrSharedLibraryNotFound, while an unsupported host matches ErrUnsupportedPlatform. Extracted files are not group/world writable, executable owner bits survive, and diagnostic URLs have credentials redacted.
result: pass

### 8. Phase-wide Compatibility and Concurrency Gates
expected: All packages and examples still compile, the short suite passes, and the Phase 2 race selector finds exactly 29 tests and passes them. With ONNX Runtime configured, the native selector finds exactly four tests and passes them.
result: pass

## Summary

total: 8
passed: 8
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

[none yet]
