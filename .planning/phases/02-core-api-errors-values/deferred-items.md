# Deferred Items

## Plan 02-07

- `ort/errors_native_test.go:53` — `make precommit-lint-new` reports an `unsafeptr` warning in the native status test added by Plan 02-01. This file is outside Plan 02-07 ownership.
- `ort/diagnostics_test.go:46` — `make precommit-lint-new` reports staticcheck `SA1012` for the intentional nil-context emitter test added by Plan 02-03. This file is outside Plan 02-07 ownership.
