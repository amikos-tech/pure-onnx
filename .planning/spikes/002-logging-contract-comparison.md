# Spike 002: Diagnostic Logging Contract Comparison

## Outcome

**Recommend 002b: accept a consumer-supplied `slog.Handler`.**

All three variants passed the hard gates:

- nil restores a silent default;
- diagnostics retain structured levels and fields;
- concurrent emission and process-wide reconfiguration pass under `-race`;
- a returned error produces no diagnostic unless an internal call site
  explicitly emits one;
- no third-party logging dependency is added.

## Head-to-Head

| Criterion | 002a: custom sink | 002b: `slog.Handler` | 002c: `*slog.Logger` |
|-----------|-------------------|----------------------|-----------------------|
| Interface boundary | project-owned, 1 method | standard-library, 4 methods | none; concrete pointer |
| Project-owned public logging types | 3 | 0 | 0 |
| Silent default | private no-op | `slog.DiscardHandler` | logger wrapping `slog.DiscardHandler` |
| No-op benchmark | 18.37 ns, 64 B, 1 alloc | 10.51 ns, 0 B, 0 allocs | 10.48 ns, 0 B, 0 allocs |
| Existing slog logger | custom adapter | `logger.Handler()` | `logger` |
| Non-slog backend | one-method adapter | full handler adapter or bridge | full handler adapter/bridge plus `slog.New` |
| Matches requested consumer-wired interface | yes | yes | no |
| Maintenance burden in this project | highest | lowest | low |

The benchmark difference is not the main decision factor because these
diagnostics are rare. The public contract is: 002b keeps the requested
interface boundary while reusing standard levels, fields, concurrency rules,
and the no-op implementation. It is the least project code to own.

002a is preferable only if one-method adapters for non-slog consumers are more
important than maintaining custom public types. Adding an `Enabled` method
could avoid its no-op allocation, but would make the custom interface larger.

002c saves a slog consumer one `Handler()` call at setup time, but replaces the
requested interface with a concrete frontend and offers no benefit to other
logging ecosystems.

## Planning Contract

The production shape should remain close to:

```go
func SetDiagnosticHandler(handler slog.Handler)
```

- `nil` resets to `slog.DiscardHandler`.
- Store a `*slog.Logger` built from the handler in an atomically replaced
  configuration box.
- Keep the emission helper private and use `Logger.LogAttrs`.
- Configure before bootstrap/environment initialization; concurrent
  replacement remains race-safe but is not the primary workflow.
- Use only `slog.LevelInfo` and `slog.LevelWarn` unless an implementation need
  proves another level is necessary.
- Audit each migrated `log.Printf` site. Emit only when the same failure cannot
  be returned to the caller.

A slog consumer wires an existing logger without changing this package:

```go
ort.SetDiagnosticHandler(logger.Handler())
```

A Zap or other consumer owns its chosen `slog.Handler` bridge. This project
does not bundle that dependency.

## Important Limitation

No logging interface can enforce the “never log a returned error” rule. The
API can keep emission private, but correctness ultimately depends on
classifying internal call sites. Planning should include a focused migration
test/audit for finalizers, bootstrap cleanup/fallbacks, lock waits, and runtime
version warnings.
