# Deferred Items — Phase 01 DX & Test Hardening

Out-of-scope discoveries logged during execution. Not fixed by the current plan.

| Discovered In | Item | Location | Reason Deferred |
|---------------|------|----------|-----------------|
| 01-01 | `go vet` reports "possible misuse of unsafe.Pointer" | `examples/experimental/main.go:85` | Pre-existing (commit e607906), unrelated to DX-01; file not touched by this plan. Out of scope per executor SCOPE BOUNDARY. |
