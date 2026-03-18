# Codebase Concerns

**Analysis Date:** 2026-03-18

## Tech Debt

**Generated ORT API binding pipeline:**
- Issue: `tools/gen_ortapi.go` uses regex parsing of upstream headers instead of a real C parser
- Why: simple generation was fast to bootstrap
- Impact: upstream header format changes could silently break `ort/ortapi_generated.go` generation or field ordering
- Fix approach: replace the parser with a proper C AST approach or add stronger structural validation around generator output

**Stale maintainer guidance in `CLAUDE.md`:**
- Issue: the file still references `main2.go` as the main implementation entry point, but that file does not exist in the current tree
- Why: documentation drift as the repo evolved
- Impact: maintainers or automation may follow outdated guidance
- Fix approach: update `CLAUDE.md` to match the current `examples/`-based layout and active packages

**Partially implemented wrapper types:**
- Issue: `ort/types.go` still contains placeholder `Status.GetErrorCode()` and `Status.GetErrorMessage()` implementations
- Why: the public API surface was sketched before all wrappers were wired to real ORT calls
- Impact: contributors can mistake these types for production-ready abstractions
- Fix approach: either finish the implementation or clearly de-emphasize/remove unused wrapper surfaces

## Known Bugs

**Incorrect usage docs are possible around version support:**
- Symptoms: docs and code can drift between `DefaultOnnxRuntimeVersion`, CI-pinned runtime versions, and README examples
- Trigger: bumping runtime versions in one place but not the others
- Workaround: treat `ort/bootstrap.go`, `.github/workflows/ci.yml`, and `README.md` as a synchronized set during upgrades
- Root cause: multiple version pins live in different files for different purposes

## Security Considerations

**Native binary bootstrap paths must stay hardened:**
- Risk: relaxing checksum, redirect, or path traversal checks in `ort/bootstrap.go` or `embeddings/openclip/bootstrap.go` would expose consumers to malicious binary/model downloads
- Current mitigation: checksum validation, redirect policy enforcement, size limits, path sanitization, and lock-guarded extraction
- Recommendations: preserve these checks during refactors and add tests first when changing bootstrap behavior

**Credential-bearing env vars are handled by runtime/test code:**
- Risk: `GITHUB_TOKEN`, `GH_TOKEN`, `HF_TOKEN`, and release secrets could be leaked through unsafe logging or insecure redirects
- Current mitigation: token use is limited and OpenCLIP bootstrap rejects insecure token-bearing base URLs
- Recommendations: keep logs free of raw env values and treat any new download URL override as security-sensitive

## Performance Bottlenecks

**Session creation remains expensive compared with warm reuse:**
- Problem: embedder packages build LRU caches to avoid repeatedly creating ORT sessions per batch size
- Measurement: the existence of dedicated benchmarks in `ort/session_benchmark_test.go` and `embeddings/splade/benchmark_test.go` indicates session setup cost is material
- Cause: native session creation and tensor wiring are heavier than reuse
- Improvement path: preserve cache hit paths and benchmark any refactor that touches session lifecycle

**Bootstrap and integration flows download large artifacts:**
- Problem: real-model tests and bootstrap code move hundreds of MB of runtime/model data
- Measurement: CI allocates dedicated cache keys and download steps in `.github/workflows/ci.yml`
- Cause: native runtime archives and model bundles are large by nature
- Improvement path: keep cache keys stable, avoid redundant downloads, and be careful with checksum/version churn

## Fragile Areas

**Global ORT lifecycle and lock ordering:**
- Why fragile: `ort/environment.go`, `ort/session.go`, and `ort/tensor.go` coordinate multiple locks and native handles with a strict ordering contract
- Common failures: deadlocks, use-after-destroy bugs, or release calls after environment teardown
- Safe modification: change these paths only with concurrency tests running and keep lock-order comments accurate
- Test coverage: strong relative coverage in `ort/environment_test.go` and `ort/session_test.go`, but still high-risk code

**Embedder tensor/session cache internals:**
- Why fragile: `embeddings/minilm`, `embeddings/splade`, and `embeddings/openclip` reuse backing slices and tensor handles across runs
- Common failures: accidental slice reallocation, stale cache entries, or broken LRU eviction
- Safe modification: preserve buffer ownership assumptions and run integration plus cache-behavior tests after changes
- Test coverage: good package-local tests, but behavior still depends on real model contracts

## Scaling Limits

**Single-process global runtime model:**
- Current capacity: one process-global ORT environment shared across sessions
- Limit: alternative multi-runtime or per-tenant isolation models are not represented in the current architecture
- Symptoms at limit: awkward lifecycle coordination for complex host applications
- Scaling path: introduce a more explicit runtime/handle ownership model only with careful API redesign

## Dependencies at Risk

**Pure FFI dependence on ONNX Runtime ABI stability:**
- Risk: upstream ABI or packaging changes can break symbol registration, archive naming, or runtime expectations
- Impact: bootstrap, initialization, or session creation failures across supported platforms
- Migration plan: keep `internal/c_api/` snapshots, generator output, CI runtime versions, and bootstrap rules aligned when upgrading

## Test Coverage Gaps

**Python tooling is effectively outside the Go test matrix:**
- What's not tested: `tools/openclip_export_onnx.py`, `tools/openclip_generate_golden.py`, and `tools/splade_generate_golden.py`
- Risk: export or dataset-generation regressions may not be caught until maintainers run them manually
- Priority: medium
- Difficulty to test: requires Python environment setup, heavyweight ML dependencies, and artifact generation time

**Release workflow behavior is mostly validated indirectly:**
- What's not tested: end-to-end `.github/workflows/release.yml` behavior against real R2 credentials and signed artifact publishing
- Risk: release-only failures can slip through normal PR CI
- Priority: medium
- Difficulty to test: depends on secrets, tag pushes, and external infrastructure

---

*Concerns audit: 2026-03-18*
*Update as issues are fixed or new ones are discovered*
