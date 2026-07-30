---
phase: quick-260730-ink-address-the-supplied-diagnostics-locking
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - README.md
  - .github/workflows/ci.yml
  - ort/bootstrap.go
  - ort/bootstrap_test.go
  - ort/bootstrap_trust_unix.go
  - ort/bootstrap_trust_other.go
  - ort/diagnostics.go
  - ort/diagnostics_test.go
  - ort/environment.go
  - ort/environment_test.go
  - ort/public_api_compat_test.go
  - ort/session.go
  - ort/session_test.go
  - ort/tensor.go
  - ort/tensor_test.go
autonomous: true
requirements: [RF-01, RF-02, RF-03, RF-04, RF-05, RF-06, RF-07, RF-08, RF-09]
must_haves:
  truths:
    - "Without consumer configuration, every non-returnable warning is written to stderr; installing a slog handler remains race-safe, returned errors are not duplicated, and a panicking finalizer handler falls back to stderr without escaping the finalizer."
    - "If a post-initialization diagnostic handler panics, initialization rolls the environment back; any rollback failure is reported without replacing the original panic value."
    - "A transient cache inspection/read/hash failure preserves the cached install and its original cause, including when downloads are disabled; only a confirmed integrity or trust mismatch is eligible for removal."
    - "Validated root-baked and read-only cache hits require no write or lock-file creation, controlled shared caches have an explicit opt-in, world-writable Unix paths remain rejected, and Windows behavior does not claim Unix ownership/mode enforcement."
    - "DestroyEnvironment cannot clear runtime state while Session.Run is using a session and tensor, concurrent AdvancedSession destruction releases exactly once, and race tests prove the documented lock relationships."
    - "Session construction and run comments identify value-local checks and run leases accurately; runtime.KeepAlive and runtime.Pinner have distinct documented purposes."
    - "The race and native CI selectors execute the renamed lease tests, new lifecycle tests, and real-runtime coverage for every supported tensor element type without disabling checkptr or weakening existing gates."
  artifacts:
    - path: "ort/diagnostics.go"
      provides: "Warning-level stderr default plus panic-safe emergency fallback"
      contains: "SetDiagnosticHandler"
    - path: "ort/bootstrap.go"
      provides: "Cache-hit fast path, explicit validation dispositions, and shared-cache opt-in"
      contains: "validateCachedRuntimeInstall"
    - path: "ort/bootstrap_trust_unix.go"
      provides: "Unix owner/group/world-write trust policy"
      contains: "validateBootstrapPathOwnershipAndMode"
    - path: "ort/bootstrap_trust_other.go"
      provides: "Explicit Windows and residual non-Unix trust behavior without fictitious Unix checks"
      contains: "validateBootstrapPathOwnershipAndMode"
    - path: "ort/session_test.go"
      provides: "Deterministic environment/run/tensor and SessionOptions-borrow ordering plus concurrent double-destroy regressions"
    - path: "ort/tensor_test.go"
      provides: "GC-pressure pinning and real-runtime element-type coverage"
    - path: ".github/workflows/ci.yml"
      provides: "Live-counted race and native selector wiring for the new contracts"
  key_links:
    - from: "ort/diagnostics.go"
      to: "os.Stderr"
      via: "default warning handler and handler-panic fallback"
      pattern: "emitEmergencyDiagnostic"
    - from: "ort/environment.go"
      to: "ort/diagnostics.go"
      via: "rollback cleanup failure is emergency-reported before re-panicking the original value"
      pattern: "emitEmergencyDiagnostic"
    - from: "ort/environment.go"
      to: "ort/environment_test.go"
      via: "AST gate for the complete partial lock-order contract"
      pattern: "TestLifecycleLockHierarchyDocumentation"
    - from: "ort/bootstrap.go"
      to: "ort/bootstrap_test.go"
      via: "typed semantic disposition plus injectable cache-validator seam"
      pattern: "bootstrapValidateCachedRuntimeInstall|cacheValidationDisposition"
    - from: "ort/bootstrap.go"
      to: "ort/bootstrap_test.go"
      via: "validated cache-hit path is proven to perform no lock-file creation or removal"
      pattern: "TestBootstrapReadOnlyCacheHit"
    - from: "ort/session.go"
      to: "ort/session_test.go"
      via: "constructor keeps the SessionOptions read lease through native creation while Destroy waits"
      pattern: "TestNewAdvancedSessionBorrowedOptionsBlocksConcurrentDestroy"
    - from: "ort/session.go"
      to: "ort/session_test.go"
      via: "source gate distinguishes validation from native-use leases"
      pattern: "TestSessionLeaseDocumentation"
    - from: ".github/workflows/ci.yml"
      to: "ort/session_test.go"
      via: "exact race selector names and count"
      pattern: "TestNewAdvancedSessionBorrowedOptionsBlocksConcurrentDestroy"
    - from: ".github/workflows/ci.yml"
      to: "ort/tensor_test.go"
      via: "real-runtime supported-element-type test in native selector"
      pattern: "NATIVE_SELECTOR"
---

<objective>
Correct the supplied diagnostics, lock-order, bootstrap-cache, concurrency, and documentation findings without weakening native-runtime trust or public API compatibility.

Purpose: Restore visible fail-safe warnings, make cache repair destructive only when corruption is confirmed, support legitimate immutable/shared deployment caches, and turn the lifecycle lock contract into deterministic executable coverage.
Output: Three focused implementation slices with unit, race, cross-platform compile, native-selector, and repository verification.
</objective>

<execution_context>
@/Users/tazarov/.codex/get-shit-done/workflows/execute-plan.md
@/Users/tazarov/.codex/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/PROJECT.md
@.planning/quick/260730-gye-address-all-five-review-findings/260730-gye-SUMMARY.md
@README.md
@.github/workflows/ci.yml
@ort/diagnostics.go
@ort/diagnostics_test.go
@ort/environment.go
@ort/environment_test.go
@ort/bootstrap.go
@ort/bootstrap_test.go
@ort/bootstrap_trust_unix.go
@ort/bootstrap_trust_other.go
@ort/session.go
@ort/session_test.go
@ort/memory.go
@ort/memory_test.go
@ort/tensor.go
@ort/tensor_test.go
@ort/types.go
@ort/public_api_compat_test.go

<interfaces>
Current contracts and seams the executor must preserve:

- `SetDiagnosticHandler(handler slog.Handler)` atomically replaces process-wide diagnostic state. General consumer-handler panics propagate synchronously; finalizer-owned emission must never panic.
- Approved diagnostic call sites are non-returnable runtime-version, bootstrap fallback/cleanup, lock-wait, and finalizer notices. Errors returned to callers must not also be logged.
- `completeEnvironmentInitialization(runtimeVersion string, newlyInitialized bool, err error) error` emits a version warning only after successful initialization and rolls back a newly initialized environment if the consumer handler panics.
- The lifecycle lock relationships are partial rather than one flat nesting chain: `AdvancedSession.runMu` precedes `ortCallMu`; `ortCallMu` precedes global `mu` and per-resource locks; `SessionOptions.handleMu` precedes `mu` when both are held; global `mu` is released before tensor or memory-info handle locks are acquired.
- `EnsureOnnxRuntimeSharedLibrary(opts ...BootstrapOption) (string, error)` returns explicit caller paths through `validateExplicitLibraryFile`, and cache paths through `validateCachedRuntimeInstall`.
- Cache installs are published under a process lock, contain `.onnx-purego-manifest.json`, reject symlinks, and compare every recorded path, size, and SHA256 before returning a loadable library.
- `bootstrapRemoveAll` is the existing test seam for destructive cache/staging cleanup. Do not add a filesystem abstraction or dependency.
- `AdvancedSession.run` holds `runMu`, then `ortCallMu.RLock`, snapshots globals under `mu`, and leases unique tensor handles through `acquireUniqueValueLeases` for the entire native call.
- `valuesToHandles` has no production caller; its only callers are three direct tests. `acquireValueLeases` and `handlesFromLeasedValues` are the production helpers those tests should exercise directly.
- `Tensor[T]` retains its data slice, pins its backing storage with `runtime.Pinner` until `Destroy`, and uses `runtime.KeepAlive` only to extend synchronous call-site liveness through raw-pointer native calls.
- The CI race selector is live-counted at 29 tests and the native selector at 4 tests. Any rename/addition must update both selector contents and exact counts in the same change.
</interfaces>

Project constraints:

- Add no dependency and introduce no CGO; all native pointers remain `uintptr`.
- Preserve all existing public entry points and supported Linux/macOS/Windows amd64/arm64 behavior.
- The supplied RF-01 requirement intentionally supersedes the earlier Phase 2 silent-default diagnostic decision: warnings that cannot be returned must now have a stderr safety net, while informational diagnostics remain opt-in and returned errors remain non-duplicated.
- Do not put internal repository or company identifiers into commits, pull requests, summaries, or other artifacts.
- Do not modify ROADMAP.md. No merge is part of this plan; if a later merge is requested, it must be a squash merge.
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Restore fail-safe diagnostics and panic rollback reporting</name>
  <files>ort/diagnostics.go, ort/diagnostics_test.go, ort/environment.go, ort/environment_test.go, README.md</files>
  <behavior>
    - Test 1 (RF-01): With no consumer handler, warning diagnostics are visible on stderr; info-level lock-wait messages remain quiet by default.
    - Test 2 (RF-01): `SetDiagnosticHandler(nil)` restores the stderr warning handler rather than `slog.DiscardHandler`, while an installed handler still receives structured records exactly once under concurrent reconfiguration.
    - Test 3 (suggestion): A panic from a configured handler during finalizer cleanup does not escape and emits a fallback stderr record containing the resource, cleanup error, and handler-panic context.
    - Test 4 (RF-08): If the runtime-version warning panics after successful initialization and `DestroyEnvironment` also fails, the original handler panic is recovered unchanged and the cleanup failure is present on stderr.
    - Test 5: Returned initialization, memory, tensor, session, and bootstrap errors still emit no diagnostic record.
    - Test 6 (RF-02): `TestLifecycleLockHierarchyDocumentation` extracts the `mu` declaration comment from `environment.go` with `go/parser` and requires the exact partial-order, release-before-resource-lock, and no-resource-lock-nesting statements.
  </behavior>
  <action>
Write the stderr and panic-path tests first. Replace the process default and nil-reset `slog.DiscardHandler` with a factory-created standard-library text handler writing to the current `os.Stderr` at warning level. Warning level is deliberate: non-returnable security/runtime/finalizer warnings must be visible per RF-01, while informational lock-wait events remain available only to consumers that install a handler. Keep the atomic immutable diagnostic-state swap and do not route through `slog.Default`, which consumer code can replace globally. Update `SetDiagnosticHandler` documentation to state the new nil/default behavior and retain the rule that ordinary handler panics propagate synchronously.

Add one private `emitEmergencyDiagnostic` stderr writer that bypasses the configured handler. It must be concurrency-safe, best-effort, contain no secrets, and never let a write/formatting panic escape. In `emitFinalizerDiagnostic`, recover a consumer-handler panic and use this emergency path to report the original finalizer cleanup failure plus the recovered handler panic instead of discarding both. Do not retry the configured handler.

In `completeEnvironmentInitialization`, preserve the original panic value exactly. When the post-success runtime-version diagnostic panics, call `DestroyEnvironment`; if rollback returns an error, report that cleanup failure through the emergency stderr path, then re-panic the original recovered value. Do not convert the panic to an error and do not let a cleanup/reporting failure mask it (RF-08). Add a deterministic test using the existing environment close hook, seeded initialized globals, captured stderr, and a panicking handler; assert rollback clears global state, the close cause is reported, and `recover()` receives the original sentinel.

Expand the authoritative comment in `environment.go` for RF-02 and add `TestLifecycleLockHierarchyDocumentation` in `environment_test.go`. Use `go/parser.ParseComments` to locate the documentation associated with the `mu` declaration and require these canonical facts: `AdvancedSession.runMu -> ortCallMu`; `ortCallMu -> SessionOptions.handleMu`, `mu`, `Tensor.runMu`, and `MemoryInfo.handleMu`; `SessionOptions.handleMu -> mu` only when both are held; `mu` is released before `Tensor.runMu` or `MemoryInfo.handleMu`; and `Tensor.runMu` and `MemoryInfo.handleMu` are never nested with each other. This is a partial order, not a claim that every listed lock is simultaneously held. Keep actual nesting consistent with that contract. Update only the diagnostics section of README in this task: warnings go to stderr by default, custom structured handling uses `SetDiagnosticHandler`, nil restores the stderr warning default, and returned errors are never duplicated. Task 2 owns all cache, shared-mode, platform, and symlink README changes so each task documents only the API it implements.
  </action>
  <verify>
    <automated>go test -race -count=1 ./ort -run '^(TestDiagnostic|TestDiagnosticRuntimeVersion|TestEnvironmentErrorChains|TestInitializeEnvironmentDiagnosticHandlerCanQueryRuntime|TestReturnedErrorsDoNotEmit|TestLifecycleLockHierarchyDocumentation)$'</automated>
  </verify>
  <done>Default and nil-reset warning diagnostics reach stderr, info remains opt-in, custom handler concurrency semantics are unchanged, finalizer handler panics fall back safely, rollback cleanup failure is observable without changing the original panic, returned failures remain non-duplicated, and the full lock hierarchy is authoritative.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Separate cache corruption from operational failure and support immutable/shared hits</name>
  <files>ort/bootstrap.go, ort/bootstrap_test.go, ort/bootstrap_trust_unix.go, ort/bootstrap_trust_other.go, README.md</files>
  <behavior>
    - Test 1 (RF-03): `TestEnsureOnnxRuntimeSharedLibraryPreservesCacheOnOperationalValidationError` injects wrapped `os.ErrPermission` and synthetic `syscall.EIO` through the cache-validator seam; both download modes return the original cause, never call `bootstrapRemoveAll`, and leave a sentinel inside the install directory intact.
    - Test 2 (RF-03): A manifest parse/checksum/file-list mismatch, cached symlink, wrong artifact metadata, missing required manifest, or insecure Unix permissions is marked as confirmed invalid/trust state and remains eligible for remove-and-redownload; disabled download never returns the invalid path.
    - Test 3 (RF-05): A fully validated cache hit succeeds after the cache tree is made read-only and with download disabled, without creating/opening a lock path or attempting any removal.
    - Test 4 (RF-05/RF-07): Unix strict mode accepts current-euid and root-owned non-group/world-writable paths, rejects a planted `0777` path, and rejects group-writable paths unless shared-cache trust is explicitly enabled; shared mode still rejects world-writable paths.
    - Test 5 (RF-06): On Windows, the platform-specific function explicitly performs no Unix UID/mode validation; cross-platform cache tests still enforce regular-file/directory checks, symlink rejection, manifest metadata, and hashes.
    - Test 6 (documentation suggestion): Explicit option/env library paths may resolve an ordinary soname symlink to a validated target, while every cache-managed path continues to reject symlinks.
    - Test 7 (RF-05): `TestResolveBootstrapConfigAllowSharedCacheEnvironment`, `TestResolveBootstrapConfigAllowSharedCacheOptionPrecedence`, and `TestResolveBootstrapConfigRejectsInvalidAllowSharedCacheEnvironment` cover true/false env parsing, option-over-env precedence in both directions, and rejection of malformed booleans.
  </behavior>
  <action>
Write regression tests before changing the flow. Introduce a small private `cacheValidationDisposition` (missing, confirmed invalid/trust mismatch, operational) backed by a typed semantic validation error, and mark semantic validation failures at their source. Confirmed invalid includes install/manifest/file symlinks or wrong types, absent/invalid manifest in an existing install, unsupported manifest metadata, checksum/file-count/path/size/digest mismatches, invalid library candidates, and rejected ownership/mode. Ordinary filesystem failures from `Lstat`, `ReadFile`, walking, opening, or hashing—including `EACCES`, `EPERM`, and `EIO`—must retain their underlying cause and remain operational. Do not classify by matching error strings.

Add exactly one narrow validator hook beside `bootstrapRemoveAll`: `bootstrapValidateCachedRuntimeInstall`, initialized to `validateCachedRuntimeInstall` and used by every cache validation call in `EnsureOnnxRuntimeSharedLibrary`. Tests must swap and restore it with `t.Cleanup`; do not introduce a filesystem interface or any other injection layer. Use this seam in the end-to-end `EnsureOnnxRuntimeSharedLibrary` table test named in Test 1, with subcases for wrapped permission/EIO errors and downloads enabled/disabled. Seed an existing install sentinel, count `bootstrapRemoveAll` calls, assert `errors.Is` retains the injected cause, and prove the cache is preserved.

Refactor the `EnsureOnnxRuntimeSharedLibrary` cache branch so a valid existing install is checked and returned before `MkdirAll`, `.locks` creation, or any write. This is the cache-hit path for root-baked images and read-only mounts (RF-05). For a missing cache with downloads disabled, return `ErrSharedLibraryNotFound` without lock creation. For an operational validation error, return it immediately and preserve the cache regardless of the download setting. Call `bootstrapRemoveAll(installDir)` only for the confirmed-invalid disposition; retain joined remove errors and redownload/disabled-download behavior for that disposition (RF-03). After acquiring the process lock for a repair/download, revalidate and apply the same disposition rules so another process can satisfy the cache while this caller waits.

Add `WithBootstrapAllowSharedCache(bool)` and the matching `ONNXRUNTIME_ALLOW_SHARED_CACHE` boolean configuration. Parse the environment with the existing strict boolean parser, apply options after environment defaults so an explicit option wins, and test env true/false, option-over-env precedence in both directions, and invalid text with the three exact test symbols in Test 7. Strict Unix policy accepts paths owned by the effective UID or UID 0 and rejects group- or world-writable paths. Explicit shared-cache mode may accept non-current owners and group-writable paths because the caller is deliberately trusting that group, but it must still reject world-writable `0777` state, symlinks, non-regular files, malformed manifests, and all hash/metadata mismatches. Pass the resolved shared-cache policy explicitly through every `validateBootstrapPathOwnershipAndMode` call covering the cache root, install tree, manifest, files, and lock paths. This opt-in must not affect explicit-library validation.

Keep the existing `bootstrap_trust_other.go` build target and make its Windows versus residual non-Unix behavior explicit in that file, using a small `runtime.GOOS` switch/comment rather than adding a dedicated Windows source file. The Windows branch must clearly document that Go POSIX mode bits and Unix UIDs are not authoritative for Windows ACLs and therefore are not claimed as validated; both Windows and residual non-Unix branches return no Unix ownership/mode failure while platform-neutral cache integrity checks remain mandatory (RF-06). Add one cross-platform test whose Windows branch proves this contract and whose Darwin/Linux branch covers root/current ownership, strict/shared group-write behavior, and the required `0777` negative path (RF-07). Do not invent a Windows ACL checker in this quick task.

Preserve the previous quick task's `validateExplicitLibraryFile`/`validateLibraryFile` separation. Update README in this task with the cache APIs and behavior it implements: validated read-only cache hits require no writes; `WithBootstrapAllowSharedCache`/`ONNXRUNTIME_ALLOW_SHARED_CACHE` deliberately trust a controlled group but never world-writable state; Unix strict ownership/mode checks differ from Windows/non-Unix limits; and caller-selected explicit paths may resolve soname symlinks while cache-managed paths reject them.
  </action>
  <verify>
    <automated>go test -count=1 ./ort -run '^(TestEnsureOnnxRuntimeSharedLibraryDownloadAndCache|TestEnsureOnnxRuntimeSharedLibraryDisableDownload|TestEnsureOnnxRuntimeSharedLibraryReplacesUntrustedCacheEntry|TestEnsureOnnxRuntimeSharedLibraryRedownloadsTamperedManifestFile|TestEnsureOnnxRuntimeSharedLibraryRejectsCachedSymlink|TestEnsureOnnxRuntimeSharedLibraryPreservesCacheOnOperationalValidationError|TestBootstrapCacheValidationDisposition|TestBootstrapReadOnlyCacheHit|TestBootstrapPlatformTrustPolicy|TestResolveBootstrapConfigAllowSharedCacheEnvironment|TestResolveBootstrapConfigAllowSharedCacheOptionPrecedence|TestResolveBootstrapConfigRejectsInvalidAllowSharedCacheEnvironment|TestEnsureOnnxRuntimeSharedLibraryExplicitSymlink|TestValidateLibraryFile)$' &amp;&amp; GOOS=windows GOARCH=amd64 go test -c -o /dev/null ./ort</automated>
  </verify>
  <done>Operational cache failures are non-destructive and inspectable, confirmed corruption alone triggers repair, immutable/root-baked hits require no writable lock directory, shared caches require explicit trust and never accept world-write, Windows claims only the checks it actually performs, and explicit/cache symlink behavior is documented and tested.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: Prove lifecycle ordering and close focused lease, pinning, and API gaps</name>
  <files>ort/session.go, ort/session_test.go, ort/tensor.go, ort/tensor_test.go, ort/public_api_compat_test.go, .github/workflows/ci.yml</files>
  <behavior>
    - Test 1 (RF-04): While a tensor-backed `AdvancedSession.Run` is blocked inside the native callback, `session.runMu.TryLock`, `ortCallMu.TryLock`, and the tensor write `runMu.TryLock` all fail; a concurrent `DestroyEnvironment` cannot return until Run exits, then clears globals exactly once without deadlock.
    - Test 2 (RF-02/RF-04): `TestNewAdvancedSessionBorrowedOptionsBlocksConcurrentDestroy` proves the constructor holds `SessionOptions.handleMu.RLock` through native creation, a concurrent `Destroy` waits, and the borrowed handle is released exactly once only after construction returns.
    - Test 3 (suggestion): Many concurrent `AdvancedSession.Destroy` calls produce no race/error and invoke `releaseSessionFunc` exactly once.
    - Test 4 (RF-09): `TestSessionLeaseDocumentation` source-gates wording that constructor validation is a value-local synchronized check and not a lease, while `acquireUniqueValueLeases` prevents release throughout native `Run`.
    - Test 5 (suggestions): Under repeated GC/allocation pressure, a live tensor retains the same captured native data pointer and contents until Destroy; comments distinguish synchronous `KeepAlive` barriers from lifetime-long `Pinner` pinning.
    - Test 6 (suggestion): With a real configured runtime, float32, float64, int32, and int64 tensor creation/destruction all return non-zero usable native handles.
    - Test 7 (suggestions): External-package code compiles pointer literals for zero-value `SessionOptions` and `MemoryInfo` without copying their embedded locks, and `go vet -copylocks` remains clean.
    - Test 8 (suggestion): Removing production-dead `valuesToHandles` leaves equivalent deduplication, error-unwind, and non-comparable coverage against the production `acquireValueLeases` plus `handlesFromLeasedValues` path.
  </behavior>
  <action>
Add `TestDestroyEnvironmentWaitsForInFlightSessionRun` using the existing native-free callback pattern. Seed one initialized reference without a live library/env handle, block `runSessionFunc` after `runMu`, `ortCallMu.RLock`, and tensor leases are acquired, and use deterministic `TryLock` probes modeled on `TestCreateMemoryInfoBlocksEnvironmentTeardown` rather than timing as the proof. Start `DestroyEnvironment`, use a timeout only as a deadlock/early-return safety net, release the run, and assert event order, successful Run/Destroy, zero refcount, cleared runtime functions, and no tensor/session release during environment teardown. This directly protects RF-04's `AdvancedSession.runMu -> ortCallMu -> mu` lifecycle path and the tensor lease.

Add `TestNewAdvancedSessionBorrowedOptionsBlocksConcurrentDestroy` as a deterministic constructor-order regression. Seed the required ORT callbacks and a valid `SessionOptions` handle; block `createSessionFunc` only after it observes the original options handle. At that point prove `options.handleMu.TryLock()` fails, start `options.Destroy`, and prove neither Destroy nor `releaseSessionOptionsFunc` completes before native creation is unblocked. Then unblock creation, require the constructor to finish with the original borrowed handle, require Destroy to release that handle exactly once afterward, and clean up any returned session. The timeout is only a deadlock/early-return safety net; callback events and `TryLock` are the ordering proof. This test makes the documented `ortCallMu -> SessionOptions.handleMu -> mu` constructor path executable.

Add a separate concurrent double-destroy test with a start barrier and atomic release count. In `AdvancedSession.Destroy`, keep all session-owned handle/name/value reads and clears guarded solely by `runMu`; use `mu` only to snapshot the global release function, then release it before touching session fields. Update the field/lock comments to match this ownership. Correct the RF-09 constructor comment with canonical wording that `validateSessionValue` is a `value-local synchronized check` that `does not lease the handle`; state that `acquireUniqueValueLeases` `prevents handle release during native Run`. Add `TestSessionLeaseDocumentation` in `session_test.go` to read `session.go` and require those exact phrases plus the helper name, so later wording cannot regress to crediting `ortCallMu` with value lifetime. Preserve SessionOptions borrowing under `handleMu.RLock`.

Delete the production-dead `valuesToHandles`. Rename and adapt its three tests to call `acquireValueLeases` and `handlesFromLeasedValues` directly, preserving deduplication, reverse release on later failure, and non-comparable rejection coverage. Do not change the sealed `Value` API or lease semantics.

Clarify in `tensor.go` that `runtime.KeepAlive(data)` covers the synchronous `CreateTensorWithDataAsOrtValue` call, while `runtime.Pinner` prevents the backing array from moving for the entire native OrtValue lifetime and the tensor's `data` field keeps it reachable. Add a GC-pressure test that captures the pointer passed to the fake native constructor, drops the caller's slice reference, repeatedly forces GC plus allocations, and proves pointer identity/content remain valid while the tensor is live; cleanly Destroy and verify one release. Keep this unsafe/pinning test out of the race selector because race-enabled checkptr is intentionally incompatible with raw-pointer lifetime probing.

Add a real-runtime table test for all four production-supported element types. It must use the existing `setupTestEnvironment`, skip only when `ONNXRUNTIME_LIB_PATH` is unavailable, create non-empty tensors, require non-zero handles, and destroy each tensor. Extend `TestExportedHandleStructCompositeLiteralsCompile` with `&amp;ort.SessionOptions{}` and `&amp;ort.MemoryInfo{}` pointer literals so the public shapes remain source-compatible without copying mutex-bearing values.

Update CI selector wiring atomically. Replace the three `TestValuesToHandles...` names with the renamed production-helper tests; add `TestDestroyEnvironmentWaitsForInFlightSessionRun`, `TestNewAdvancedSessionBorrowedOptionsBlocksConcurrentDestroy`, the concurrent AdvancedSession destroy test, and existing `TestMemoryInfoIsValidConcurrentDestroy` to the race selector; and update its exact count from 29 to 33. Add the supported-element-types test to the native selector and change its count from 4 to 5; keep `TestNativeORTStatusRoundTrip` wired unchanged. Do not add the documentation or raw-pointer GC tests to race, do not disable checkptr, and do not add platform-native status jobs where CI supplies no runtime.
  </action>
  <verify>
    <automated>go test -race -count=10 ./ort -run '^(TestAcquireValueLeasesDeduplicatesRepeatedLockableValue|TestAcquireValueLeasesReleasesPriorLeasesOnError|TestAcquireValueLeasesRejectsNonComparableLockable|TestAdvancedSessionRunAndDestroyConcurrent|TestDestroyEnvironmentWaitsForInFlightSessionRun|TestNewAdvancedSessionBorrowedOptionsBlocksConcurrentDestroy|TestAdvancedSessionDestroyConcurrentCallsReleaseOnce|TestMemoryInfoIsValidConcurrentDestroy|TestTensorDestroyWaitsForInFlightRun|TestTensorDestroyConcurrentCallsReleaseOnce)$' &amp;&amp; go test -count=1 ./ort -run '^(TestSessionLeaseDocumentation|TestTensorPinnedBackingSurvivesGC|TestTensorSupportedElementTypesWithORT|TestExportedHandleStructCompositeLiteralsCompile)$' &amp;&amp; go vet -copylocks ./ort/...</automated>
  </verify>
  <done>Environment teardown demonstrably waits for a tensor-backed run, AdvancedSession concurrent destruction releases once, comments match real lock/lease ownership, dead production code is gone, GC/native/public-API gaps are covered, and live-counted CI executes the appropriate tests in race versus native lanes.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| library diagnostics → process stderr / consumer handler | Non-returnable warnings must survive absent or panicking consumer handlers without leaking returned errors twice. |
| filesystem cache → dynamic loader | Cache paths and bytes are untrusted until path type, policy, manifest metadata, and content hashes all pass. |
| shared cache group → validated runtime path | Shared mode deliberately trusts group writers but must retain world-write rejection and all content/path integrity checks. |
| goroutines → ORT global/session/tensor handles | Environment teardown, Run, resource Destroy, and finalizers can otherwise race into stale function pointers or double release. |
| Go heap → native OrtValue | Native code retains a Go slice pointer beyond construction, requiring reachability plus pinning until explicit destruction. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-Q-01 | Repudiation | diagnostics default/finalizers | mitigate | Warning-level stderr default, panic-safe emergency fallback, captured-stderr tests, and continued zero-emission checks for returned errors. |
| T-Q-02 | Tampering | cache validation/removal | mitigate | Typed internal dispositions; only confirmed semantic integrity/trust failures can reach `bootstrapRemoveAll`; EACCES/EIO regressions prove preservation. |
| T-Q-03 | Tampering | shared/read-only cache | mitigate | Validate full manifest and hashes before a no-write hit; require explicit shared trust; reject world-write and symlinks in every mode. |
| T-Q-04 | Spoofing | Windows trust claims | mitigate | Platform-specific implementation/docs/tests state that Unix UID/mode validation is unavailable while retaining platform-neutral checks. |
| T-Q-05 | Denial of Service | environment/session/tensor locking | mitigate | Document the complete partial order and use deterministic TryLock/event-order race tests for teardown, leases, and exact-once release. |
| T-Q-06 | Elevation of Privilege | pinned Go data passed to native code | mitigate | Preserve Pinner lifetime, KeepAlive barriers, explicit Destroy/Unpin, and add GC-pressure plus real-runtime element-type tests. |
| T-Q-SC | Tampering | dependency/tool supply chain | mitigate | Add no package or action; preserve pinned CI actions and existing native/race separation. |
</threat_model>

<verification>
After all three tasks:

1. `gofmt -l` on every modified Go file prints no paths.
2. `go test -count=1 -short ./...` passes.
3. The Task 1 focused diagnostic/race command passes and stderr tests prove warnings are not silently discarded.
4. The Task 2 focused cache/trust command passes on Darwin/Linux; `GOOS=windows GOARCH=amd64 go test -c -o /dev/null ./ort` compiles the explicit Windows branch in the consolidated non-Unix implementation and tests. The repository's Windows CI matrix must run `TestBootstrapPlatformTrustPolicy`.
5. The Task 3 focused race command passes ten consecutive runs.
6. `go test -list` against the committed `RACE_SELECTOR` reports exactly 33 tests, includes `TestNewAdvancedSessionBorrowedOptionsBlocksConcurrentDestroy`, then `go test -race ./ort -run "$RACE_SELECTOR"` passes.
7. `go test -list` against the committed `NATIVE_SELECTOR` reports exactly 5 tests. When `ONNXRUNTIME_LIB_PATH` is set, `go test -v ./ort -run "$NATIVE_SELECTOR"` runs rather than skips all five.
8. `go vet -copylocks ./ort/...` and `go vet -unsafeptr=false ./ort/...` pass.
9. `make precommit-lint-new PRECOMMIT_BASE_REF=main` passes.
10. `git diff -- go.mod go.sum` is empty, and action `uses:` lines in `.github/workflows/ci.yml` are unchanged.
</verification>

<source_audit>

| Source | ID | Feature / Requirement | Task | Status | Notes |
|--------|----|-----------------------|------|--------|-------|
| GOAL | — | Address diagnostics, locking, bootstrap validation/trust, concurrency, and documentation findings atomically | 1-3 | COVERED | One quick-full plan, three cohesive tasks |
| REQ | RF-01 | Default diagnostics must not silently discard non-returnable warnings | 1 | COVERED | Stderr warning default and nil reset |
| REQ | RF-02 | Lock hierarchy docs include SessionOptions.handleMu and MemoryInfo.handleMu and match nesting | 1, 3 | COVERED | Exact AST documentation gate plus deterministic SessionOptions borrow test |
| REQ | RF-03 | Preserve cache on transient validation errors; remove only confirmed integrity/trust failures | 2 | COVERED | Typed dispositions and end-to-end validator-hook EACCES/EIO preservation |
| REQ | RF-04 | Cover DestroyEnvironment racing in-flight Session.Run/tensor use | 3 | COVERED | Actual teardown goroutine plus TryLock/event-order proof |
| REQ | RF-05 | Support root-baked, read-only, and controlled shared cache hits | 2 | COVERED | No-write fast hit plus explicit shared trust |
| REQ | RF-06 | Make Windows ownership/mode behavior explicit, documented, and tested | 2 | COVERED | Consolidated non-Unix implementation with explicit Windows branch and Windows-matrix test |
| REQ | RF-07 | Reject insecure Unix planted permissions such as 0777 | 2 | COVERED | Required strict/shared negative test |
| REQ | RF-08 | Preserve/report rollback cleanup failure without masking original panic | 1 | COVERED | Emergency stderr report then original re-panic |
| REQ | RF-09 | Correct session validation comment to identify leases as the real native-use protection | 3 | COVERED | Exact source documentation gate for constructor check versus Run lease wording |
| REVIEW | S-01 | Finalizer handler panic falls back to stderr | 1 | COVERED | Same emergency path as rollback reporting |
| REVIEW | S-02 | Explain explicit-path symlink resolution versus cache rejection | 2 | COVERED | Public comments and README |
| REVIEW | S-03 | Clarify Tensor KeepAlive versus Pinner | 3 | COVERED | Lifetime-specific comments |
| REVIEW | S-04 | Add GC-pressure pinning invariant | 3 | COVERED | Native-pointer capture under GC pressure |
| REVIEW | S-05 | Add AdvancedSession concurrent double Destroy | 3 | COVERED | Exact-once race regression |
| REVIEW | S-06 | Add real-runtime supported type coverage where practical | 3 | COVERED | All four supported types in existing Linux native lane |
| REVIEW | S-07 | Add copylock-sensitive SessionOptions/MemoryInfo public API coverage | 3 | COVERED | Pointer literals plus copylocks vet |
| REVIEW | S-08 | Extend platform native status wiring only where infrastructure exists | 3 | DEFERRED | Existing `TestNativeORTStatusRoundTrip` remains live in the Linux native lane. macOS/Windows native-status execution is deferred because current CI provides no runtime library in those platform jobs; adding that infrastructure exceeds this quick task. |
| REVIEW | S-09 | Remove production-dead valuesToHandles and adapt tests | 3 | COVERED | Direct production-helper tests and selector rename |
| REVIEW | S-10 | Fix AdvancedSession field-lock comment inconsistency | 3 | COVERED | runMu owns fields; mu only snapshots globals |
| CONTEXT | D-17/D-20 | Previous Phase 2 silent diagnostic default | 1 | SUPERSEDED | The current supplied critical RF-01 explicitly reverses this older decision while preserving opt-in info and no duplicate returned errors. |
| RESEARCH | — | No new dependency or external integration research required | — | EXCLUDED | Level 0: all work follows existing stdlib, hook, cache, and race-test patterns |

</source_audit>

<success_criteria>
- All nine numbered findings have a behavioral regression or exact automated gate.
- Unconfigured/finalizer diagnostics cannot silently lose warnings, yet returned errors are never double-reported.
- Cache operational failures are non-destructive; only marked corruption/trust failures can remove an install.
- Immutable/root-owned cache hits work without writes, shared Unix cache trust is explicit, `0777` is always rejected, and Windows claims no Unix ACL guarantees.
- DestroyEnvironment, AdvancedSession, SessionOptions, MemoryInfo, and Tensor lock relationships are accurately documented and race-tested.
- The dead helper is removed, all four tensor element types run against the existing native fixture, pinning survives GC pressure, and mutex-bearing public structs retain pointer-literal compatibility.
- Full short tests, focused race tests, copylock/unsafeptr vet, Windows compile, live-counted selectors, and new-issues lint pass without dependency or action-pin changes.
- README explains diagnostics defaults, cache trust modes/platform limits, and explicit-versus-cache symlink handling.
- No ROADMAP edit, merge, unrelated cleanup, or internal repository/company reference is introduced.
</success_criteria>

<output>
Create `.planning/quick/260730-ink-address-the-supplied-diagnostics-locking/260730-ink-SUMMARY.md` when done.
</output>
