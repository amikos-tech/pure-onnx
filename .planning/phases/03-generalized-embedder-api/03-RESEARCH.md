# Phase 3: Generalized Embedder API - Research

<user_constraints>
## User Constraints (from CONTEXT.md)

**Provenance:** The locked decisions, discretion, and deferred scope below are copied verbatim from `03-CONTEXT.md`. [VERIFIED: .planning/phases/03-generalized-embedder-api/03-CONTEXT.md]

### Locked Decisions

#### Shared dense/sparse type model
- **D-01:** Add a generic `embeddings.Embedder[T]` interface. `T` represents one embedding row: MiniLM and OpenCLIP use `[]float32`; SPLADE uses its existing `splade.SparseVector`.
- **D-02:** Keep the type parameter unconstrained (`T any`). The contract provides compile-time result safety without a runtime kind tag, type switch, result wrapper, coercion, or copy.
- **D-03:** A heterogeneous runtime registry is not a Phase 3 requirement. Do not weaken the typed API to make dense and sparse embedders fit in one non-generic collection.

#### Common call surface
- **D-04:** The shared interface contains the existing retrieval-shaped methods `EmbedDocuments([]string) ([]T, error)`, `EmbedQuery(string) (T, error)`, and `Close() error`.
- **D-05:** MiniLM and SPLADE conform through their existing method sets; do not rename or duplicate their public methods.
- **D-06:** Add `EmbedDocuments` and `EmbedQuery` forwarding methods to OpenCLIP's concrete embedder. They delegate to `EmbedTexts` and `EmbedText` respectively, without changing text results or inference behavior.
- **D-07:** Preserve OpenCLIP's existing `EmbedTexts`, `EmbedText`, `EmbedImages`, and `EmbedImage` methods. Image embedding remains a concrete OpenCLIP capability and is not added to the common text contract.

#### Package placement and compatibility
- **D-08:** Create the root Go package `embeddings` as a dependency-light contract package containing the generic interface. It must not become a facade that imports or constructs the model implementations.
- **D-09:** Keep `minilm.NewEmbedder`, `splade.NewEmbedder`, and `openclip.NewEmbedder` returning their existing concrete pointer types. Do not replace constructor results with interfaces.
- **D-10:** Keep all existing result types in their current packages, including `splade.SparseVector` and `splade.SparseEmbedding`. Do not move them, replace them with root-owned types, or add compatibility aliases merely for namespace uniformity.
- **D-11:** The change is additive: no existing exported constructor, method, parameter, return type, field, or behavior may be removed or changed.
- **D-12:** Do not add generalized factories, adapters, runtime-tagged result unions, or extra public capability packages in this phase.

#### Spike-validated constraints
- **D-13:** Spike 003 compiled the exact generic contract against the real model packages. `*minilm.Embedder` and `*openclip.Embedder` satisfy `embeddings.Embedder[[]float32]`; `*splade.Embedder` satisfies `embeddings.Embedder[splade.SparseVector]` without result conversion or copying.
- **D-14:** The overlaid root `embeddings` package has zero imports. Preserve that dependency-free shape so the generalized contract cannot create an import cycle with its model subpackages.
- **D-15:** The negative control proved that OpenCLIP's only conformance gaps are `EmbedDocuments` and `EmbedQuery`; adding the two direct forwarding methods makes it conform while preserving its existing text/image API and validation behavior.
- **D-16:** Carry Spike 003's exact constructor/method signature assertions into Phase 3 verification. Planning must include compile-time conformance checks, `go vet`, the embedding regression suites, and the complete short module suite.

### the agent's Discretion
- Exact interface and package documentation wording.
- Placement and naming of compile-time interface assertions and conformance tests.
- Test-file organization, provided tests prove all three embedders conform at compile time and existing functional/golden behavior remains unchanged.

### Deferred Ideas (OUT OF SCOPE)

None — runtime-selected heterogeneous registries, root factories/facades, runtime-tagged results, shared result-type migration, and additional capability packages were considered and explicitly excluded rather than added to future scope.
</user_constraints>

**Researched:** 2026-08-01  
**Domain:** Additive Go generic interface, implicit method-set conformance, and embedding regression verification. [VERIFIED: 03-CONTEXT.md; Spike 003]  
**Confidence:** HIGH

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| API-01 | A generalized embedder API supports both dense and sparse embeddings, including SPLADE (#49). | The exact generic interface, the two required OpenCLIP forwarders, current concrete signatures, compile-time assertions, and dense/sparse golden gates are mapped below. [VERIFIED: .planning/REQUIREMENTS.md; 03-CONTEXT.md; Spike 003; current embedder tests] |
</phase_requirements>

## Project Constraints (from AGENTS.md)

- Never place private repository information in commit messages, pull requests, or related artifacts; notify the user if a requested change would require it. [VERIFIED: user-provided AGENTS.md instructions]
- Use squash merges. [VERIFIED: user-provided AGENTS.md instructions]
- Prefer the smallest implementation and avoid code that is not needed. [VERIFIED: user-provided AGENTS.md instructions]
- Explain concepts in plain language with small, direct examples. [VERIFIED: user-provided AGENTS.md instructions]

## Summary

Phase 3 is a small public-contract addition, not a new embedding implementation. MiniLM already returns dense `[]float32` rows through `EmbedDocuments`, `EmbedQuery`, and `Close`; SPLADE already returns `splade.SparseVector` rows through the same method names; OpenCLIP already has the equivalent text operations under `EmbedTexts` and `EmbedText`. [VERIFIED: embeddings/minilm/embedder.go; embeddings/splade/embedder.go; embeddings/openclip/embedder.go]

Spike 003 already proved the production shape against the real packages: one zero-import root interface and two direct OpenCLIP forwarding methods. It also proved exact constructor/method compatibility, typed interface dispatch, the absence of a production import cycle, and passage of the embedding and short-module regression suites under the overlay. Planning should promote that proof directly, not add another abstraction layer or feasibility task. [VERIFIED: .planning/spikes/003-generic-embedder-contract-proof/README.md; overlay sources; contract_test.go; local spike rerun]

The only new behavior needing focused unit coverage is OpenCLIP's method-name forwarding. Numerical confidence continues to come from the existing MiniLM integration test, SPLADE golden regression/public parity tests, and OpenCLIP text/image integration and parity tests. Native tests skip when their runtime configuration is absent, so a green local short suite is necessary but not sufficient for the phase gate. [VERIFIED: current embedding test files; local environment probe]

**Primary recommendation:** Implement exactly three files—`embeddings/embedder.go`, `embeddings/openclip/generalized_embedder.go`, and an external-package `embeddings/embedder_test.go` derived from Spike 003—and make no constructor, result-type, factory, or inference-path changes. [RECOMMENDED] [VERIFIED: 03-CONTEXT.md; Spike 003]

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Generic dense/sparse text contract | Root library API (`embeddings`) | Caller generic code | The root owns only the three-method type contract and must have zero production imports. [VERIFIED: D-01, D-04, D-08, D-14] |
| Dense MiniLM execution | Model adapter (`embeddings/minilm`) | `ort` inference core | The current method set already matches `Embedder[[]float32]`; no production edit is needed. [VERIFIED: embeddings/minilm/embedder.go; Spike 003] |
| Sparse SPLADE execution and row type | Model adapter (`embeddings/splade`) | `ort` inference core | The current method set already matches `Embedder[splade.SparseVector]`, and the package continues to own `SparseVector`. [VERIFIED: embeddings/splade/embedder.go; Spike 003] |
| OpenCLIP text conformance | Model adapter (`embeddings/openclip`) | Existing text methods | Two name-compatible forwarders delegate directly to `EmbedTexts` and `EmbedText`; image methods stay concrete-only. [VERIFIED: D-06, D-07; Spike 003 overlay] |
| Resource ownership and inference | Existing concrete embedders | `ort.AdvancedSession` and tokenizer | The interface exposes existing `Close`; it does not alter sessions, tensors, locking, or ownership. [VERIFIED: current embedder sources; D-04, D-11] |
| Conformance and numerical acceptance | Go test packages | Existing native integration environment | Compile-time pins protect API shape; existing real-model and golden tests protect behavior. [VERIFIED: Spike 003 contract_test.go; current embedding tests] |

## Current API and Exact File Map

### Existing signatures that must remain unchanged

```go
// MiniLM
func minilm.NewEmbedder(string, string, ...minilm.Option) (*minilm.Embedder, error)
func (*minilm.Embedder).EmbedDocuments([]string) ([][]float32, error)
func (*minilm.Embedder).EmbedQuery(string) ([]float32, error)
func (*minilm.Embedder).Close() error

// SPLADE
func splade.NewEmbedder(string, string, ...splade.Option) (*splade.Embedder, error)
func (*splade.Embedder).EmbedDocuments([]string) ([]splade.SparseVector, error)
func (*splade.Embedder).EmbedQuery(string) (splade.SparseVector, error)
func (*splade.Embedder).Close() error

// OpenCLIP
func openclip.NewEmbedder(string, string, string, string, ...openclip.Option) (*openclip.Embedder, error)
func (*openclip.Embedder).EmbedTexts([]string) ([][]float32, error)
func (*openclip.Embedder).EmbedText(string) ([]float32, error)
func (*openclip.Embedder).EmbedImages([]image.Image) ([][]float32, error)
func (*openclip.Embedder).EmbedImage(image.Image) ([]float32, error)
func (*openclip.Embedder).Close() error
```

These signatures were read from the current source and are already pinned by Spike 003's function-type assignments. [VERIFIED: current embedder sources; Spike 003 contract_test.go]

### Planned files

| File | Action | Exact responsibility |
|------|--------|----------------------|
| `embeddings/embedder.go` | Add | Package documentation plus the dependency-free `Embedder[T any]` interface only. [RECOMMENDED] [VERIFIED: Spike 003 overlay/embeddings/embedder.go] |
| `embeddings/openclip/generalized_embedder.go` | Add | Two one-line forwarders: `EmbedDocuments` → `EmbedTexts`, `EmbedQuery` → `EmbedText`. [RECOMMENDED] [VERIFIED: Spike 003 overlay/embeddings/openclip/generalized_embedder.go] |
| `embeddings/embedder_test.go` | Add using `package embeddings_test` | Exact constructor/existing-method pins, all three generic conformance assertions, typed dispatch checks, and direct-vs-forwarded OpenCLIP validation comparison. [RECOMMENDED] [VERIFIED: Spike 003 contract_test.go] |
| `embeddings/minilm/embedder.go` | No edit | Its current pointer method set already conforms. [VERIFIED: source; Spike 003] |
| `embeddings/splade/embedder.go` | No edit | Its current pointer method set and result types already conform. [VERIFIED: source; Spike 003] |
| Existing golden/integration tests | Reuse | They remain the numerical and behavior gates; no fixture or dataset rewrite is needed. [VERIFIED: named test files in 03-CONTEXT.md] |

Using an external test package is important: tests may import the three child packages while the production root package remains import-free. [RECOMMENDED] [VERIFIED: Spike 003 package graph]

## Minimal Task Decomposition

One plan with three sequential tasks is sufficient; splitting this tiny change across plans would add coordination without isolating meaningful risk. [RECOMMENDED] [VERIFIED: three-file delta proven by Spike 003]

| Task | Files | Work | Task-level verification |
|------|-------|------|-------------------------|
| 1. Promote the production contract | `embeddings/embedder.go`, `embeddings/openclip/generalized_embedder.go` | Copy the validated shapes, use normal GoDoc, and make no other production edits. [RECOMMENDED] | `go test -run '^$' ./embeddings/...` and `go list -f '{{join .Imports " "}}' ./embeddings` (expect blank). [RECOMMENDED] |
| 2. Promote compatibility proof | `embeddings/embedder_test.go` | Adapt Spike 003's test to `package embeddings_test`; retain every existing signature pin and all three conformance assertions; pin the two new forwarding signatures too. [RECOMMENDED] | `go test -count=1 ./embeddings` [RECOMMENDED] |
| 3. Run acceptance gates | No production files | Run format, vet, embedding regression, full short-module, and native golden/integration commands. [RECOMMENDED] | Commands are listed in Validation Architecture. [VERIFIED: D-16; existing tests] |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Go language and standard toolchain | Module baseline `go 1.25.0` | Generic interface, implicit method-set conformance, formatting, vet, and tests | The repository already requires this baseline; the contract needs no external library. [VERIFIED: go.mod; Spike 003] |
| Root `embeddings` package | New internal package in this module | Public three-method generic contract | The overlay proves it compiles with no imports and does not construct implementations. [VERIFIED: Spike 003 overlay and `go list -json` rerun] |
| Existing model packages | Current repository code | Dense MiniLM, sparse SPLADE, and dense OpenCLIP implementations | They already own model configuration, inference, output processing, and resource cleanup. [VERIFIED: current embedder sources] |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Go `testing` | Module baseline | Compile-time signature/conformance proof and forwarding behavior | Use the existing test framework; no new test dependency or configuration is required. [VERIFIED: current tests; go.mod] |
| Spike 003 overlay and contract test | Committed project evidence | Production blueprint and compatibility assertion source | Copy the proven shapes into production/tests; do not keep the overlay as the Phase 3 implementation. [VERIFIED: Spike 003] |
| Existing golden and integration suites | Current repository tests | Numerical and lifecycle regression coverage | Run unchanged in an environment configured with ONNX Runtime and model/golden assets. [VERIFIED: current embedding tests] |

### Alternatives Considered

These are recorded only because the locked context explicitly rejects them; they are not planning options. [VERIFIED: D-02, D-03, D-08 through D-12]

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `Embedder[T any]` | Non-generic result union or `any` | Loses compile-time row typing and adds tags, switches, wrappers, or conversions forbidden by the phase. [VERIFIED: D-01 through D-03] |
| Implicit MiniLM/SPLADE conformance | Adapter wrappers | Adds code and potentially changes identity/ownership without solving a real mismatch. [VERIFIED: D-05, D-13] |
| Direct OpenCLIP forwarders | Duplicate text inference implementation | Risks validation and numerical drift; delegation is already proven. [VERIFIED: D-06, D-15] |
| Concrete constructors | Constructors returning the interface | Hides OpenCLIP image methods and breaks exact exported signatures. [VERIFIED: D-07, D-09, D-11] |
| Root contract leaf | Root factory/facade importing models | Violates the zero-import package boundary and creates avoidable coupling. [VERIFIED: D-08, D-12, D-14] |

**Installation:**

```bash
# No package installation and no go.mod/go.sum change.
```

Phase 3 adds no external package. [VERIFIED: locked scope; recommended implementation]

## Package Legitimacy Audit

Not applicable: no package is installed, upgraded, or added to `go.mod`; the package-legitimacy gate is not triggered. [VERIFIED: Standard Stack; D-12]

## Architecture Patterns

### System Architecture Diagram

```text
Caller constructs an existing concrete embedder
        │
        ├─ *minilm.Embedder ───────┐
        ├─ *splade.Embedder ───────┼─ implicit assignment to embeddings.Embedder[T]
        └─ *openclip.Embedder ─────┘
                                        │
                         EmbedDocuments / EmbedQuery / Close
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             ▼                          ▼                          ▼
      MiniLM existing path       SPLADE existing path       OpenCLIP forwarders
       T = []float32             T = SparseVector         Documents → EmbedTexts
             │                          │                  Query → EmbedText
             │                          │                          │
             └──────── existing tokenizer + AdvancedSession.Run ─┘
                                        │
                                        ▼
                          existing typed result; no conversion

OpenCLIP image entry points stay on *openclip.Embedder and do not enter the contract.
External model files/tokenizer/ONNX Runtime remain outside the new package boundary.
```

The only decision is made at compile time by the chosen `T`; there is no runtime dense/sparse branch in the generalized contract. [VERIFIED: D-01 through D-03; Spike 003]

### Recommended Project Structure

```text
embeddings/
├── embedder.go                    # new: zero-import generic contract
├── embedder_test.go               # new: external-package compatibility proof
├── internal/                      # unchanged
├── minilm/                        # unchanged production implementation/tests
├── splade/                        # unchanged production implementation/tests
└── openclip/
    ├── embedder.go                # unchanged existing text/image behavior
    └── generalized_embedder.go    # new: two direct forwarding methods
```

This layout is the smallest production form validated by Spike 003. [RECOMMENDED] [VERIFIED: Spike 003 overlay]

### Pattern 1: Dependency-free contract leaf

**What:** Define only a generic interface in the root package; do not import child model packages. [VERIFIED: D-08, D-14]

**When to use:** Callers use an instantiated interface when an algorithm needs document/query embedding and cleanup but does not need model-specific options or OpenCLIP image methods. [VERIFIED: D-04, D-07]

```go
// Source: Spike 003 overlay/embeddings/embedder.go
type Embedder[T any] interface {
	EmbedDocuments(documents []string) ([]T, error)
	EmbedQuery(query string) (T, error)
	Close() error
}
```

### Pattern 2: Implicit conformance with compile-time assertions

**What:** Let existing pointer method sets satisfy the instantiated interfaces, and make mismatches compile failures. [VERIFIED: current sources; Spike 003]

```go
// Source: Spike 003 contract_test.go
var (
	_ embeddings.Embedder[[]float32]           = (*minilm.Embedder)(nil)
	_ embeddings.Embedder[splade.SparseVector] = (*splade.Embedder)(nil)
	_ embeddings.Embedder[[]float32]           = (*openclip.Embedder)(nil)
)
```

### Pattern 3: Additive name forwarding

**What:** Add names required by the common retrieval contract while routing through OpenCLIP's existing text path. [VERIFIED: D-06; Spike 003]

```go
// Source: Spike 003 overlay/embeddings/openclip/generalized_embedder.go
func (e *Embedder) EmbedDocuments(documents []string) ([][]float32, error) {
	return e.EmbedTexts(documents)
}

func (e *Embedder) EmbedQuery(query string) ([]float32, error) {
	return e.EmbedText(query)
}
```

### Anti-Patterns to Avoid

- **Root facade imports model packages:** breaks D-08/D-14 and turns a leaf contract into a dependency hub. [VERIFIED: 03-CONTEXT.md]
- **Result covariance assumptions:** Go conformance requires the instantiated method signatures to match exactly; do not substitute wrappers, aliases, or `any`. [VERIFIED: Spike 003 positive and negative controls]
- **Recursive OpenCLIP aliases:** `EmbedDocuments` must call `EmbedTexts`, and `EmbedQuery` must call `EmbedText`; never route both names back through each other. [RECOMMENDED] [VERIFIED: Spike 003 overlay]
- **Interface-returning constructors:** removes concrete-only capabilities from callers and changes public signatures. [VERIFIED: D-07, D-09, D-11]
- **Production-only conformance imports:** keep cross-package assertions in `embeddings_test` so model dependencies exist only in tests. [RECOMMENDED] [VERIFIED: D-14]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Dense/sparse result abstraction | Tagged union, kind enum, reflection, or `interface{}` payload | `Embedder[T any]` | The type argument already identifies one row at compile time. [VERIFIED: D-01 through D-03; Spike 003] |
| MiniLM/SPLADE adaptation | Wrapper structs | Their existing method sets | Both already conform without conversion. [VERIFIED: D-05, D-13] |
| OpenCLIP generalized text inference | Second tokenizer/session/post-processing path | Two direct forwarders | Reusing existing methods preserves validation, locking, caching, and results. [VERIFIED: D-06, D-15; openclip/embedder.go] |
| Model selection | Root factory or heterogeneous registry | Existing concrete constructors | Runtime selection is outside scope and concrete results preserve model-specific methods. [VERIFIED: D-03, D-09, D-12] |
| Sparse representation | Root-owned sparse type or conversion layer | `splade.SparseVector` and existing alias | Moving or copying results would break package ownership and compatibility. [VERIFIED: D-10; splade/embedder.go] |
| Compatibility test harness | New API-diff tooling | Spike 003's exact function assignments and interface assertions | The committed proof already covers every relevant constructor and method signature. [VERIFIED: D-16; Spike 003 contract_test.go] |

**Key insight:** The generalized API is a shared set of method names, not a shared runtime representation. [VERIFIED: D-01 through D-04]

## Common Pitfalls

### Pitfall 1: Turning the root package into a facade

**What goes wrong:** The root imports model packages or exposes factories, growing coupling and violating the proven leaf boundary. [VERIFIED: D-08, D-12, D-14]  
**Why it happens:** “Unified API” is mistaken for “central construction and runtime dispatch.” [VERIFIED: excluded designs in 03-CONTEXT.md]  
**How to avoid:** Keep `embeddings/embedder.go` import-free and limited to the interface and its documentation. [RECOMMENDED]  
**Warning signs:** Imports in the production root file, `NewEmbedder` in the root, model-name switches, or model-specific types in the contract. [RECOMMENDED]

### Pitfall 2: Weakening types to mix dense and sparse values

**What goes wrong:** Results need type assertions, tags, or copying, and API-01 loses compile-time row safety. [VERIFIED: D-02, D-03]  
**Why it happens:** Distinct instantiations are incorrectly expected to fit one non-generic collection. [VERIFIED: D-03]  
**How to avoid:** Use `Embedder[[]float32]` and `Embedder[splade.SparseVector]` as separate typed interfaces. [RECOMMENDED]  
**Warning signs:** `any`, `map[string]any`, a `Kind` field, or a dense/sparse switch. [RECOMMENDED]

### Pitfall 3: Reimplementing OpenCLIP behavior

**What goes wrong:** Generalized calls drift from `EmbedTexts`/`EmbedText` in validation, caching, errors, or numerical output. [VERIFIED: openclip/embedder.go; D-06]  
**Why it happens:** The two new names look like new inference entry points rather than aliases. [VERIFIED: D-15]  
**How to avoid:** Each method must be a single direct return of the existing method call; compare direct and forwarded validation errors without pinning literal text. [RECOMMENDED] [VERIFIED: Spike 003 contract test]  
**Warning signs:** New locks, tokenization, session calls, error wrapping, or result allocation in the forwarder file. [RECOMMENDED]

### Pitfall 4: Breaking existing public signatures while introducing the interface

**What goes wrong:** Concrete callers stop compiling or lose OpenCLIP image operations. [VERIFIED: D-07, D-09, D-11]  
**Why it happens:** Constructors are changed to return the new interface, or existing methods are renamed for uniformity. [VERIFIED: excluded changes in 03-CONTEXT.md]  
**How to avoid:** Retain the complete function-type assignment block from Spike 003 and add pins for the two new methods. [RECOMMENDED]  
**Warning signs:** Any diff in existing constructor signatures, deleted methods, or aliases replacing existing result types. [RECOMMENDED]

### Pitfall 5: Mistaking skipped native tests for parity evidence

**What goes wrong:** The short suite is green while MiniLM, SPLADE, and OpenCLIP native inference and hosted parity did not run. [VERIFIED: current test setup helpers; local environment probe]  
**Why it happens:** The suites call `t.Skip` when `ONNXRUNTIME_LIB_PATH` or golden configuration is missing. [VERIFIED: named integration/golden test files]  
**How to avoid:** Record native command output from the configured integration environment and confirm the named tests report `PASS`, not `SKIP`. [RECOMMENDED]  
**Warning signs:** Sub-second package runs, skip messages, or missing named test lines in acceptance evidence. [RECOMMENDED]

### Pitfall 6: Using the local Go 1.26 toolchain as the compatibility baseline

**What goes wrong:** New code may compile locally but require an API newer than the module's Go 1.25 declaration. [VERIFIED: local `go version`; go.mod]  
**Why it happens:** The installed toolchain is newer than the declared module baseline. [VERIFIED: environment probe]  
**How to avoid:** Keep the implementation to the spike-proven syntax and standard built-ins, then rely on the project's Go 1.25 verification environment. [RECOMMENDED]  
**Warning signs:** Any new API beyond the interface syntax already compiled by Spike 003. [RECOMMENDED]

## Code Examples

### One generic consumer, two result types

```go
// Pattern derived from the Spike 003 contract.
func embedQuery[T any](e embeddings.Embedder[T], text string) (T, error) {
	return e.EmbedQuery(text)
}

var dense embeddings.Embedder[[]float32] = miniLMEmbedder
var sparse embeddings.Embedder[splade.SparseVector] = spladeEmbedder

denseRow, denseErr := embedQuery(dense, "query")
sparseRow, sparseErr := embedQuery(sparse, "query")
```

The helper has one call shape while each result remains statically typed; it does not create a common runtime container. [RECOMMENDED] [VERIFIED: D-01 through D-03; Spike 003]

### Exact forwarding-behavior check

```go
// Source: Spike 003 contract_test.go
var e *openclip.Embedder

_, directErr := e.EmbedText("query")
_, forwardedErr := e.EmbedQuery("query")
if directErr == nil || forwardedErr == nil || directErr.Error() != forwardedErr.Error() {
	t.Fatal("EmbedQuery did not preserve EmbedText validation")
}
```

This compares the two paths in the same run; it does not make a literal error message part of the API contract. [VERIFIED: Spike 003 contract_test.go]

## State of the Art

| Old Approach | Current Phase 3 Approach | When Changed | Impact |
|--------------|--------------------------|--------------|--------|
| Separate concrete APIs with no root contract | Instantiated `Embedder[T]` family over the existing method sets | Phase 3 | Generic callers gain one compile-time API shape without changing results or inference. [VERIFIED: phase goal; Spike 003] |
| OpenCLIP text names only (`EmbedTexts`/`EmbedText`) | Preserve those names and add retrieval-shaped forwarding names | Phase 3 | OpenCLIP conforms while its text/image API remains intact. [VERIFIED: D-06, D-07] |
| Proposed contract proven only through a build overlay | Promote the same two overlay sources to ordinary package files | Phase 3 implementation | The feasibility proof becomes the supported public API with the same dependency direction. [VERIFIED: Spike 003] |

**Deprecated/outdated for this phase:**

- Re-running a generic-interface feasibility spike is unnecessary; Spike 003 has a validated positive proof and negative control. [VERIFIED: 03-CONTEXT.md; Spike 003]
- Runtime unions, root factories, adapters for MiniLM/SPLADE, and result-type migration are explicitly excluded. [VERIFIED: D-03, D-05, D-10, D-12]
- A benchmark for interface dispatch is not an acceptance requirement; concrete call paths remain available and the spike makes no performance claim. [VERIFIED: Spike 003 limits; D-09]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| — | None. Project claims are locked in `03-CONTEXT.md`, read from current source/tests, or reproduced from committed Spike 003 evidence. | — | — |

## Open Questions (RESOLVED)

1. **RESOLVED — Put cross-package assertions in `embeddings/embedder_test.go` with `package embeddings_test`.**
   - What we know: The test must import the root contract and all three child packages while the production root must keep zero imports. [VERIFIED: D-14; Spike 003]
   - Resolution: Use the external test package; this keeps test-only dependencies out of the production package graph. [RECOMMENDED]

2. **RESOLVED — Keep OpenCLIP forwarders in a small new file.**
   - What we know: The overlay already validates a standalone `generalized_embedder.go`, and no existing inference code needs editing. [VERIFIED: Spike 003 overlay]
   - Resolution: Promote that file shape directly so review can see the full behavior at a glance. [RECOMMENDED]

3. **RESOLVED — Do not edit integration fixtures or workflow selection.**
   - What we know: The named MiniLM, SPLADE, and OpenCLIP integration/parity tests already exercise the unchanged implementations, and the complete short suite already discovers `./embeddings/...`. [VERIFIED: current tests; Spike 003 commands]
   - Resolution: Reuse the existing lanes and require named native test evidence; add only the root conformance test. [RECOMMENDED]

No unresolved design question remains for planning. [VERIFIED: 03-CONTEXT.md; resolutions above]

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| Go toolchain | Build, test, vet | ✓ | Local 1.26.5; module baseline 1.25.0 | Keep code baseline-safe and verify in the project's Go 1.25 environment. [VERIFIED: environment probe; go.mod] |
| `gofmt` / `go vet` | Format and static checks | ✓ | Bundled with local Go | None needed. [VERIFIED: environment probe] |
| ONNX Runtime shared library | Native MiniLM/SPLADE/OpenCLIP tests | ✗ in current shell | `ONNXRUNTIME_LIB_PATH` unset | Run acceptance in the existing configured integration environment; local unit/short suites remain available. [VERIFIED: environment probe; test helpers] |
| Model and golden dataset configuration | Numerical parity | ✗ in current shell | Relevant model/golden variables unset | Existing helpers/configured integration environment supply the assets; require non-skipped named test evidence. [VERIFIED: environment probe; current tests] |

**Missing dependencies with no fallback:**

- None for implementation or compile/unit verification. [VERIFIED: local short-suite and Spike 003 reruns]

**Missing dependencies with fallback:**

- Native runtime/model/golden configuration is absent locally; use the existing configured integration environment for the phase acceptance gate. [VERIFIED: environment probe; current tests]

## Validation Architecture

`workflow.nyquist_validation` is enabled, so Phase 3 needs compile-time contract coverage, local forwarding behavior, unchanged package regressions, and configured native parity evidence. [VERIFIED: .planning/config.json; D-16]

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Go `testing`, module baseline Go 1.25.0. [VERIFIED: go.mod; existing tests] |
| Config file | None; package tests and command-line selectors are already used. [VERIFIED: current test layout; Spike 003] |
| Quick run command | `go test -count=1 -short ./embeddings/...` [VERIFIED: local baseline run; Spike 003] |
| Full suite command | `go test -count=1 -short ./...` [VERIFIED: local baseline run; D-16] |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| API-01 | MiniLM, SPLADE, and OpenCLIP satisfy the correct instantiated interface; all existing public signatures remain exact | compile/unit | `go test -count=1 ./embeddings` | ❌ Wave 0: add `embeddings/embedder_test.go` from Spike 003 |
| API-01 | Typed interface dispatch reaches existing validation and OpenCLIP forwarding preserves direct validation | unit | `go test -count=1 ./embeddings -run '^(TestTypedInterfaceDispatchReachesExistingValidation|TestOpenCLIPForwardersPreserveExistingValidation)$'` | ❌ Wave 0: same test file |
| API-01 | Root production package remains import-free and the package graph has no cycle | static/build | `test -z "$(go list -f '{{join .Imports " "}}' ./embeddings)" && go test -run '^$' ./embeddings/...` | ❌ Root package appears only after Wave 0 implementation; command is ready |
| API-01 | MiniLM dense document/query results and session reuse remain unchanged | native integration | `go test -count=1 -v ./embeddings/minilm -run '^TestEmbedDocumentsWithAllMiniLML6V2$'` | ✅ `embeddings/minilm/embedder_integration_test.go` |
| API-01 | SPLADE produces sparse vectors matching local golden rows and repeatability | native golden regression | `go test -count=1 -v ./embeddings/splade -run '^(TestEmbedDocumentsWithSPLADEModel|TestSPLADEGoldenRegressionTopK16WithLabels|TestSPLADERepeatabilityTopK16)$'` | ✅ existing SPLADE integration/regression tests |
| API-01 | SPLADE matches the hosted golden parity dataset | native hosted parity | `go test -count=1 -v ./embeddings/splade -run '^TestSPLADEGoldenDatasetParity$'` | ✅ `embeddings/splade/golden_dataset_parity_test.go` |
| API-01 | Existing OpenCLIP text/image behavior remains unchanged | native integration | `go test -count=1 -v ./embeddings/openclip -run '^(TestEmbedTextsAndImagesWithOpenCLIPModel|TestOpenCLIPFailsWithWrongInputOutputNames|TestOpenCLIPFailsWithWrongEmbeddingDimension|TestOpenCLIPFailsWithImageSizeMismatch|TestOpenCLIPErrorsAfterClose|TestOpenCLIPCloseIsIdempotent)$'` | ✅ `embeddings/openclip/embedder_integration_test.go` |
| API-01 | Existing OpenCLIP text/image golden parity remains unchanged | native hosted parity | `go test -count=1 -v ./embeddings/openclip -run '^TestOpenCLIPGoldenDatasetParity$'` | ✅ `embeddings/openclip/golden_dataset_parity_test.go` |
| API-01 | All embedding packages vet and the full module remains compatible | static/regression | `go vet ./embeddings/... && go test -count=1 -short ./...` | ✅ existing toolchain/tests; new root files included automatically after implementation |

### Exact Compile-Time Assertions to Preserve

```go
var (
	_ embeddings.Embedder[[]float32]           = (*minilm.Embedder)(nil)
	_ embeddings.Embedder[splade.SparseVector] = (*splade.Embedder)(nil)
	_ embeddings.Embedder[[]float32]           = (*openclip.Embedder)(nil)

	_ func(string, string, ...minilm.Option) (*minilm.Embedder, error) = minilm.NewEmbedder
	_ func(string, string, ...splade.Option) (*splade.Embedder, error) = splade.NewEmbedder
	_ func(string, string, string, string, ...openclip.Option) (*openclip.Embedder, error) = openclip.NewEmbedder

	_ func(*openclip.Embedder, []string) ([][]float32, error) = (*openclip.Embedder).EmbedDocuments
	_ func(*openclip.Embedder, string) ([]float32, error) = (*openclip.Embedder).EmbedQuery
)
```

Retain Spike 003's pins for every existing MiniLM, SPLADE, and OpenCLIP method in addition to the abbreviated block shown here. [RECOMMENDED] [VERIFIED: D-16; Spike 003 contract_test.go]

### Sampling Rate

- **Per task commit:** `go test -count=1 ./embeddings` after the root package exists. [RECOMMENDED]
- **Per wave merge:** `go test -count=1 -short ./embeddings/... && go vet ./embeddings/...` [RECOMMENDED] [VERIFIED: D-16]
- **Phase gate:** Root dependency check, full short module suite, and every named native MiniLM/SPLADE/OpenCLIP integration/parity command above must pass; native evidence must show the tests ran rather than skipped. [RECOMMENDED] [VERIFIED: D-16; test skip behavior]

### Wave 0 Gaps

- [ ] `embeddings/embedder_test.go` — external-package exact-signature, generic-conformance, typed-dispatch, and OpenCLIP-forwarding proof for API-01. [RECOMMENDED] [VERIFIED: Spike 003 contract_test.go]
- [ ] No test framework installation or configuration is needed. [VERIFIED: existing Go test infrastructure]
- [ ] No new native fixture, golden dataset, or workflow selector is needed; reuse the named existing suites. [RECOMMENDED] [VERIFIED: current tests; D-16]

### Baseline Evidence Collected During Research

- `go test -count=1 -short ./embeddings/...` passed for MiniLM, SPLADE, and OpenCLIP. [VERIFIED: local command, 2026-08-01]
- `go vet ./embeddings/...` passed. [VERIFIED: local command, 2026-08-01]
- `go test -count=1 -short ./...` passed across the module. [VERIFIED: local command, 2026-08-01]
- Spike 003's focused positive overlay test passed again; its root package reported no imports. [VERIFIED: local spike rerun, 2026-08-01]
- Native/golden tests were not executed locally because the required environment variables are unset. [VERIFIED: environment probe; test helpers]

## Security Domain

This phase adds an in-process compile-time contract and two method aliases; it adds no network endpoint, authentication/session system, file loader, secret, or cryptographic operation. Existing model/bootstrap security boundaries remain unchanged. [VERIFIED: phase scope; planned files]

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No identity or authentication surface is added. [VERIFIED: phase scope] |
| V3 Session Management | no | ONNX inference sessions are resource objects, not authenticated user sessions. [VERIFIED: current embedder sources] |
| V4 Access Control | no | No principal, permission, or authorization boundary is added. [VERIFIED: phase scope] |
| V5 Input Validation | yes, preserved | Forwarders must delegate to existing text methods so nil/closed/runtime and text validation cannot be bypassed. [VERIFIED: openclip/embedder.go; Spike 003 forwarding test] |
| V6 Cryptography | no new control | No cryptographic behavior changes; existing artifact verification remains outside the Phase 3 delta. [VERIFIED: planned files; D-11] |

### Known Threat Patterns for the Generalized Go API

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Forwarding method bypasses existing receiver/state validation | Tampering / Denial of Service | Directly return `EmbedTexts`/`EmbedText`; compare direct and forwarded validation behavior. [VERIFIED: D-06, D-15; Spike 003] |
| Runtime-tagged result accepts the wrong dense/sparse shape | Tampering | Keep `T` typed and unconstrained; add compile-time assertions for both row types. [VERIFIED: D-01 through D-03] |
| Generalized API loses cleanup capability and leaks native/tokenizer resources | Denial of Service | Keep `Close() error` in the interface and preserve concrete ownership behavior. [VERIFIED: D-04; current embedder sources] |
| Root facade introduces model initialization or dependency side effects | Denial of Service / Supply-chain expansion | Keep the production root package at zero imports and expose no factory. [VERIFIED: D-08, D-12, D-14] |
| API migration silently removes model-specific capabilities | Denial of Service | Preserve concrete constructor returns and all existing methods; compile-pin exact signatures. [VERIFIED: D-07, D-09, D-11, D-16] |

## Sources

### Primary (HIGH confidence)

- `.planning/phases/03-generalized-embedder-api/03-CONTEXT.md` — locked scope, exact API decisions, Spike 003 conclusions, and verification requirements. [VERIFIED: repository]
- `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md`, `.planning/PROJECT.md`, and `.planning/STATE.md` — API-01 traceability, phase goal, project constraints, and dependency on the settled Phase 2 core. [VERIFIED: repository]
- `.planning/spikes/003-generic-embedder-contract-proof/README.md`, overlay sources, `contract_test.go`, and `overlay.json` — validated production blueprint, negative control, signature pins, and test commands. [VERIFIED: repository; local rerun]
- `embeddings/minilm/embedder.go`, `embeddings/splade/embedder.go`, and `embeddings/openclip/embedder.go` — current constructors, method sets, result types, forwarding targets, and ownership behavior. [VERIFIED: repository]
- `embeddings/minilm/embedder_integration_test.go`, `embeddings/splade/golden_dataset_parity_test.go`, `embeddings/splade/golden_regression_test.go`, and `embeddings/openclip/golden_dataset_parity_test.go` — dense/sparse numerical acceptance and skip conditions. [VERIFIED: repository]
- `go.mod` — Go 1.25.0 module baseline and confirmation that no dependency is needed. [VERIFIED: repository]

### Secondary (MEDIUM confidence)

- Local `go version`, environment probes, `go test`, `go vet`, and overlaid `go list` runs on 2026-08-01 — tool availability and baseline evidence. [VERIFIED: local commands]

### Tertiary (LOW confidence)

- None. No web-only or training-only claim is used in the recommendation. [VERIFIED: research log]

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH — no package is added, and the exact Go shape is already overlay-compiled against the module baseline. [VERIFIED: go.mod; Spike 003]
- Architecture: HIGH — the locked context, current method sets, overlay package graph, and negative control agree on the three-file delta. [VERIFIED: 03-CONTEXT.md; Spike 003; current sources]
- Pitfalls: HIGH — import direction, signature compatibility, direct forwarding, and native-test skips were directly inspected or reproduced. [VERIFIED: codebase; local commands]
- Validation: HIGH — all required named suites already exist; only the root conformance test is missing. [VERIFIED: current tests; D-16]
- Security: HIGH for the narrow Phase 3 delta; existing runtime/model-loading controls are unchanged. [VERIFIED: planned files; phase scope]

**Research date:** 2026-08-01  
**Valid until:** 2026-08-31 — the locked API proof is stable, while local tool/runtime availability should be rechecked after 30 days. [RECOMMENDED]
