# Phase 3: Generalized Embedder API - Context

**Gathered:** 2026-07-31
**Spike validation folded:** 2026-08-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Add one small public contract that gives dense MiniLM and sparse SPLADE embedders the same compile-time API shape, and make OpenCLIP's text side conform through additive forwarding methods. Existing model construction, result types, inference behavior, resource ownership, and public methods remain compatible. SPLADE golden parity and all existing MiniLM, SPLADE, and OpenCLIP behavior must continue to pass. New models, runtime-polymorphic registries, unified factories, and result-type migrations are outside this phase.

</domain>

<decisions>
## Implementation Decisions

### Shared dense/sparse type model
- **D-01:** Add a generic `embeddings.Embedder[T]` interface. `T` represents one embedding row: MiniLM and OpenCLIP use `[]float32`; SPLADE uses its existing `splade.SparseVector`.
- **D-02:** Keep the type parameter unconstrained (`T any`). The contract provides compile-time result safety without a runtime kind tag, type switch, result wrapper, coercion, or copy.
- **D-03:** A heterogeneous runtime registry is not a Phase 3 requirement. Do not weaken the typed API to make dense and sparse embedders fit in one non-generic collection.

### Common call surface
- **D-04:** The shared interface contains the existing retrieval-shaped methods `EmbedDocuments([]string) ([]T, error)`, `EmbedQuery(string) (T, error)`, and `Close() error`.
- **D-05:** MiniLM and SPLADE conform through their existing method sets; do not rename or duplicate their public methods.
- **D-06:** Add `EmbedDocuments` and `EmbedQuery` forwarding methods to OpenCLIP's concrete embedder. They delegate to `EmbedTexts` and `EmbedText` respectively, without changing text results or inference behavior.
- **D-07:** Preserve OpenCLIP's existing `EmbedTexts`, `EmbedText`, `EmbedImages`, and `EmbedImage` methods. Image embedding remains a concrete OpenCLIP capability and is not added to the common text contract.

### Package placement and compatibility
- **D-08:** Create the root Go package `embeddings` as a dependency-light contract package containing the generic interface. It must not become a facade that imports or constructs the model implementations.
- **D-09:** Keep `minilm.NewEmbedder`, `splade.NewEmbedder`, and `openclip.NewEmbedder` returning their existing concrete pointer types. Do not replace constructor results with interfaces.
- **D-10:** Keep all existing result types in their current packages, including `splade.SparseVector` and `splade.SparseEmbedding`. Do not move them, replace them with root-owned types, or add compatibility aliases merely for namespace uniformity.
- **D-11:** The change is additive: no existing exported constructor, method, parameter, return type, field, or behavior may be removed or changed.
- **D-12:** Do not add generalized factories, adapters, runtime-tagged result unions, or extra public capability packages in this phase.

### Spike-validated constraints
- **D-13:** Spike 003 compiled the exact generic contract against the real model packages. `*minilm.Embedder` and `*openclip.Embedder` satisfy `embeddings.Embedder[[]float32]`; `*splade.Embedder` satisfies `embeddings.Embedder[splade.SparseVector]` without result conversion or copying.
- **D-14:** The overlaid root `embeddings` package has zero imports. Preserve that dependency-free shape so the generalized contract cannot create an import cycle with its model subpackages.
- **D-15:** The negative control proved that OpenCLIP's only conformance gaps are `EmbedDocuments` and `EmbedQuery`; adding the two direct forwarding methods makes it conform while preserving its existing text/image API and validation behavior.
- **D-16:** Carry Spike 003's exact constructor/method signature assertions into Phase 3 verification. Planning must include compile-time conformance checks, `go vet`, the embedding regression suites, and the complete short module suite.

### the agent's Discretion
- Exact interface and package documentation wording.
- Placement and naming of compile-time interface assertions and conformance tests.
- Test-file organization, provided tests prove all three embedders conform at compile time and existing functional/golden behavior remains unchanged.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project scope and requirements
- `.planning/PROJECT.md` — core value, no-CGO constraint, platform compatibility, and the requirement to preserve the settled Phase 2 core API.
- `.planning/REQUIREMENTS.md` — API-01 requirement and Phase 3 traceability.
- `.planning/ROADMAP.md` §Phase 3 — phase goal and literal success criteria.
- `.planning/phases/02-core-api-errors-values/02-CONTEXT.md` — locked `Value`, ownership, `Run`, locking, and compatibility decisions that Phase 3 must preserve.
- `https://github.com/amikos-tech/pure-onnx/issues/49` — original generalized embedder API scope and compatibility intent.

### Validated Phase 3 spike — MUST read before planning
- `.planning/spikes/003-generic-embedder-contract-proof/README.md` — VALIDATED verdict, commands, evidence, negative control, limits, and the explicit signal for Phase 3 planning.
- `.planning/spikes/003-generic-embedder-contract-proof/overlay/embeddings/embedder.go` — proven minimal root generic contract; use as the production-shape blueprint.
- `.planning/spikes/003-generic-embedder-contract-proof/overlay/embeddings/openclip/generalized_embedder.go` — proven two-method OpenCLIP forwarding shape.
- `.planning/spikes/003-generic-embedder-contract-proof/contract_test.go` — exact existing-signature pins, compile-time conformance assertions, and typed interface-dispatch checks to carry into implementation tests.
- `.planning/spikes/003-generic-embedder-contract-proof/overlay.json` — reproducible mapping used to compile the proposed files against the real package graph without editing production code.

### Existing embedder APIs
- `embeddings/minilm/embedder.go` — dense `EmbedDocuments`, `EmbedQuery`, `Close`, concrete constructor, and cached-session behavior that already match the common contract.
- `embeddings/splade/embedder.go` — sparse method set, `SparseVector`, `SparseEmbedding`, pruning configuration, and concrete constructor that remain canonical.
- `embeddings/openclip/embedder.go` — existing text/image API and the integration point for additive document/query forwarding methods.
- `README.md` — current public examples whose call shapes must remain valid.
- `go.mod` — module path and supported Go version for generic interfaces.

### Verification anchors
- `embeddings/splade/golden_dataset_parity_test.go` — existing SPLADE golden-parity requirement.
- `embeddings/splade/golden_regression_test.go` — deterministic sparse result and repeatability checks.
- `embeddings/minilm/embedder_integration_test.go` — dense document/query parity and cached-session behavior.
- `embeddings/openclip/golden_dataset_parity_test.go` — OpenCLIP text/image parity that forwarding methods must not regress.

### Language and compatibility guidance
- `https://go.dev/blog/generic-interfaces` — generic interfaces as typed interface families and guidance to avoid unnecessary constraints.
- `https://go.dev/blog/module-compatibility` — additive public-API evolution and exported-signature compatibility.
- `https://go.dev/ref/spec#Method_sets` — exact method-set rules governing implicit interface conformance.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `minilm.Embedder`: its existing document/query/close method set can satisfy `embeddings.Embedder[[]float32]` without production changes.
- `splade.Embedder`: its existing method set can satisfy `embeddings.Embedder[splade.SparseVector]` without result conversion or copying.
- `openclip.Embedder.EmbedTexts` / `EmbedText`: direct delegation targets for the two additive compatibility methods.
- SPLADE golden and regression tests: existing proof that generalized API work does not change sparse numerical output.
- Spike 003 overlay sources: compile-validated production-shape prototypes for the root contract and OpenCLIP forwarders.
- Spike 003 contract test: reusable exact-signature and conformance assertions that turn compatibility requirements into compile failures.

### Established Patterns
- Model packages own their concrete embedder types, functional-option constructors, and model-specific result/configuration types.
- Embedders cache batch-sized sessions and return ordinary Go-owned result slices while `Close` releases native/tokenizer resources explicitly.
- Go interfaces are satisfied implicitly; the root contract does not need to import its implementations.
- Phase 2 preserved constructor-bound `AdvancedSession.Run()` for the embedder hot path, including its ownership, lease, and lock behavior.

### Integration Points
- New root `embeddings` package: public generic contract and package documentation only.
- `embeddings/openclip/embedder.go`: additive document/query forwarding methods.
- Model package tests: compile-time conformance checks plus regression coverage for existing APIs.
- SPLADE parity suite: acceptance gate for unchanged sparse output.
- `.planning/spikes/003-generic-embedder-contract-proof/`: mandatory pre-planning evidence and implementation blueprint; do not re-research the contract's basic feasibility.

</code_context>

<specifics>
## Specific Ideas

- Favor the smallest additive surface: one generic interface plus two OpenCLIP forwarding methods.
- Existing caller code using concrete embedders must continue to compile unchanged.
- Callers that need OpenCLIP image methods retain the concrete `*openclip.Embedder`; the common interface intentionally exposes only text embedding.
- Spike 003 has already validated feasibility, dependency direction, exact method conformance, and the additive compatibility shape. Planning should convert that proof into production tasks rather than propose another API experiment.

</specifics>

<deferred>
## Deferred Ideas

None — runtime-selected heterogeneous registries, root factories/facades, runtime-tagged results, shared result-type migration, and additional capability packages were considered and explicitly excluded rather than added to future scope.

</deferred>

---

*Phase: 3-Generalized Embedder API*
*Context gathered: 2026-07-31*
*Spike 003 evidence folded: 2026-08-01*
