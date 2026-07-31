# Phase 3: Generalized Embedder API - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-31
**Phase:** 3-Generalized Embedder API
**Areas discussed:** Shared dense/sparse type model, cross-model call surface, package placement and backward compatibility

---

## Shared dense/sparse type model

| Option | Description | Selected |
|--------|-------------|----------|
| Generic `Embedder[T]` | One compile-time API family where `T` is the result for one row. Existing MiniLM and SPLADE signatures fit without conversion. | ✓ |
| Separate dense and sparse interfaces | Simple typed APIs, but no single shared callable contract and more duplicated integration paths. | |
| One runtime-tagged result | One non-generic interface can hold either kind, but every caller must inspect a tag and existing APIs need wrappers or new methods. | |

**User's choice:** Generic `Embedder[T]`.

**Notes:** The user explicitly asked whether this would be breaking, then approved it only with non-breaking guarantees. Existing constructors must keep returning concrete pointers, and no existing methods or result types may change. A heterogeneous runtime registry is not required.

---

## Cross-model call surface

| Option | Description | Selected |
|--------|-------------|----------|
| Document/query contract | Keep `EmbedDocuments` and `EmbedQuery`; add forwarding methods to OpenCLIP while retaining its existing text/image API. | ✓ |
| Neutral text contract | Standardize on `EmbedTexts` and `EmbedText`, requiring aliases on MiniLM and SPLADE and losing the query/document distinction. | |
| Separate capabilities and adapters | Preserve every concrete method set but add more interfaces, wrappers, and ownership/lifecycle concepts. | |

**User's choice:** Document/query contract with additive OpenCLIP forwarding methods.

**Notes:** OpenCLIP's existing `EmbedTexts`, `EmbedText`, `EmbedImages`, and `EmbedImage` remain available. The common interface intentionally exposes only text embedding; callers needing image operations retain the concrete embedder.

---

## Package placement and backward compatibility

| Option | Description | Selected |
|--------|-------------|----------|
| Root contract only | Add `embeddings.Embedder[T]`; leave concrete constructors, model packages, and result types unchanged. | ✓ |
| Root contract and shared result types | Put the contract and dense/sparse output types together, using aliases to soften migration. | |
| Root facade with adapters/factories | Construct and wrap model implementations through one package at the cost of more dependencies and lifecycle code. | |
| Separate contracts subpackage | Keep an acyclic leaf package for contracts but introduce another permanent and less obvious import path. | |

**User's choice:** Root contract only; retain existing result types.

**Notes:** The user selected the smallest additive option. `splade.SparseVector` and `splade.SparseEmbedding` remain owned by the SPLADE package; no root facade, generalized factory, adapter layer, or type migration is introduced.

---

## the agent's Discretion

- Exact package and interface documentation wording.
- Placement and naming of compile-time conformance assertions.
- Test-file organization, subject to compile-time conformance and full functional/golden regression coverage.

## Deferred Ideas

None. Runtime registries, tagged results, shared result-type migration, root factories/facades, and additional capability packages were considered and explicitly excluded from Phase 3 rather than deferred.
