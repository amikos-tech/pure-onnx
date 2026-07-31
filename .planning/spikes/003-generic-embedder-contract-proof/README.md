---
spike: 003
name: generic-embedder-contract-proof
type: standard
validates: "Given the existing MiniLM, SPLADE, and OpenCLIP APIs, when the proposed root generic contract and additive OpenCLIP forwarding methods are compiled in isolation, then all three conform without changing current constructors, methods, result types, or creating import cycles"
verdict: VALIDATED
related: []
tags: [embeddings, generics, api, compatibility]
---

# Spike 003: Generic Embedder Contract Proof

## What This Validates

Given the current model packages, compile the proposed `embeddings.Embedder[T]`
contract and OpenCLIP forwarding methods without changing production files.
The proof must show:

- MiniLM satisfies `Embedder[[]float32]` with its existing method set;
- SPLADE satisfies `Embedder[splade.SparseVector]` without converting results;
- OpenCLIP satisfies `Embedder[[]float32]` after adding only two forwarding methods;
- every existing constructor and public embedder method keeps its exact signature;
- the root contract imports no implementation package and creates no import cycle;
- existing embedding package tests still compile and pass with the proposed files overlaid.

## Research

Go generic interfaces are instantiated as typed interface families, and method
signatures must match exactly. A build overlay can introduce proposed files for
compilation without editing the working production packages.

| Approach | Tool/Library | Pros | Cons | Status |
|----------|--------------|------|------|--------|
| Build overlay against real packages | Go `-overlay` | Tests exact package boundaries and method sets without production edits | Overlay paths must be run from the module root | Chosen |
| Local wrapper around OpenCLIP | Go | Small and easy | Does not prove methods can be added directly to the concrete type | Rejected |
| Implement production changes and revert | Go | Tests the literal change | Breaks spike isolation and risks disturbing user work | Rejected |

Primary references:

- https://go.dev/blog/generic-interfaces
- https://go.dev/ref/spec#Method_sets
- https://go.dev/cmd/go/
- https://go.dev/blog/module-compatibility

## How to Run

Run from the repository root:

```bash
go test -count=1 -overlay .planning/spikes/003-generic-embedder-contract-proof/overlay.json \
  ./.planning/spikes/003-generic-embedder-contract-proof

go list -deps -overlay .planning/spikes/003-generic-embedder-contract-proof/overlay.json \
  github.com/amikos-tech/pure-onnx/embeddings \
  github.com/amikos-tech/pure-onnx/embeddings/minilm \
  github.com/amikos-tech/pure-onnx/embeddings/splade \
  github.com/amikos-tech/pure-onnx/embeddings/openclip

go test -count=1 -short \
  -overlay .planning/spikes/003-generic-embedder-contract-proof/overlay.json \
  ./embeddings/...

go vet -overlay .planning/spikes/003-generic-embedder-contract-proof/overlay.json \
  ./.planning/spikes/003-generic-embedder-contract-proof ./embeddings/...

# Negative control: expected to fail because OpenCLIP lacks the two forwarders.
go test -count=1 \
  -overlay .planning/spikes/003-generic-embedder-contract-proof/overlay-contract-only.json \
  ./.planning/spikes/003-generic-embedder-contract-proof
```

## What to Expect

- Compile-time assertions accept all three concrete embedders.
- Exact constructor and existing-method assignments compile unchanged.
- Generic calls reach each embedder's existing validation behavior.
- OpenCLIP forwarding methods preserve the errors from `EmbedText` and
  `EmbedTexts`.
- Package listing completes without an import-cycle error.
- Existing short embedding tests and `go vet` pass.

## Investigation Trail

1. Selected a Go build overlay so the proof exercises real package boundaries
   while keeping production packages untouched.
2. Added exact function-type assignments for every constructor and existing
   public embedder method; a signature change now fails compilation.
3. Added generic compile-time assertions for dense MiniLM, sparse SPLADE, and
   dense OpenCLIP.
4. Added nil-receiver dispatch tests so the proof checks interface forwarding,
   not only static assignability.
5. Ran a contract-only negative control. MiniLM and SPLADE compiled, while
   OpenCLIP failed with the expected missing `EmbedDocuments` and undefined
   `EmbedQuery` diagnostics.
6. Restored the two forwarding methods through the positive overlay. The
   contract proof, import graph, embedding package tests, vet, and complete
   short repository suite all passed.
7. Inspected the overlaid root package with `go list -json`; it has no imports,
   so model packages cannot form a cycle through the contract package.

## Results

**Verdict: VALIDATED.**

The focused contract proof passed:

```text
ok  github.com/amikos-tech/pure-onnx/.planning/spikes/003-generic-embedder-contract-proof  0.396s
```

The root package is a true leaf:

```json
{
  "ImportPath": "github.com/amikos-tech/pure-onnx/embeddings",
  "GoFiles": ["embedder_spike.go"],
  "Imports": null
}
```

The negative control failed for the intended reason:

```text
*openclip.Embedder does not implement embeddings.Embedder[[]float32]
(missing method EmbedDocuments)
embedder.EmbedQuery undefined
```

With the full overlay restored, `go vet` passed and the complete short module
suite passed across `ort`, all embedding packages, examples, and tools. The
embedding package timings were:

```text
ok  github.com/amikos-tech/pure-onnx/embeddings/minilm    0.349s
ok  github.com/amikos-tech/pure-onnx/embeddings/openclip 2.950s
ok  github.com/amikos-tech/pure-onnx/embeddings/splade   0.895s
```

### What the proof establishes

- `Embedder[T]` matches the existing dense and sparse signatures exactly;
- `T = []float32` and `T = splade.SparseVector` require no conversion or copy;
- OpenCLIP needs exactly the two additive forwarding methods captured here;
- all existing constructors and model-specific methods keep their exact types;
- the root contract adds no dependency edge and therefore no import cycle;
- the proposed shape does not disturb the current short regression suite.

### Limits

- Remote model/dataset parity tests still depend on their normal environment
  variables and were not forced by this spike. Their production code paths are
  untouched, and existing local golden regression coverage passed.
- Exact function-type assignments protect the current public surface used by
  this project; they are not a general exported-API diff tool for unknown
  downstream modules.
- The proof does not benchmark interface dispatch. Existing concrete calls
  remain available, and no performance claim is needed for the compatibility
  decision.

### Signal for Phase 3

Planning can use the two overlaid production files as the minimal implementation
shape: one dependency-free generic contract plus two OpenCLIP forwarders. Keep
the exact compatibility assertions and regression checks, and do not add a
factory, adapter layer, tagged result, or shared result-type migration.
