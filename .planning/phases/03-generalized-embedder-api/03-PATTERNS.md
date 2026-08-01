# Phase 3: Generalized Embedder API - Pattern Map

**Mapped:** 2026-08-01
**Files analyzed:** 3 new files
**Analogs found:** 3 / 3
**Requirement:** API-01

## Scope Conclusion

The smallest validated Phase 3 change is exactly three additions:

1. `embeddings/embedder.go` — a zero-import generic contract.
2. `embeddings/openclip/generalized_embedder.go` — two direct text-method forwarders.
3. `embeddings/embedder_test.go` — external-package compile-time and forwarding proof.

Do not modify `embeddings/minilm/embedder.go`, `embeddings/splade/embedder.go`, or
`embeddings/openclip/embedder.go`. Their existing concrete constructors, result
types, inference paths, image methods, validation, locking, caches, and cleanup
behavior are compatibility boundaries, not implementation targets.

## File Classification

| New File | Role | Data Flow | Closest Analog | Match Quality |
|----------|------|-----------|----------------|---------------|
| `embeddings/embedder.go` | provider (interface contract) | request-response | `.planning/spikes/003-generic-embedder-contract-proof/overlay/embeddings/embedder.go` | exact validated prototype |
| `embeddings/openclip/generalized_embedder.go` | model adapter | request-response | `.planning/spikes/003-generic-embedder-contract-proof/overlay/embeddings/openclip/generalized_embedder.go` | exact validated prototype |
| `embeddings/embedder_test.go` | test | request-response + compile-time conformance | `.planning/spikes/003-generic-embedder-contract-proof/contract_test.go` | exact proof with package-name adaptation |

## Pattern Assignments

### `embeddings/embedder.go` (provider contract, request-response)

**Primary analog:**
`.planning/spikes/003-generic-embedder-contract-proof/overlay/embeddings/embedder.go`

**Why this is the closest match:** Spike 003 compiled this exact interface against
the real MiniLM, SPLADE, and OpenCLIP packages and confirmed that the root package
has no imports. Promote the prototype; do not redesign it.

**Package and imports pattern** (lines 1-2):

```go
// Package embeddings defines contracts shared by model-specific embedders.
package embeddings
```

There is deliberately no `import` block. Keep the production root package a leaf:
it must not import `minilm`, `splade`, `openclip`, `ort`, or any third-party package.

**Core contract pattern** (lines 4-9):

```go
// Embedder produces one typed embedding per document or query.
type Embedder[T any] interface {
	EmbedDocuments(documents []string) ([]T, error)
	EmbedQuery(query string) (T, error)
	Close() error
}
```

Copy this method set and unconstrained `T any` exactly. One `T` is one result row:

- MiniLM and OpenCLIP instantiate it as `Embedder[[]float32]`.
- SPLADE instantiates it as `Embedder[splade.SparseVector]`.

**Supporting repository convention:** `ort/types.go` lines 78-102 already uses a
small documented interface and unconstrained generic typing without conversion:

```go
type Value interface {
	Destroy() error
	Type() ValueType
	ortValue()
}

func AsTensor[T any](value Value) (*Tensor[T], bool) {
	tensor, ok := value.(*Tensor[T])
	if !ok || tensor == nil {
		return nil, false
	}
	return tensor, true
}
```

Use its concise GoDoc style, but copy the Phase 3 method set from the Spike overlay.

**Validation and error pattern:** None belongs in this file. The interface only
declares the existing `error` returns; concrete embedders continue to own input,
state, runtime, and cleanup validation.

**Auth/guard pattern:** Not applicable. This is an in-process Go contract with no
authentication or authorization boundary.

**Do not add:**

- imports or implementation references;
- a root constructor/factory;
- a runtime kind tag, union, `any` result, or type switch;
- model-specific methods such as OpenCLIP image embedding;
- a type constraint narrower than `any`;
- adapters or aliases for existing model result types.

---

### `embeddings/openclip/generalized_embedder.go` (model adapter, request-response)

**Primary analog:**
`.planning/spikes/003-generic-embedder-contract-proof/overlay/embeddings/openclip/generalized_embedder.go`

**Why this is the closest match:** The negative control showed these two method
names are OpenCLIP's only conformance gap. The positive overlay compiled and passed
the embedding regression suite with these exact direct returns.

**Package and imports pattern** (line 1):

```go
package openclip
```

There is no import block. Both receiver and result types already belong to the
package or use predeclared types.

**Core forwarding pattern** (lines 3-11):

```go
// EmbedDocuments forwards document-shaped text input to EmbedTexts.
func (e *Embedder) EmbedDocuments(documents []string) ([][]float32, error) {
	return e.EmbedTexts(documents)
}

// EmbedQuery forwards query-shaped text input to EmbedText.
func (e *Embedder) EmbedQuery(query string) ([]float32, error) {
	return e.EmbedText(query)
}
```

Each method must remain a single direct return. Do not add allocation, locking,
tokenization, inference, post-processing, validation, or error wrapping here.

**Validation target pattern:** `embeddings/openclip/embedder.go` lines 391-430
contains the existing text validation and inference path that both generalized
methods must continue to use:

```go
// EmbedTexts embeds input strings with the CLIP text encoder.
func (e *Embedder) EmbedTexts(texts []string) (_ [][]float32, err error) {
	if e == nil {
		return nil, fmt.Errorf("embedder is nil")
	}
	if len(texts) == 0 {
		return [][]float32{}, nil
	}

	e.runMu.Lock()
	defer e.runMu.Unlock()

	if e.tokenizer == nil || e.textSessionsByBatch == nil || e.visionSessionsByBatch == nil {
		return nil, fmt.Errorf("embedder has been closed")
	}
	if !ort.IsInitialized() {
		return nil, fmt.Errorf("ONNX Runtime not initialized: call ort.SetSharedLibraryPath and ort.InitializeEnvironment first")
	}

	// Existing session, tokenization, inference, and post-processing continue here.
}
```

**Existing single-item delegation pattern:** `embeddings/openclip/embedder.go`
lines 432-442 already implements the same transparent delegation shape:

```go
// EmbedText embeds a single string with the CLIP text encoder.
func (e *Embedder) EmbedText(text string) ([]float32, error) {
	embeddings, err := e.EmbedTexts([]string{text})
	if err != nil {
		return nil, err
	}
	if len(embeddings) != 1 {
		return nil, fmt.Errorf("unexpected embedding row count: got %d, want 1", len(embeddings))
	}
	return embeddings[0], nil
}
```

`EmbedQuery` must call this method rather than recreate its logic.

**Error handling pattern:** Return the target method's values unchanged. In
particular, do not wrap errors: direct and generalized calls should produce the
same validation error in the same run.

**Auth/guard pattern:** Not applicable. Existing receiver/runtime checks in
`EmbedTexts` and `EmbedText` are the relevant guards and must be reached by
delegation.

**Compatibility boundary:** Preserve `EmbedTexts`, `EmbedText`, `EmbedImages`,
`EmbedImage`, `Close`, and the concrete return from `openclip.NewEmbedder`.
Image embedding stays concrete-only and must not enter the shared interface.

---

### `embeddings/embedder_test.go` (test, request-response + compile-time)

**Primary analog:**
`.planning/spikes/003-generic-embedder-contract-proof/contract_test.go`

**Supporting local analog:** `ort/public_api_compat_test.go` lines 1-15 proves the
repository already places exported API compatibility checks in an external test
package (`package ort_test`) that imports the package under test.

**Required package adaptation:** The Spike file uses `package contractproof` at
line 1. At its production location use `package embeddings_test`, as required by
`03-RESEARCH.md` lines 112-118. This lets the test import the root contract and all
three child packages without adding those dependencies to production `embeddings`.

**Imports pattern** (Spike lines 3-11):

```go
import (
	"image"
	"testing"

	"github.com/amikos-tech/pure-onnx/embeddings"
	"github.com/amikos-tech/pure-onnx/embeddings/minilm"
	"github.com/amikos-tech/pure-onnx/embeddings/openclip"
	"github.com/amikos-tech/pure-onnx/embeddings/splade"
)
```

Follow the existing standard-library blank-line project-import grouping.

**Compile-time conformance pattern** (Spike lines 13-19):

```go
// These assertions prove exact generic conformance. They cannot pass through
// return-type coercion because Go interface method signatures must match.
var (
	_ embeddings.Embedder[[]float32]           = (*minilm.Embedder)(nil)
	_ embeddings.Embedder[splade.SparseVector] = (*splade.Embedder)(nil)
	_ embeddings.Embedder[[]float32]           = (*openclip.Embedder)(nil)
)
```

Do not replace these with runtime type assertions. A method mismatch should make
the package fail to compile.

**Existing public signature pins** (Spike lines 21-42):

```go
var (
	_ func(string, string, ...minilm.Option) (*minilm.Embedder, error)                     = minilm.NewEmbedder
	_ func(string, string, ...splade.Option) (*splade.Embedder, error)                     = splade.NewEmbedder
	_ func(string, string, string, string, ...openclip.Option) (*openclip.Embedder, error) = openclip.NewEmbedder

	_ func(*minilm.Embedder, []string) ([][]float32, error) = (*minilm.Embedder).EmbedDocuments
	_ func(*minilm.Embedder, string) ([]float32, error)     = (*minilm.Embedder).EmbedQuery
	_ func(*minilm.Embedder) error                          = (*minilm.Embedder).Close

	_ func(*splade.Embedder, []string) ([]splade.SparseVector, error) = (*splade.Embedder).EmbedDocuments
	_ func(*splade.Embedder, string) (splade.SparseVector, error)     = (*splade.Embedder).EmbedQuery
	_ func(*splade.Embedder) error                                    = (*splade.Embedder).Close

	_ func(*openclip.Embedder, []string) ([][]float32, error)      = (*openclip.Embedder).EmbedTexts
	_ func(*openclip.Embedder, string) ([]float32, error)          = (*openclip.Embedder).EmbedText
	_ func(*openclip.Embedder, []image.Image) ([][]float32, error) = (*openclip.Embedder).EmbedImages
	_ func(*openclip.Embedder, image.Image) ([]float32, error)     = (*openclip.Embedder).EmbedImage
	_ func(*openclip.Embedder) error                               = (*openclip.Embedder).Close
)
```

Retain the whole block. It protects concrete constructor results, the dense/sparse
result types, OpenCLIP image capability, and `Close` while the shared interface is
added.

**New OpenCLIP signature pins:** Add these two assignments from
`03-RESEARCH.md` lines 457-458:

```go
_ func(*openclip.Embedder, []string) ([][]float32, error) = (*openclip.Embedder).EmbedDocuments
_ func(*openclip.Embedder, string) ([]float32, error)     = (*openclip.Embedder).EmbedQuery
```

They belong with the existing OpenCLIP method assignments.

**Typed interface-dispatch helpers** (Spike lines 44-50):

```go
func queryThroughContract[T any](embedder embeddings.Embedder[T], query string) (T, error) {
	return embedder.EmbedQuery(query)
}

func documentsThroughContract[T any](embedder embeddings.Embedder[T], documents []string) ([]T, error) {
	return embedder.EmbedDocuments(documents)
}
```

**Validation dispatch test pattern** (Spike lines 52-76):

```go
func TestTypedInterfaceDispatchReachesExistingValidation(t *testing.T) {
	t.Run("minilm", func(t *testing.T) {
		var embedder *minilm.Embedder
		if _, err := queryThroughContract[[]float32](embedder, "query"); err == nil {
			t.Fatal("nil MiniLM embedder unexpectedly succeeded")
		}
	})

	t.Run("splade", func(t *testing.T) {
		var embedder *splade.Embedder
		if _, err := queryThroughContract[splade.SparseVector](embedder, "query"); err == nil {
			t.Fatal("nil SPLADE embedder unexpectedly succeeded")
		}
	})

	t.Run("openclip", func(t *testing.T) {
		var embedder *openclip.Embedder
		if _, err := queryThroughContract[[]float32](embedder, "query"); err == nil {
			t.Fatal("nil OpenCLIP embedder unexpectedly succeeded")
		}
		if _, err := documentsThroughContract[[]float32](embedder, []string{"document"}); err == nil {
			t.Fatal("nil OpenCLIP embedder unexpectedly succeeded")
		}
	})
}
```

This test needs no model, tokenizer, ONNX Runtime library, or network access. Nil
receivers deliberately exercise the real method dispatch and existing validation.

**Forwarding error-preservation pattern** (Spike lines 78-98):

```go
func TestOpenCLIPForwardersPreserveExistingValidation(t *testing.T) {
	var embedder *openclip.Embedder

	_, directQueryErr := embedder.EmbedText("query")
	_, forwardedQueryErr := embedder.EmbedQuery("query")
	if directQueryErr == nil || forwardedQueryErr == nil {
		t.Fatal("nil OpenCLIP embedder unexpectedly succeeded")
	}
	if directQueryErr.Error() != forwardedQueryErr.Error() {
		t.Fatalf("query forwarding changed validation error: direct=%q forwarded=%q", directQueryErr, forwardedQueryErr)
	}

	_, directDocumentsErr := embedder.EmbedTexts([]string{"document"})
	_, forwardedDocumentsErr := embedder.EmbedDocuments([]string{"document"})
	if directDocumentsErr == nil || forwardedDocumentsErr == nil {
		t.Fatal("nil OpenCLIP embedder unexpectedly succeeded")
	}
	if directDocumentsErr.Error() != forwardedDocumentsErr.Error() {
		t.Fatalf("document forwarding changed validation error: direct=%q forwarded=%q", directDocumentsErr, forwardedDocumentsErr)
	}
}
```

Compare direct and forwarded errors from the same call shape. Do not pin a literal
message string as a public contract.

**Auth/guard pattern:** Not applicable. The test verifies in-process receiver
validation, not identity or access control.

## Shared Patterns

### Dependency Direction

**Source:** Spike overlay `embeddings/embedder.go` lines 1-9 and Spike README
lines 98-118.

**Apply to:** `embeddings/embedder.go` and `embeddings/embedder_test.go`.

- Production `embeddings` has zero imports.
- Cross-package dependencies live only in `package embeddings_test`.
- Model packages implicitly satisfy the contract; the contract never references
  implementations.

The planner should include a static package-boundary check:

```bash
test -z "$(go list -f '{{join .Imports " "}}' ./embeddings)"
```

### Implicit Conformance, No Adapters

**Sources:**

- `embeddings/minilm/embedder.go` lines 294-375 and 542-552.
- `embeddings/splade/embedder.go` lines 422-493 and 807-817.
- `embeddings/openclip/embedder.go` lines 352-442.

**Apply to:** all compile-time assertions in `embeddings/embedder_test.go`.

MiniLM and SPLADE already expose the three exact methods. OpenCLIP should satisfy
the same dense instantiation only after the two additive forwarders land. Do not
add production conformance assertions that import the root package into a child
package; the external root test is the single compatibility proof location.

### Transparent Forwarding and Error Handling

**Source:** Spike OpenCLIP overlay lines 3-11.

**Apply to:** both methods in `embeddings/openclip/generalized_embedder.go`.

The direct-return shape is the error-handling pattern. It preserves nil-receiver,
empty-input, closed-state, runtime-initialization, tokenization, inference, and
post-processing behavior without duplicating any branch.

### Exact Public Signature Compatibility

**Source:** Spike contract test lines 21-42, extended by `03-RESEARCH.md` lines
457-458 for the two new methods.

**Apply to:** `embeddings/embedder_test.go`.

Use function-type assignments, not reflection. They make any constructor return
change, result-type change, receiver change, or method signature change fail at
compile time.

### Resource Ownership

**Sources:**

- `embeddings/minilm/embedder.go` lines 294-322.
- `embeddings/splade/embedder.go` lines 422-451.
- `embeddings/openclip/embedder.go` lines 352-389.

**Apply to:** the root interface's `Close() error` method and compatibility pins.

The shared contract exposes the existing cleanup operation but does not change
ownership. Concrete embedders continue to release cached sessions and tokenizers.
No finalizer, lifecycle wrapper, or generalized resource manager belongs in Phase 3.

### Regression Tests Are Reused, Not Rewritten

| Existing Gate | Concrete Pattern to Preserve | Relevant Lines |
|---------------|------------------------------|----------------|
| `embeddings/minilm/embedder_integration_test.go` | document/query numerical parity and cached-session reuse | 66-146 |
| `embeddings/splade/golden_dataset_parity_test.go` | hosted sparse indices/values/labels parity | 26-141 |
| `embeddings/splade/golden_regression_test.go` | deterministic local sparse golden rows and repeatability | 41-150 |
| `embeddings/openclip/embedder_integration_test.go` | existing text/image behavior, errors-after-close, idempotent close | 11-85, 187-235 |
| `embeddings/openclip/golden_dataset_parity_test.go` | existing text/image/logit golden parity | 47-176 |

Do not edit these gates or their fixtures for Phase 3. Run them against the additive
surface. Native and hosted parity tests may skip when runtime/model/dataset
configuration is absent, so phase evidence must distinguish `PASS` from `SKIP`.

### Verification Command Pattern

Use the same progression validated by Spike 003:

```bash
go test -count=1 ./embeddings
go test -run '^$' ./embeddings/...
test -z "$(go list -f '{{join .Imports " "}}' ./embeddings)"
go test -count=1 -short ./embeddings/...
go vet ./embeddings/...
go test -count=1 -short ./...
```

Then run the named MiniLM, SPLADE, and OpenCLIP native/golden selectors from
`03-RESEARCH.md` in a configured environment and confirm they ran rather than
skipped.

## No Analog Found

None. Spike 003 provides an exact validated prototype for every new Phase 3 file.
Current production sources and regression tests provide the supporting repository
conventions and compatibility boundaries.

## Files Explicitly Not Modified

| Existing File | Reason |
|---------------|--------|
| `embeddings/minilm/embedder.go` | Its current pointer method set already satisfies `Embedder[[]float32]`. |
| `embeddings/splade/embedder.go` | Its current pointer method set already satisfies `Embedder[splade.SparseVector]`; sparse types stay package-owned. |
| `embeddings/openclip/embedder.go` | Existing text/image inference stays unchanged; additive aliases live in the small new file. |
| Existing integration/golden test files | They remain regression gates; no fixture or numerical expectation changes are needed. |
| `go.mod` / `go.sum` | The implementation uses the Go 1.25 language baseline and adds no dependency. |

## Metadata

**Analog search scope:** `.planning/spikes/003-generic-embedder-contract-proof/`,
`embeddings/`, and external API compatibility tests under `ort/`.

**Files enumerated:** 28 files across the Spike 003 and embedding trees.

**Source/test analog candidates read:** 14, including all three current embedder
implementations, the exact overlay files, the Spike contract proof, external API
compatibility convention, and the named Phase 3 regression gates.

**Primary analogs selected:** 3 exact validated prototypes.

**Pattern extraction date:** 2026-08-01
