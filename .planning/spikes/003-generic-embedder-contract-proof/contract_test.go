package contractproof

import (
	"image"
	"testing"

	"github.com/amikos-tech/pure-onnx/embeddings"
	"github.com/amikos-tech/pure-onnx/embeddings/minilm"
	"github.com/amikos-tech/pure-onnx/embeddings/openclip"
	"github.com/amikos-tech/pure-onnx/embeddings/splade"
)

// These assertions prove exact generic conformance. They cannot pass through
// return-type coercion because Go interface method signatures must match.
var (
	_ embeddings.Embedder[[]float32]           = (*minilm.Embedder)(nil)
	_ embeddings.Embedder[splade.SparseVector] = (*splade.Embedder)(nil)
	_ embeddings.Embedder[[]float32]           = (*openclip.Embedder)(nil)
)

// These assignments pin the existing public constructor and method signatures.
// If the proposed additive API changes an existing signature, this package no
// longer compiles.
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

func queryThroughContract[T any](embedder embeddings.Embedder[T], query string) (T, error) {
	return embedder.EmbedQuery(query)
}

func documentsThroughContract[T any](embedder embeddings.Embedder[T], documents []string) ([]T, error) {
	return embedder.EmbedDocuments(documents)
}

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
