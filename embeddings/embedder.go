// Package embeddings defines contracts shared by model-specific embedders.
package embeddings

// Embedder produces one typed embedding per document or query.
//
// T is intentionally unconstrained: it must support both dense ([]float32)
// and sparse (splade.SparseVector) rows without a shared type set.
type Embedder[T any] interface {
	EmbedDocuments(documents []string) ([]T, error)
	EmbedQuery(query string) (T, error)
	Close() error
}
