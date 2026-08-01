// Package embeddings defines contracts shared by model-specific embedders.
package embeddings

// Embedder produces one typed embedding per document or query.
type Embedder[T any] interface {
	EmbedDocuments(documents []string) ([]T, error)
	EmbedQuery(query string) (T, error)
	Close() error
}
