package openclip

// EmbedDocuments forwards to EmbedTexts; it exists solely to satisfy embeddings.Embedder[T].
// Unlike splade.Embedder, OpenCLIP's text encoder is symmetric: this produces
// identical output to EmbedTexts, not a document-specific encoding.
func (e *Embedder) EmbedDocuments(documents []string) ([][]float32, error) {
	return e.EmbedTexts(documents)
}

// EmbedQuery forwards to EmbedText; it exists solely to satisfy embeddings.Embedder[T].
// Unlike splade.Embedder, OpenCLIP's text encoder is symmetric: this produces
// identical output to EmbedText, not a query-specific encoding.
func (e *Embedder) EmbedQuery(query string) ([]float32, error) {
	return e.EmbedText(query)
}
