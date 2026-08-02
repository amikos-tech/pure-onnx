package openclip

// EmbedDocuments forwards to EmbedTexts; it exists solely to satisfy embeddings.Embedder[T].
func (e *Embedder) EmbedDocuments(documents []string) ([][]float32, error) {
	return e.EmbedTexts(documents)
}

// EmbedQuery forwards to EmbedText; it exists solely to satisfy embeddings.Embedder[T].
func (e *Embedder) EmbedQuery(query string) ([]float32, error) {
	return e.EmbedText(query)
}
