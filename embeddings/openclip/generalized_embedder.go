package openclip

// EmbedDocuments is an alias for EmbedTexts, added solely to satisfy embeddings.Embedder[T].
func (e *Embedder) EmbedDocuments(documents []string) ([][]float32, error) {
	return e.EmbedTexts(documents)
}

// EmbedQuery is an alias for EmbedText, added solely to satisfy embeddings.Embedder[T].
func (e *Embedder) EmbedQuery(query string) ([]float32, error) {
	return e.EmbedText(query)
}
