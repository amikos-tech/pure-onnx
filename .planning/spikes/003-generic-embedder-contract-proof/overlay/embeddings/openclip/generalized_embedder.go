package openclip

// EmbedDocuments forwards document-shaped text input to EmbedTexts.
func (e *Embedder) EmbedDocuments(documents []string) ([][]float32, error) {
	return e.EmbedTexts(documents)
}

// EmbedQuery forwards query-shaped text input to EmbedText.
func (e *Embedder) EmbedQuery(query string) ([]float32, error) {
	return e.EmbedText(query)
}
