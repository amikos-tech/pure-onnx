package minilm

import (
	"errors"
	"fmt"
	"math"
	"os"
	"reflect"
	"sync"

	"github.com/amikos-tech/pure-onnx/ort"
	tokenizers "github.com/amikos-tech/pure-tokenizers"
)

const (
	// DefaultSequenceLength matches the Python all-MiniLM-L6-v2 embedding path.
	DefaultSequenceLength = 256
	// OutputEmbeddingDimension is the all-MiniLM-L6-v2 embedding width.
	OutputEmbeddingDimension = 384

	poolingDenominatorEpsilon = float32(1e-9)
	l2NormEpsilon             = float32(1e-12)
)

const (
	defaultInputIDsName      = "input_ids"
	defaultAttentionMaskName = "attention_mask"
	// #nosec G101 -- ONNX input identifier string, not credential material.
	defaultTokenTypeIDsName = "token_type_ids"
	defaultOutputName       = "last_hidden_state"
)

// Option customizes embedder initialization.
type Option func(*config) error

type config struct {
	sequenceLength       int
	tokenizerLibraryPath string
	inputIDsName         string
	attentionMaskName    string
	tokenTypeIDsName     string
	outputName           string
}

func defaultConfig() config {
	return config{
		sequenceLength:    DefaultSequenceLength,
		inputIDsName:      defaultInputIDsName,
		attentionMaskName: defaultAttentionMaskName,
		tokenTypeIDsName:  defaultTokenTypeIDsName,
		outputName:        defaultOutputName,
	}
}

// WithSequenceLength sets truncation and fixed padding length.
func WithSequenceLength(length int) Option {
	return func(cfg *config) error {
		if length <= 0 {
			return fmt.Errorf("sequence length must be > 0, got %d", length)
		}
		cfg.sequenceLength = length
		return nil
	}
}

// WithTokenizerLibraryPath sets the explicit pure-tokenizers shared library path.
func WithTokenizerLibraryPath(path string) Option {
	return func(cfg *config) error {
		if path == "" {
			return fmt.Errorf("tokenizer library path cannot be empty")
		}
		cfg.tokenizerLibraryPath = path
		return nil
	}
}

// WithInputOutputNames overrides ONNX input/output names.
func WithInputOutputNames(inputIDsName, attentionMaskName, tokenTypeIDsName, outputName string) Option {
	return func(cfg *config) error {
		if inputIDsName == "" || attentionMaskName == "" || tokenTypeIDsName == "" || outputName == "" {
			return fmt.Errorf("input/output names cannot be empty")
		}
		cfg.inputIDsName = inputIDsName
		cfg.attentionMaskName = attentionMaskName
		cfg.tokenTypeIDsName = tokenTypeIDsName
		cfg.outputName = outputName
		return nil
	}
}

// Embedder provides local all-MiniLM-L6-v2 embeddings on top of ort.
//
// The caller must initialize ONNX Runtime via ort.SetSharedLibraryPath and
// ort.InitializeEnvironment before calling EmbedDocuments/EmbedQuery.
type Embedder struct {
	modelPath       string
	sequenceLength  int
	tokenizer       *tokenizers.Tokenizer
	inputNames      []string
	outputNames     []string
	sessionsByBatch map[int]*embeddingSession
	runMu           sync.Mutex
}

type embeddingSession struct {
	inputIDs      []int64
	attentionMask []int64
	tokenTypeIDs  []int64

	inputIDsTensor      *ort.Tensor[int64]
	attentionMaskTensor *ort.Tensor[int64]
	tokenTypeIDsTensor  *ort.Tensor[int64]
	outputTensor        *ort.Tensor[float32]
	session             *ort.AdvancedSession
}

// NewEmbedder creates a high-level all-MiniLM-L6-v2 embedder.
//
// modelPath must point to the local ONNX model file.
// tokenizerPath must point to the local tokenizer.json file.
func NewEmbedder(modelPath string, tokenizerPath string, opts ...Option) (*Embedder, error) {
	if modelPath == "" {
		return nil, fmt.Errorf("model path cannot be empty")
	}
	if tokenizerPath == "" {
		return nil, fmt.Errorf("tokenizer path cannot be empty")
	}
	if _, err := os.Stat(modelPath); err != nil {
		return nil, fmt.Errorf("model path %q is not usable: %w", modelPath, err)
	}
	if _, err := os.Stat(tokenizerPath); err != nil {
		return nil, fmt.Errorf("tokenizer path %q is not usable: %w", tokenizerPath, err)
	}

	cfg := defaultConfig()
	for _, opt := range opts {
		if err := opt(&cfg); err != nil {
			return nil, err
		}
	}

	tokenizerOpts := []tokenizers.TokenizerOption{
		tokenizers.WithTruncation(
			uintptr(cfg.sequenceLength),
			tokenizers.TruncationDirectionRight,
			tokenizers.TruncationStrategyLongestFirst,
		),
		tokenizers.WithPadding(true, tokenizers.PaddingStrategy{
			Tag:       tokenizers.PaddingStrategyFixed,
			FixedSize: uintptr(cfg.sequenceLength),
		}),
	}
	if cfg.tokenizerLibraryPath != "" {
		tokenizerOpts = append(tokenizerOpts, tokenizers.WithLibraryPath(cfg.tokenizerLibraryPath))
	}

	tokenizer, err := tokenizers.FromFile(tokenizerPath, tokenizerOpts...)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer: %w", err)
	}

	return &Embedder{
		modelPath:      modelPath,
		sequenceLength: cfg.sequenceLength,
		tokenizer:      tokenizer,
		inputNames: []string{
			cfg.inputIDsName,
			cfg.attentionMaskName,
			cfg.tokenTypeIDsName,
		},
		outputNames:     []string{cfg.outputName},
		sessionsByBatch: make(map[int]*embeddingSession),
	}, nil
}

// Close releases tokenizer resources.
func (e *Embedder) Close() error {
	if e == nil {
		return nil
	}

	e.runMu.Lock()
	defer e.runMu.Unlock()

	var err error

	for batchSize, session := range e.sessionsByBatch {
		if destroyErr := session.Destroy(); destroyErr != nil {
			err = errors.Join(err, fmt.Errorf("failed to destroy batch-%d embedding resources: %w", batchSize, destroyErr))
		}
	}
	e.sessionsByBatch = nil

	if e.tokenizer != nil {
		if closeErr := e.tokenizer.Close(); closeErr != nil {
			err = errors.Join(err, closeErr)
		}
		e.tokenizer = nil
	}

	return err
}

// EmbedDocuments embeds input documents into deterministic 384-d vectors.
func (e *Embedder) EmbedDocuments(documents []string) (_ [][]float32, err error) {
	if e == nil {
		return nil, fmt.Errorf("embedder is nil")
	}
	if len(documents) == 0 {
		return [][]float32{}, nil
	}

	e.runMu.Lock()
	defer e.runMu.Unlock()

	if e.tokenizer == nil || e.sessionsByBatch == nil {
		return nil, fmt.Errorf("embedder has been closed")
	}
	if !ort.IsInitialized() {
		return nil, fmt.Errorf("ONNX Runtime not initialized: call ort.SetSharedLibraryPath and ort.InitializeEnvironment first")
	}

	session, err := e.sessionForBatchLocked(len(documents))
	if err != nil {
		return nil, err
	}

	if err := e.tokenizeInto(
		documents,
		session.inputIDs,
		session.attentionMask,
		session.tokenTypeIDs,
	); err != nil {
		return nil, err
	}

	if err := session.session.Run(); err != nil {
		return nil, fmt.Errorf("embedding inference failed: %w", err)
	}

	embeddings, err := meanPoolAndNormalize(
		session.outputTensor.GetData(),
		session.attentionMask,
		len(documents),
		e.sequenceLength,
		OutputEmbeddingDimension,
	)
	if err != nil {
		return nil, err
	}

	return embeddings, nil
}

func (e *Embedder) sessionForBatchLocked(batchSize int) (_ *embeddingSession, err error) {
	if batchSize <= 0 {
		return nil, fmt.Errorf("batch size must be > 0, got %d", batchSize)
	}

	if session, ok := e.sessionsByBatch[batchSize]; ok {
		return session, nil
	}

	session, err := newEmbeddingSession(
		e.modelPath,
		e.inputNames,
		e.outputNames,
		e.sequenceLength,
		batchSize,
	)
	if err != nil {
		return nil, err
	}
	e.sessionsByBatch[batchSize] = session
	return session, nil
}

func newEmbeddingSession(modelPath string, inputNames []string, outputNames []string, sequenceLength int, batchSize int) (_ *embeddingSession, err error) {
	totalTokens := batchSize * sequenceLength
	inputIDs := make([]int64, totalTokens)
	attentionMask := make([]int64, totalTokens)
	tokenTypeIDs := make([]int64, totalTokens)

	shape := ort.Shape{int64(batchSize), int64(sequenceLength)}
	inputIDsTensor, err := ort.NewTensor[int64](shape, inputIDs)
	if err != nil {
		return nil, fmt.Errorf("failed to create input_ids tensor: %w", err)
	}
	attentionMaskTensor, err := ort.NewTensor[int64](shape, attentionMask)
	if err != nil {
		_ = inputIDsTensor.Destroy()
		return nil, fmt.Errorf("failed to create attention_mask tensor: %w", err)
	}
	tokenTypeIDsTensor, err := ort.NewTensor[int64](shape, tokenTypeIDs)
	if err != nil {
		_ = attentionMaskTensor.Destroy()
		_ = inputIDsTensor.Destroy()
		return nil, fmt.Errorf("failed to create token_type_ids tensor: %w", err)
	}

	outputShape := ort.Shape{int64(batchSize), int64(sequenceLength), OutputEmbeddingDimension}
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		_ = tokenTypeIDsTensor.Destroy()
		_ = attentionMaskTensor.Destroy()
		_ = inputIDsTensor.Destroy()
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	session, err := ort.NewAdvancedSession(
		modelPath,
		inputNames,
		outputNames,
		[]ort.Value{inputIDsTensor, attentionMaskTensor, tokenTypeIDsTensor},
		[]ort.Value{outputTensor},
		nil,
	)
	if err != nil {
		_ = outputTensor.Destroy()
		_ = tokenTypeIDsTensor.Destroy()
		_ = attentionMaskTensor.Destroy()
		_ = inputIDsTensor.Destroy()
		return nil, fmt.Errorf("failed to create embedding session: %w", err)
	}

	return &embeddingSession{
		inputIDs:            inputIDs,
		attentionMask:       attentionMask,
		tokenTypeIDs:        tokenTypeIDs,
		inputIDsTensor:      inputIDsTensor,
		attentionMaskTensor: attentionMaskTensor,
		tokenTypeIDsTensor:  tokenTypeIDsTensor,
		outputTensor:        outputTensor,
		session:             session,
	}, nil
}

func (s *embeddingSession) Destroy() error {
	if s == nil {
		return nil
	}

	err := destroyAll(
		s.session,
		s.outputTensor,
		s.tokenTypeIDsTensor,
		s.attentionMaskTensor,
		s.inputIDsTensor,
	)

	s.inputIDs = nil
	s.attentionMask = nil
	s.tokenTypeIDs = nil
	s.session = nil
	s.outputTensor = nil
	s.tokenTypeIDsTensor = nil
	s.attentionMaskTensor = nil
	s.inputIDsTensor = nil
	return err
}

// EmbedQuery embeds a single query string.
func (e *Embedder) EmbedQuery(query string) ([]float32, error) {
	embeddings, err := e.EmbedDocuments([]string{query})
	if err != nil {
		return nil, err
	}
	if len(embeddings) != 1 {
		return nil, fmt.Errorf("unexpected embedding row count: got %d, want 1", len(embeddings))
	}
	return embeddings[0], nil
}

func (e *Embedder) tokenizeInto(documents []string, inputIDs []int64, attentionMask []int64, tokenTypeIDs []int64) error {
	sequenceLength := e.sequenceLength
	batchSize := len(documents)
	totalTokens := batchSize * sequenceLength

	if len(inputIDs) != totalTokens || len(attentionMask) != totalTokens || len(tokenTypeIDs) != totalTokens {
		return fmt.Errorf(
			"token buffer length mismatch: got input_ids=%d attention_mask=%d token_type_ids=%d, want %d",
			len(inputIDs),
			len(attentionMask),
			len(tokenTypeIDs),
			totalTokens,
		)
	}

	clear(inputIDs)
	clear(attentionMask)
	clear(tokenTypeIDs)

	for i, document := range documents {
		encoding, err := e.tokenizer.Encode(
			document,
			tokenizers.WithAddSpecialTokens(),
			tokenizers.WithReturnAttentionMask(),
			tokenizers.WithReturnTypeIDs(),
		)
		if err != nil {
			return fmt.Errorf("failed to tokenize document %d: %w", i, err)
		}
		if encoding == nil {
			return fmt.Errorf("failed to tokenize document %d: empty tokenizer result", i)
		}

		rowStart := i * sequenceLength
		rowEnd := rowStart + sequenceLength
		fillUint32AsInt64(inputIDs[rowStart:rowEnd], encoding.IDs)

		if len(encoding.AttentionMask) > 0 {
			fillUint32AsInt64(attentionMask[rowStart:rowEnd], encoding.AttentionMask)
		} else {
			deriveAttentionMask(attentionMask[rowStart:rowEnd], inputIDs[rowStart:rowEnd])
		}

		if len(encoding.TypeIDs) > 0 {
			fillUint32AsInt64(tokenTypeIDs[rowStart:rowEnd], encoding.TypeIDs)
		}
	}

	return nil
}

func fillUint32AsInt64(dst []int64, src []uint32) {
	if len(dst) == 0 || len(src) == 0 {
		return
	}
	copyCount := len(dst)
	if len(src) < copyCount {
		copyCount = len(src)
	}
	for i := 0; i < copyCount; i++ {
		dst[i] = int64(src[i])
	}
}

func deriveAttentionMask(dst []int64, tokenIDs []int64) {
	for i := range dst {
		if tokenIDs[i] != 0 {
			dst[i] = 1
		}
	}
}

func meanPoolAndNormalize(lastHiddenState []float32, attentionMask []int64, batchSize int, sequenceLength int, embeddingDim int64) ([][]float32, error) {
	if batchSize <= 0 {
		return nil, fmt.Errorf("batch size must be > 0, got %d", batchSize)
	}
	if sequenceLength <= 0 {
		return nil, fmt.Errorf("sequence length must be > 0, got %d", sequenceLength)
	}
	if embeddingDim <= 0 {
		return nil, fmt.Errorf("embedding dim must be > 0, got %d", embeddingDim)
	}

	expectedMaskLen := batchSize * sequenceLength
	if len(attentionMask) != expectedMaskLen {
		return nil, fmt.Errorf("attention mask length mismatch: got %d, want %d", len(attentionMask), expectedMaskLen)
	}

	expectedHiddenLen := expectedMaskLen * int(embeddingDim)
	if len(lastHiddenState) != expectedHiddenLen {
		return nil, fmt.Errorf("last_hidden_state length mismatch: got %d, want %d", len(lastHiddenState), expectedHiddenLen)
	}

	embeddings := make([][]float32, batchSize)
	dim := int(embeddingDim)
	for row := 0; row < batchSize; row++ {
		embedding := make([]float32, dim)
		rowMaskOffset := row * sequenceLength

		denominator := float32(0)
		for tokenIndex := 0; tokenIndex < sequenceLength; tokenIndex++ {
			mask := attentionMask[rowMaskOffset+tokenIndex]
			if mask == 0 {
				continue
			}
			weight := float32(mask)
			denominator += weight

			hiddenOffset := (rowMaskOffset + tokenIndex) * dim
			for d := 0; d < dim; d++ {
				embedding[d] += lastHiddenState[hiddenOffset+d] * weight
			}
		}

		if denominator < poolingDenominatorEpsilon {
			denominator = poolingDenominatorEpsilon
		}
		invDenominator := float32(1.0) / denominator
		for d := 0; d < dim; d++ {
			embedding[d] *= invDenominator
		}

		normSquared := 0.0
		for _, value := range embedding {
			normSquared += float64(value * value)
		}
		norm := float32(math.Sqrt(normSquared))
		if norm < l2NormEpsilon {
			norm = l2NormEpsilon
		}
		invNorm := float32(1.0) / norm
		for d := 0; d < dim; d++ {
			embedding[d] *= invNorm
		}

		embeddings[row] = embedding
	}

	return embeddings, nil
}

type destroyer interface {
	Destroy() error
}

func destroyAll(resources ...destroyer) error {
	var err error
	for _, resource := range resources {
		if isNilDestroyer(resource) {
			continue
		}
		if destroyErr := resource.Destroy(); destroyErr != nil {
			err = errors.Join(err, destroyErr)
		}
	}
	return err
}

func isNilDestroyer(resource destroyer) bool {
	if resource == nil {
		return true
	}
	value := reflect.ValueOf(resource)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return value.IsNil()
	default:
		return false
	}
}
