package splade

import (
	"container/list"
	"errors"
	"fmt"
	"math"
	"os"
	"runtime/debug"
	"sort"
	"sync"

	"github.com/amikos-tech/pure-onnx/embeddings/internal/ortutil"
	"github.com/amikos-tech/pure-onnx/ort"
	tokenizers "github.com/amikos-tech/pure-tokenizers"
)

const (
	// DefaultSequenceLength matches the common SPLADE token window.
	DefaultSequenceLength = 256
	// DefaultMaxCachedBatchSessions bounds in-memory ONNX session cache growth.
	DefaultMaxCachedBatchSessions = 8
)

const (
	defaultInputIDsName      = "input_ids"
	defaultAttentionMaskName = "input_mask"
	// #nosec G101 -- ONNX input identifier string, not credential material.
	defaultTokenTypeIDsName = "segment_ids"
	defaultOutputName       = "output"
)

// OutputLayout describes the tensor layout returned by the model.
type OutputLayout string

const (
	// OutputLayoutTokenLogits expects [batch, sequenceLength, vocabSize] token logits.
	OutputLayoutTokenLogits OutputLayout = "token_logits"
	// OutputLayoutDocumentLogits expects [batch, vocabSize] document logits.
	OutputLayoutDocumentLogits OutputLayout = "document_logits"
)

// SparseVector is a sparse representation of one document embedding.
type SparseVector struct {
	Indices []int     `json:"indices"`
	Values  []float32 `json:"values"`
	Labels  []string  `json:"labels,omitempty"`
}

// Validate checks sparse vector parallel-slice invariants.
func (v SparseVector) Validate() error {
	if len(v.Indices) != len(v.Values) {
		return fmt.Errorf("sparse vector has mismatched indices/values lengths: indices=%d values=%d", len(v.Indices), len(v.Values))
	}
	if len(v.Labels) > 0 && len(v.Labels) != len(v.Indices) {
		return fmt.Errorf("sparse vector has mismatched labels/indices lengths: labels=%d indices=%d", len(v.Labels), len(v.Indices))
	}
	return nil
}

// SparseEmbedding is an alias for SparseVector.
type SparseEmbedding = SparseVector

// Option customizes embedder initialization.
type Option func(*config) error

type config struct {
	sequenceLength       int
	maxCachedBatchCount  int
	tokenizerLibraryPath string
	inputIDsName         string
	attentionMaskName    string
	tokenTypeIDsName     string
	outputName           string
	useTokenTypeIDs      bool
	vocabSize            int
	outputLayout         OutputLayout
	pruneThreshold       float32
	topK                 int
	applyLog1pReLU       bool
	returnLabels         bool
	slidingWindowEnabled bool
	slidingWindowStride  int
	preProcessor         func(string) string
}

func defaultConfig() config {
	return config{
		sequenceLength:       DefaultSequenceLength,
		maxCachedBatchCount:  DefaultMaxCachedBatchSessions,
		inputIDsName:         defaultInputIDsName,
		attentionMaskName:    defaultAttentionMaskName,
		tokenTypeIDsName:     defaultTokenTypeIDsName,
		outputName:           defaultOutputName,
		useTokenTypeIDs:      true,
		outputLayout:         OutputLayoutTokenLogits,
		pruneThreshold:       0,
		topK:                 0,
		applyLog1pReLU:       true,
		returnLabels:         false,
		slidingWindowEnabled: false,
		slidingWindowStride:  0,
		preProcessor:         nil,
	}
}

// WithSequenceLength sets truncation and fixed padding length.
// When sliding-window mode is enabled, it also defines each window width.
func WithSequenceLength(length int) Option {
	return func(cfg *config) error {
		if length <= 0 {
			return fmt.Errorf("sequence length must be > 0, got %d", length)
		}
		cfg.sequenceLength = length
		return nil
	}
}

// WithMaxCachedBatchSessions bounds how many batch-size-specific sessions are cached.
func WithMaxCachedBatchSessions(limit int) Option {
	return func(cfg *config) error {
		if limit <= 0 {
			return fmt.Errorf("max cached batch sessions must be > 0, got %d", limit)
		}
		cfg.maxCachedBatchCount = limit
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
// tokenTypeIDsName may be empty for models without token_type_ids.
func WithInputOutputNames(inputIDsName, attentionMaskName, tokenTypeIDsName, outputName string) Option {
	return func(cfg *config) error {
		if inputIDsName == "" || attentionMaskName == "" || outputName == "" {
			return fmt.Errorf("input_ids, attention_mask, and output names cannot be empty")
		}
		cfg.inputIDsName = inputIDsName
		cfg.attentionMaskName = attentionMaskName
		cfg.tokenTypeIDsName = tokenTypeIDsName
		cfg.useTokenTypeIDs = tokenTypeIDsName != ""
		cfg.outputName = outputName
		return nil
	}
}

// WithoutTokenTypeIDsInput configures the embedder for models that do not consume token_type_ids.
func WithoutTokenTypeIDsInput() Option {
	return func(cfg *config) error {
		cfg.useTokenTypeIDs = false
		cfg.tokenTypeIDsName = ""
		return nil
	}
}

// WithVocabularySize sets the sparse output vocabulary width expected from the model.
// If omitted, vocab size is read from tokenizer metadata.
func WithVocabularySize(size int) Option {
	return func(cfg *config) error {
		if size <= 0 {
			return fmt.Errorf("vocabulary size must be > 0, got %d", size)
		}
		cfg.vocabSize = size
		return nil
	}
}

// WithTokenLogitsOutput configures output layout [batch, sequenceLength, vocabSize].
func WithTokenLogitsOutput() Option {
	return func(cfg *config) error {
		cfg.outputLayout = OutputLayoutTokenLogits
		return nil
	}
}

// WithDocumentLogitsOutput configures output layout [batch, vocabSize].
func WithDocumentLogitsOutput() Option {
	return func(cfg *config) error {
		cfg.outputLayout = OutputLayoutDocumentLogits
		return nil
	}
}

// WithPruneThreshold drops sparse dimensions with values <= threshold.
func WithPruneThreshold(threshold float32) Option {
	return func(cfg *config) error {
		if threshold < 0 {
			return fmt.Errorf("prune threshold must be >= 0, got %f", threshold)
		}
		cfg.pruneThreshold = threshold
		return nil
	}
}

// WithTopK keeps at most top-k sparse dimensions per embedding (0 means unbounded).
func WithTopK(topK int) Option {
	return func(cfg *config) error {
		if topK < 0 {
			return fmt.Errorf("topK must be >= 0, got %d", topK)
		}
		cfg.topK = topK
		return nil
	}
}

// WithLog1pReLU enables SPLADE-style log(1+relu(x)) transformation.
func WithLog1pReLU() Option {
	return func(cfg *config) error {
		cfg.applyLog1pReLU = true
		return nil
	}
}

// WithoutLog1pReLU disables SPLADE-style log(1+relu(x)) transformation.
func WithoutLog1pReLU() Option {
	return func(cfg *config) error {
		cfg.applyLog1pReLU = false
		return nil
	}
}

// WithReturnLabels includes decoded token labels for each sparse index.
func WithReturnLabels() Option {
	return func(cfg *config) error {
		cfg.returnLabels = true
		return nil
	}
}

// WithSlidingWindow enables overlapping token-window inference.
// Window size is sequence length configured via WithSequenceLength.
func WithSlidingWindow(stride int) Option {
	return func(cfg *config) error {
		if stride <= 0 {
			return fmt.Errorf("sliding window stride must be > 0, got %d", stride)
		}
		cfg.slidingWindowEnabled = true
		cfg.slidingWindowStride = stride
		return nil
	}
}

// WithPreProcessor applies caller-provided text pre-processing before tokenization.
func WithPreProcessor(fn func(string) string) Option {
	return func(cfg *config) error {
		if fn == nil {
			return fmt.Errorf("pre-processor function cannot be nil")
		}
		cfg.preProcessor = fn
		return nil
	}
}

// Embedder provides sparse transformer embeddings on top of ort.
//
// The caller must initialize ONNX Runtime via ort.SetSharedLibraryPath and
// ort.InitializeEnvironment before calling EmbedDocuments/EmbedQuery.
type Embedder struct {
	modelPath       string
	sequenceLength  int
	vocabSize       int
	outputLayout    OutputLayout
	pruneThreshold  float32
	topK            int
	applyLog1pReLU  bool
	returnLabels    bool
	slidingWindow   bool
	slidingStride   int
	preProcessor    func(string) string
	useTokenTypeIDs bool
	tokenizer       *tokenizers.Tokenizer
	labelCache      map[int]string
	inputNames      []string
	outputNames     []string
	// sessionsByBatch caches one session per unique batch size and is LRU-bounded
	// by maxCachedBatchCount to avoid unbounded memory growth.
	sessionsByBatch     map[int]*embeddingSession
	sessionLRU          *list.List
	sessionLRUIndex     map[int]*list.Element
	maxCachedBatchCount int
	runMu               sync.Mutex
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

type tokenWindow struct {
	inputIDs      []int64
	attentionMask []int64
	tokenTypeIDs  []int64
}

// NewEmbedder creates a SPLADE-compatible sparse embedder.
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
	switch cfg.outputLayout {
	case OutputLayoutTokenLogits, OutputLayoutDocumentLogits:
	default:
		return nil, fmt.Errorf("unsupported output layout: %q", cfg.outputLayout)
	}

	if cfg.slidingWindowEnabled && cfg.slidingWindowStride > cfg.sequenceLength {
		return nil, fmt.Errorf("sliding window stride must be <= sequence length (%d), got %d", cfg.sequenceLength, cfg.slidingWindowStride)
	}

	tokenizerOpts := []tokenizers.TokenizerOption{}
	if !cfg.slidingWindowEnabled {
		tokenizerOpts = append(tokenizerOpts,
			tokenizers.WithTruncation(
				uintptr(cfg.sequenceLength),
				tokenizers.TruncationDirectionRight,
				tokenizers.TruncationStrategyLongestFirst,
			),
			tokenizers.WithPadding(true, tokenizers.PaddingStrategy{
				Tag:       tokenizers.PaddingStrategyFixed,
				FixedSize: uintptr(cfg.sequenceLength),
			}),
		)
	}
	if cfg.tokenizerLibraryPath != "" {
		tokenizerOpts = append(tokenizerOpts, tokenizers.WithLibraryPath(cfg.tokenizerLibraryPath))
	}

	tokenizer, err := tokenizers.FromFile(tokenizerPath, tokenizerOpts...)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer: %w", err)
	}

	vocabSize := cfg.vocabSize
	if vocabSize == 0 {
		size, err := tokenizer.VocabSize()
		if err != nil {
			if closeErr := tokenizer.Close(); closeErr != nil {
				return nil, errors.Join(
					fmt.Errorf("failed to derive vocabulary size from tokenizer: %w", err),
					fmt.Errorf("failed to close tokenizer after initialization failure: %w", closeErr),
				)
			}
			return nil, fmt.Errorf("failed to derive vocabulary size from tokenizer: %w", err)
		}
		if size == 0 {
			if closeErr := tokenizer.Close(); closeErr != nil {
				return nil, errors.Join(
					fmt.Errorf("derived vocabulary size is zero"),
					fmt.Errorf("failed to close tokenizer after initialization failure: %w", closeErr),
				)
			}
			return nil, fmt.Errorf("derived vocabulary size is zero")
		}
		vocabSize = int(size)
	}

	inputNames := []string{cfg.inputIDsName, cfg.attentionMaskName}
	if cfg.useTokenTypeIDs {
		inputNames = append(inputNames, cfg.tokenTypeIDsName)
	}

	return &Embedder{
		modelPath:           modelPath,
		sequenceLength:      cfg.sequenceLength,
		vocabSize:           vocabSize,
		outputLayout:        cfg.outputLayout,
		pruneThreshold:      cfg.pruneThreshold,
		topK:                cfg.topK,
		applyLog1pReLU:      cfg.applyLog1pReLU,
		returnLabels:        cfg.returnLabels,
		slidingWindow:       cfg.slidingWindowEnabled,
		slidingStride:       cfg.slidingWindowStride,
		preProcessor:        cfg.preProcessor,
		useTokenTypeIDs:     cfg.useTokenTypeIDs,
		tokenizer:           tokenizer,
		labelCache:          make(map[int]string),
		inputNames:          inputNames,
		outputNames:         []string{cfg.outputName},
		sessionsByBatch:     make(map[int]*embeddingSession),
		sessionLRU:          list.New(),
		sessionLRUIndex:     make(map[int]*list.Element),
		maxCachedBatchCount: cfg.maxCachedBatchCount,
	}, nil
}

// Close releases ONNX session resources and tokenizer resources.
func (e *Embedder) Close() error {
	if e == nil {
		return nil
	}

	e.runMu.Lock()
	defer e.runMu.Unlock()

	var err error

	for batchSize, session := range e.sessionsByBatch {
		if destroyErr := session.Destroy(); destroyErr != nil {
			err = errors.Join(err, fmt.Errorf("failed to destroy batch-%d sparse embedding resources: %w", batchSize, destroyErr))
		}
	}
	e.sessionsByBatch = nil
	e.sessionLRU = nil
	e.sessionLRUIndex = nil
	e.labelCache = nil

	if e.tokenizer != nil {
		if closeErr := e.tokenizer.Close(); closeErr != nil {
			err = errors.Join(err, closeErr)
		}
		e.tokenizer = nil
	}

	return err
}

// EmbedDocuments embeds input documents into sparse vectors.
func (e *Embedder) EmbedDocuments(documents []string) (_ []SparseVector, err error) {
	if e == nil {
		return nil, fmt.Errorf("embedder is nil")
	}
	if len(documents) == 0 {
		return []SparseVector{}, nil
	}

	e.runMu.Lock()
	defer e.runMu.Unlock()

	if e.tokenizer == nil || e.sessionsByBatch == nil {
		return nil, fmt.Errorf("embedder has been closed")
	}
	if !ort.IsInitialized() {
		return nil, fmt.Errorf("ONNX Runtime not initialized: call ort.SetSharedLibraryPath and ort.InitializeEnvironment first")
	}

	processedDocuments, err := e.preprocessDocuments(documents)
	if err != nil {
		return nil, err
	}

	var embeddings []SparseVector
	if e.slidingWindow {
		embeddings, err = e.embedDocumentsSlidingLocked(processedDocuments)
	} else {
		embeddings, err = e.embedDocumentsFixedWindowLocked(processedDocuments)
	}
	if err != nil {
		return nil, err
	}
	if e.returnLabels {
		if err := e.attachLabels(embeddings); err != nil {
			return nil, err
		}
	}

	return embeddings, nil
}

func (e *Embedder) embedDocumentsFixedWindowLocked(documents []string) ([]SparseVector, error) {
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
		return nil, fmt.Errorf("sparse embedding inference failed: %w", err)
	}

	embeddings, err := sparseFromOutput(
		session.outputTensor.GetData(),
		session.attentionMask,
		len(documents),
		e.sequenceLength,
		e.vocabSize,
		e.outputLayout,
		e.pruneThreshold,
		e.topK,
		e.applyLog1pReLU,
	)
	if err != nil {
		return nil, err
	}
	return embeddings, nil
}

func (e *Embedder) embedDocumentsSlidingLocked(documents []string) ([]SparseVector, error) {
	embeddings := make([]SparseVector, len(documents))
	for docIndex, document := range documents {
		windows, err := e.tokenizeSlidingWindows(document)
		if err != nil {
			return nil, fmt.Errorf("failed to tokenize sliding windows for document %d: %w", docIndex, err)
		}

		session, err := e.sessionForBatchLocked(len(windows))
		if err != nil {
			return nil, err
		}
		if err := fillSessionFromWindows(session, windows, e.sequenceLength); err != nil {
			return nil, fmt.Errorf("failed to prepare sliding window tensors for document %d: %w", docIndex, err)
		}

		if err := session.session.Run(); err != nil {
			return nil, fmt.Errorf("sparse embedding inference failed: %w", err)
		}

		windowEmbeddings, err := sparseFromOutput(
			session.outputTensor.GetData(),
			session.attentionMask,
			len(windows),
			e.sequenceLength,
			e.vocabSize,
			e.outputLayout,
			0,
			0,
			e.applyLog1pReLU,
		)
		if err != nil {
			return nil, err
		}
		merged, mergeErr := mergeWindowEmbeddings(windowEmbeddings, e.pruneThreshold, e.topK)
		if mergeErr != nil {
			return nil, fmt.Errorf("failed to merge sliding window embeddings for document %d: %w", docIndex, mergeErr)
		}
		embeddings[docIndex] = merged
	}
	return embeddings, nil
}

func (e *Embedder) sessionForBatchLocked(batchSize int) (_ *embeddingSession, err error) {
	if batchSize <= 0 {
		return nil, fmt.Errorf("batch size must be > 0, got %d", batchSize)
	}

	if session, ok := e.sessionsByBatch[batchSize]; ok {
		e.touchBatchSizeLocked(batchSize)
		return session, nil
	}
	if e.maxCachedBatchCount > 0 && len(e.sessionsByBatch) >= e.maxCachedBatchCount {
		if err := e.evictLeastRecentlyUsedSessionLocked(); err != nil {
			return nil, err
		}
	}

	session, err := newEmbeddingSession(
		e.modelPath,
		e.inputNames,
		e.outputNames,
		e.sequenceLength,
		batchSize,
		e.vocabSize,
		e.outputLayout,
		e.useTokenTypeIDs,
	)
	if err != nil {
		return nil, err
	}
	e.sessionsByBatch[batchSize] = session
	e.touchBatchSizeLocked(batchSize)
	return session, nil
}

func (e *Embedder) touchBatchSizeLocked(batchSize int) {
	if existing := e.sessionLRUIndex[batchSize]; existing != nil {
		e.sessionLRU.MoveToBack(existing)
		return
	}
	e.sessionLRUIndex[batchSize] = e.sessionLRU.PushBack(batchSize)
}

func (e *Embedder) evictLeastRecentlyUsedSessionLocked() error {
	if e.sessionLRU == nil {
		return nil
	}
	oldest := e.sessionLRU.Front()
	if oldest == nil {
		return nil
	}
	batchSize, ok := oldest.Value.(int)
	if !ok {
		return fmt.Errorf("invalid cache bookkeeping value: %T", oldest.Value)
	}
	session := e.sessionsByBatch[batchSize]
	delete(e.sessionsByBatch, batchSize)
	delete(e.sessionLRUIndex, batchSize)
	e.sessionLRU.Remove(oldest)
	if session == nil {
		return nil
	}
	if err := session.Destroy(); err != nil {
		return fmt.Errorf("failed to evict batch-%d sparse embedding resources: %w", batchSize, err)
	}
	return nil
}

func fillSessionFromWindows(session *embeddingSession, windows []tokenWindow, sequenceLength int) error {
	if session == nil {
		return fmt.Errorf("embedding session is nil")
	}
	if sequenceLength <= 0 {
		return fmt.Errorf("sequence length must be > 0, got %d", sequenceLength)
	}
	if len(windows) == 0 {
		return fmt.Errorf("window batch cannot be empty")
	}

	totalTokens := len(windows) * sequenceLength
	if len(session.inputIDs) != totalTokens || len(session.attentionMask) != totalTokens {
		return fmt.Errorf(
			"session token buffer length mismatch: got input_ids=%d attention_mask=%d, want %d",
			len(session.inputIDs),
			len(session.attentionMask),
			totalTokens,
		)
	}
	if session.tokenTypeIDs != nil && len(session.tokenTypeIDs) != totalTokens {
		return fmt.Errorf("session token_type_ids buffer length mismatch: got %d, want %d", len(session.tokenTypeIDs), totalTokens)
	}

	clear(session.inputIDs)
	clear(session.attentionMask)
	if session.tokenTypeIDs != nil {
		clear(session.tokenTypeIDs)
	}

	for i := range windows {
		window := windows[i]
		if len(window.inputIDs) != sequenceLength || len(window.attentionMask) != sequenceLength {
			return fmt.Errorf("window %d has invalid sequence length: input_ids=%d attention_mask=%d want %d", i, len(window.inputIDs), len(window.attentionMask), sequenceLength)
		}
		if session.tokenTypeIDs != nil {
			if len(window.tokenTypeIDs) != sequenceLength {
				return fmt.Errorf("window %d has invalid token_type_ids length: got %d want %d", i, len(window.tokenTypeIDs), sequenceLength)
			}
		} else if window.tokenTypeIDs != nil {
			return fmt.Errorf("window %d includes token_type_ids but session does not expect them", i)
		}

		rowStart := i * sequenceLength
		rowEnd := rowStart + sequenceLength
		copy(session.inputIDs[rowStart:rowEnd], window.inputIDs)
		copy(session.attentionMask[rowStart:rowEnd], window.attentionMask)
		if session.tokenTypeIDs != nil {
			copy(session.tokenTypeIDs[rowStart:rowEnd], window.tokenTypeIDs)
		}
	}
	return nil
}

func newEmbeddingSession(modelPath string, inputNames []string, outputNames []string, sequenceLength int, batchSize int, vocabSize int, outputLayout OutputLayout, useTokenTypeIDs bool) (_ *embeddingSession, err error) {
	totalTokens := batchSize * sequenceLength
	inputIDs := make([]int64, totalTokens)
	attentionMask := make([]int64, totalTokens)

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

	var tokenTypeIDs []int64
	var tokenTypeIDsTensor *ort.Tensor[int64]
	if useTokenTypeIDs {
		tokenTypeIDs = make([]int64, totalTokens)
		tokenTypeIDsTensor, err = ort.NewTensor[int64](shape, tokenTypeIDs)
		if err != nil {
			_ = attentionMaskTensor.Destroy()
			_ = inputIDsTensor.Destroy()
			return nil, fmt.Errorf("failed to create token_type_ids tensor: %w", err)
		}
	}

	var outputShape ort.Shape
	switch outputLayout {
	case OutputLayoutTokenLogits:
		outputShape = ort.Shape{int64(batchSize), int64(sequenceLength), int64(vocabSize)}
	case OutputLayoutDocumentLogits:
		outputShape = ort.Shape{int64(batchSize), int64(vocabSize)}
	default:
		cleanupErr := ortutil.DestroyAll(tokenTypeIDsTensor, attentionMaskTensor, inputIDsTensor)
		if cleanupErr != nil {
			return nil, errors.Join(fmt.Errorf("unsupported output layout: %q", outputLayout), fmt.Errorf("failed to clean up session tensors: %w", cleanupErr))
		}
		return nil, fmt.Errorf("unsupported output layout: %q", outputLayout)
	}

	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		cleanupErr := ortutil.DestroyAll(tokenTypeIDsTensor, attentionMaskTensor, inputIDsTensor)
		if cleanupErr != nil {
			return nil, errors.Join(fmt.Errorf("failed to create output tensor: %w", err), fmt.Errorf("failed to clean up session tensors: %w", cleanupErr))
		}
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	inputValues := []ort.Value{inputIDsTensor, attentionMaskTensor}
	if tokenTypeIDsTensor != nil {
		inputValues = append(inputValues, tokenTypeIDsTensor)
	}

	session, err := ort.NewAdvancedSession(
		modelPath,
		inputNames,
		outputNames,
		inputValues,
		[]ort.Value{outputTensor},
		nil,
	)
	if err != nil {
		cleanupErr := ortutil.DestroyAll(outputTensor, tokenTypeIDsTensor, attentionMaskTensor, inputIDsTensor)
		if cleanupErr != nil {
			return nil, errors.Join(fmt.Errorf("failed to create sparse embedding session: %w", err), fmt.Errorf("failed to clean up session tensors: %w", cleanupErr))
		}
		return nil, fmt.Errorf("failed to create sparse embedding session: %w", err)
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

	err := ortutil.DestroyAll(
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
func (e *Embedder) EmbedQuery(query string) (SparseVector, error) {
	embeddings, err := e.EmbedDocuments([]string{query})
	if err != nil {
		return SparseVector{}, err
	}
	if len(embeddings) != 1 {
		return SparseVector{}, fmt.Errorf("unexpected embedding row count: got %d, want 1", len(embeddings))
	}
	return embeddings[0], nil
}

func (e *Embedder) attachLabels(vectors []SparseVector) error {
	if len(vectors) == 0 {
		return nil
	}
	if e.tokenizer == nil {
		return fmt.Errorf("embedder tokenizer is not initialized")
	}
	if e.labelCache == nil {
		e.labelCache = make(map[int]string)
	}

	const maxUint32 = ^uint32(0)

	for row := range vectors {
		if len(vectors[row].Indices) == 0 {
			continue
		}
		labels := make([]string, len(vectors[row].Indices))
		for i, idx := range vectors[row].Indices {
			if idx < 0 || uint64(idx) > uint64(maxUint32) {
				return fmt.Errorf("sparse index %d is out of uint32 range", idx)
			}
			label, ok := e.labelCache[idx]
			if !ok {
				decoded, err := e.tokenizer.Decode([]uint32{uint32(idx)}, false)
				if err != nil {
					return fmt.Errorf("failed to decode sparse index %d: %w", idx, err)
				}
				label = decoded
				e.labelCache[idx] = label
			}
			labels[i] = label
		}
		vectors[row].Labels = labels
	}
	return nil
}

func (e *Embedder) tokenizeInto(documents []string, inputIDs []int64, attentionMask []int64, tokenTypeIDs []int64) error {
	sequenceLength := e.sequenceLength
	batchSize := len(documents)
	totalTokens := batchSize * sequenceLength

	if len(inputIDs) != totalTokens || len(attentionMask) != totalTokens {
		return fmt.Errorf(
			"token buffer length mismatch: got input_ids=%d attention_mask=%d, want %d",
			len(inputIDs),
			len(attentionMask),
			totalTokens,
		)
	}
	if tokenTypeIDs != nil && len(tokenTypeIDs) != totalTokens {
		return fmt.Errorf("token_type_ids buffer length mismatch: got %d, want %d", len(tokenTypeIDs), totalTokens)
	}

	clear(inputIDs)
	clear(attentionMask)
	if tokenTypeIDs != nil {
		clear(tokenTypeIDs)
	}

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

		if tokenTypeIDs != nil && len(encoding.TypeIDs) > 0 {
			fillUint32AsInt64(tokenTypeIDs[rowStart:rowEnd], encoding.TypeIDs)
		}
	}

	return nil
}

func (e *Embedder) preprocessDocuments(documents []string) ([]string, error) {
	if len(documents) == 0 || e.preProcessor == nil {
		return documents, nil
	}

	processed := make([]string, len(documents))
	for i, document := range documents {
		next, err := e.preprocessDocument(i, document)
		if err != nil {
			return nil, err
		}
		processed[i] = next
	}
	return processed, nil
}

func (e *Embedder) preprocessDocument(index int, document string) (_ string, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("pre-processor panic on document %d: %v\n%s", index, recovered, debug.Stack())
		}
	}()
	return e.preProcessor(document), nil
}

func (e *Embedder) tokenizeSlidingWindows(document string) ([]tokenWindow, error) {
	encoding, err := e.tokenizer.Encode(
		document,
		tokenizers.WithAddSpecialTokens(),
		tokenizers.WithReturnAttentionMask(),
		tokenizers.WithReturnTypeIDs(),
	)
	if err != nil {
		return nil, err
	}
	if encoding == nil {
		return nil, fmt.Errorf("empty tokenizer result")
	}
	return splitEncodingIntoWindows(encoding, e.sequenceLength, e.slidingStride, e.useTokenTypeIDs)
}

func splitEncodingIntoWindows(encoding *tokenizers.EncodeResult, sequenceLength int, stride int, useTokenTypeIDs bool) ([]tokenWindow, error) {
	if encoding == nil {
		return nil, fmt.Errorf("encoding cannot be nil")
	}
	if sequenceLength <= 0 {
		return nil, fmt.Errorf("sequence length must be > 0, got %d", sequenceLength)
	}
	if stride <= 0 {
		return nil, fmt.Errorf("sliding window stride must be > 0, got %d", stride)
	}
	if stride > sequenceLength {
		return nil, fmt.Errorf("sliding window stride must be <= sequence length (%d), got %d", sequenceLength, stride)
	}
	if len(encoding.IDs) == 0 {
		empty := tokenWindow{
			inputIDs:      make([]int64, sequenceLength),
			attentionMask: make([]int64, sequenceLength),
		}
		if useTokenTypeIDs {
			empty.tokenTypeIDs = make([]int64, sequenceLength)
		}
		return []tokenWindow{empty}, nil
	}

	tokenCount := len(encoding.IDs)
	ids := make([]int64, tokenCount)
	fillUint32AsInt64(ids, encoding.IDs)

	attention := make([]int64, tokenCount)
	if len(encoding.AttentionMask) > 0 {
		fillUint32AsInt64(attention, encoding.AttentionMask)
	} else {
		deriveAttentionMask(attention, ids)
	}

	var typeIDs []int64
	if useTokenTypeIDs {
		typeIDs = make([]int64, tokenCount)
		if len(encoding.TypeIDs) > 0 {
			fillUint32AsInt64(typeIDs, encoding.TypeIDs)
		}
	}

	windows := make([]tokenWindow, 0, 1+tokenCount/stride)
	for start := 0; start < tokenCount; start += stride {
		end := start + sequenceLength
		if end > tokenCount {
			end = tokenCount
		}

		window := tokenWindow{
			inputIDs:      make([]int64, sequenceLength),
			attentionMask: make([]int64, sequenceLength),
		}
		copy(window.inputIDs, ids[start:end])
		copy(window.attentionMask, attention[start:end])

		if useTokenTypeIDs {
			window.tokenTypeIDs = make([]int64, sequenceLength)
			copy(window.tokenTypeIDs, typeIDs[start:end])
		}

		windows = append(windows, window)
		if end == tokenCount {
			break
		}
	}

	return windows, nil
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

func sparseFromOutput(output []float32, attentionMask []int64, batchSize int, sequenceLength int, vocabSize int, outputLayout OutputLayout, pruneThreshold float32, topK int, applyLog1pReLU bool) ([]SparseVector, error) {
	if batchSize <= 0 {
		return nil, fmt.Errorf("batch size must be > 0, got %d", batchSize)
	}
	if sequenceLength <= 0 {
		return nil, fmt.Errorf("sequence length must be > 0, got %d", sequenceLength)
	}
	if vocabSize <= 0 {
		return nil, fmt.Errorf("vocabulary size must be > 0, got %d", vocabSize)
	}
	if pruneThreshold < 0 {
		return nil, fmt.Errorf("prune threshold must be >= 0, got %f", pruneThreshold)
	}
	if topK < 0 {
		return nil, fmt.Errorf("topK must be >= 0, got %d", topK)
	}

	expectedMaskLen := batchSize * sequenceLength
	if len(attentionMask) != expectedMaskLen {
		return nil, fmt.Errorf("attention mask length mismatch: got %d, want %d", len(attentionMask), expectedMaskLen)
	}

	embeddings := make([]SparseVector, batchSize)
	switch outputLayout {
	case OutputLayoutTokenLogits:
		expectedLen := expectedMaskLen * vocabSize
		if len(output) != expectedLen {
			return nil, fmt.Errorf("token logits length mismatch: got %d, want %d", len(output), expectedLen)
		}
		for row := 0; row < batchSize; row++ {
			dense := make([]float32, vocabSize)
			rowTokenOffset := row * sequenceLength
			for tokenIndex := 0; tokenIndex < sequenceLength; tokenIndex++ {
				if attentionMask[rowTokenOffset+tokenIndex] == 0 {
					continue
				}
				tokenOffset := (rowTokenOffset + tokenIndex) * vocabSize
				for vocabIndex := 0; vocabIndex < vocabSize; vocabIndex++ {
					value := output[tokenOffset+vocabIndex]
					if applyLog1pReLU {
						if value <= 0 {
							continue
						}
						value = float32(math.Log1p(float64(value)))
					}
					if value > dense[vocabIndex] {
						dense[vocabIndex] = value
					}
				}
			}
			embeddings[row] = denseToSparse(dense, pruneThreshold, topK)
		}
	case OutputLayoutDocumentLogits:
		expectedLen := batchSize * vocabSize
		if len(output) != expectedLen {
			return nil, fmt.Errorf("document logits length mismatch: got %d, want %d", len(output), expectedLen)
		}
		for row := 0; row < batchSize; row++ {
			rowStart := row * vocabSize
			dense := make([]float32, vocabSize)
			copy(dense, output[rowStart:rowStart+vocabSize])
			if applyLog1pReLU {
				for i := range dense {
					if dense[i] <= 0 {
						dense[i] = 0
						continue
					}
					dense[i] = float32(math.Log1p(float64(dense[i])))
				}
			}
			embeddings[row] = denseToSparse(dense, pruneThreshold, topK)
		}
	default:
		return nil, fmt.Errorf("unsupported output layout: %q", outputLayout)
	}

	return embeddings, nil
}

type indexedValue struct {
	index int
	value float32
}

func denseToSparse(dense []float32, pruneThreshold float32, topK int) SparseVector {
	candidates := make([]indexedValue, 0, len(dense)/16)
	for i, value := range dense {
		if value <= pruneThreshold {
			continue
		}
		candidates = append(candidates, indexedValue{index: i, value: value})
	}

	if topK > 0 && len(candidates) > topK {
		sort.Slice(candidates, func(i, j int) bool {
			if candidates[i].value == candidates[j].value {
				return candidates[i].index < candidates[j].index
			}
			return candidates[i].value > candidates[j].value
		})
		candidates = candidates[:topK]
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].index < candidates[j].index
	})

	indices := make([]int, len(candidates))
	values := make([]float32, len(candidates))
	for i := range candidates {
		indices[i] = candidates[i].index
		values[i] = candidates[i].value
	}

	return SparseVector{
		Indices: indices,
		Values:  values,
	}
}

func mergeWindowEmbeddings(windows []SparseVector, pruneThreshold float32, topK int) (SparseVector, error) {
	if len(windows) == 0 {
		return SparseVector{}, nil
	}

	maxPerIndex := make(map[int]float32, len(windows)*8)
	for i, window := range windows {
		if err := window.Validate(); err != nil {
			return SparseVector{}, fmt.Errorf("invalid sparse window %d: %w", i, err)
		}
		pairCount := len(window.Indices)
		for j := 0; j < pairCount; j++ {
			index := window.Indices[j]
			value := window.Values[j]
			if previous, ok := maxPerIndex[index]; !ok || value > previous {
				maxPerIndex[index] = value
			}
		}
	}

	candidates := make([]indexedValue, 0, len(maxPerIndex))
	for index, value := range maxPerIndex {
		if value <= pruneThreshold {
			continue
		}
		candidates = append(candidates, indexedValue{index: index, value: value})
	}

	if topK > 0 && len(candidates) > topK {
		sort.Slice(candidates, func(i, j int) bool {
			if candidates[i].value == candidates[j].value {
				return candidates[i].index < candidates[j].index
			}
			return candidates[i].value > candidates[j].value
		})
		candidates = candidates[:topK]
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].index < candidates[j].index
	})

	indices := make([]int, len(candidates))
	values := make([]float32, len(candidates))
	for i := range candidates {
		indices[i] = candidates[i].index
		values[i] = candidates[i].value
	}

	return SparseVector{
		Indices: indices,
		Values:  values,
	}, nil
}
