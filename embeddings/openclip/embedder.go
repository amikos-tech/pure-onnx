package openclip

import (
	"container/list"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"math"
	"os"
	"sync"

	"github.com/amikos-tech/pure-onnx/embeddings/internal/ortutil"
	"github.com/amikos-tech/pure-onnx/ort"
	tokenizers "github.com/amikos-tech/pure-tokenizers"
)

const (
	// DefaultSequenceLength matches CLIP text token length.
	DefaultSequenceLength = 77
	// DefaultImageSize matches CLIP image processor size.
	DefaultImageSize = 224
	// OutputEmbeddingDimension is the CLIP projection width for ViT-B/32.
	OutputEmbeddingDimension = 512
	// DefaultMaxCachedBatchSessions bounds in-memory ONNX session cache growth.
	DefaultMaxCachedBatchSessions = 8
	// DefaultCLIPLogitScale is 1/0.07, used by CLIP zero-shot similarity workflows.
	DefaultCLIPLogitScale = float32(14.285714)

	l2NormEpsilon = float32(1e-12)
)

const (
	defaultTextInputIDsName      = "input_ids"
	defaultTextAttentionMaskName = "attention_mask"
	defaultTextOutputName        = "text_embeds"

	defaultVisionInputName  = "pixel_values"
	defaultVisionOutputName = "image_embeds"
)

type clipPreprocessorConfig struct {
	doResize           bool
	resizeShortestEdge int
	doCenterCrop       bool
	cropSize           int
	doNormalize        bool
	doRescale          bool
	rescaleFactor      float32
	doConvertRGB       bool
	imageMean          [3]float32
	imageStd           [3]float32
}

func defaultClipPreprocessorConfig() clipPreprocessorConfig {
	return clipPreprocessorConfig{
		doResize:           true,
		resizeShortestEdge: DefaultImageSize,
		doCenterCrop:       true,
		cropSize:           DefaultImageSize,
		doNormalize:        true,
		doRescale:          true,
		rescaleFactor:      float32(1.0 / 255.0),
		doConvertRGB:       true,
		imageMean:          [3]float32{0.48145466, 0.4578275, 0.40821073},
		imageStd:           [3]float32{0.26862954, 0.26130258, 0.27577711},
	}
}

// Option customizes embedder initialization.
type Option func(*config) error

type config struct {
	sequenceLength       int
	imageSize            int
	embeddingDimension   int64
	maxCachedBatchCount  int
	tokenizerLibraryPath string
	textInputIDsName     string
	textAttentionMask    string
	textOutputName       string
	visionInputName      string
	visionOutputName     string
	l2Normalize          bool
}

func defaultConfig() config {
	return config{
		sequenceLength:      DefaultSequenceLength,
		imageSize:           0,
		embeddingDimension:  OutputEmbeddingDimension,
		maxCachedBatchCount: DefaultMaxCachedBatchSessions,
		textInputIDsName:    defaultTextInputIDsName,
		textAttentionMask:   defaultTextAttentionMaskName,
		textOutputName:      defaultTextOutputName,
		visionInputName:     defaultVisionInputName,
		visionOutputName:    defaultVisionOutputName,
		l2Normalize:         true,
	}
}

// WithSequenceLength sets truncation and fixed padding length for text inputs.
func WithSequenceLength(length int) Option {
	return func(cfg *config) error {
		if length <= 0 {
			return fmt.Errorf("sequence length must be > 0, got %d", length)
		}
		cfg.sequenceLength = length
		return nil
	}
}

// WithImageSize forces image preprocessing output size to size x size.
func WithImageSize(size int) Option {
	return func(cfg *config) error {
		if size <= 0 {
			return fmt.Errorf("image size must be > 0, got %d", size)
		}
		cfg.imageSize = size
		return nil
	}
}

// WithEmbeddingDimension configures expected embedding width from both models.
func WithEmbeddingDimension(dim int64) Option {
	return func(cfg *config) error {
		if dim <= 0 {
			return fmt.Errorf("embedding dimension must be > 0, got %d", dim)
		}
		cfg.embeddingDimension = dim
		return nil
	}
}

// WithMaxCachedBatchSessions bounds how many batch-size-specific sessions are cached per modality.
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

// WithTextInputOutputNames overrides ONNX text model input/output names.
func WithTextInputOutputNames(inputIDsName, attentionMaskName, outputName string) Option {
	return func(cfg *config) error {
		if inputIDsName == "" || attentionMaskName == "" || outputName == "" {
			return fmt.Errorf("text input_ids, attention_mask, and output names cannot be empty")
		}
		cfg.textInputIDsName = inputIDsName
		cfg.textAttentionMask = attentionMaskName
		cfg.textOutputName = outputName
		return nil
	}
}

// WithVisionInputOutputNames overrides ONNX vision model input/output names.
func WithVisionInputOutputNames(inputName, outputName string) Option {
	return func(cfg *config) error {
		if inputName == "" || outputName == "" {
			return fmt.Errorf("vision input and output names cannot be empty")
		}
		cfg.visionInputName = inputName
		cfg.visionOutputName = outputName
		return nil
	}
}

// WithL2Normalization applies row-level L2 normalization to embeddings.
func WithL2Normalization() Option {
	return func(cfg *config) error {
		cfg.l2Normalize = true
		return nil
	}
}

// WithoutL2Normalization disables row-level L2 normalization.
func WithoutL2Normalization() Option {
	return func(cfg *config) error {
		cfg.l2Normalize = false
		return nil
	}
}

// Embedder provides OpenCLIP text+vision embeddings on top of ort.
//
// Required artifacts:
//   - text_model.onnx
//   - vision_model.onnx
//   - tokenizer.json
//   - preprocessor_config.json
//
// The caller must initialize ONNX Runtime via ort.SetSharedLibraryPath and
// ort.InitializeEnvironment before calling EmbedTexts/EmbedImages.
type Embedder struct {
	textModelPath         string
	visionModelPath       string
	sequenceLength        int
	imageSize             int
	embeddingDimension    int64
	l2Normalize           bool
	tokenizer             *tokenizers.Tokenizer
	preprocessor          clipPreprocessorConfig
	textInputNames        []string
	textOutputNames       []string
	visionInputNames      []string
	visionOutputNames     []string
	textSessionsByBatch   map[int]*textEmbeddingSession
	textSessionLRU        *list.List
	textSessionLRUIndex   map[int]*list.Element
	visionSessionsByBatch map[int]*visionEmbeddingSession
	visionSessionLRU      *list.List
	visionSessionLRUIndex map[int]*list.Element
	maxCachedBatchCount   int
	runMu                 sync.Mutex
}

type textEmbeddingSession struct {
	// inputIDs and attentionMask are tensor-backed backing stores. Do not
	// reallocate, re-slice, or append after tensor creation.
	inputIDs      []int64
	attentionMask []int64

	inputIDsTensor      *ort.Tensor[int64]
	attentionMaskTensor *ort.Tensor[int64]
	outputTensor        *ort.Tensor[float32]
	session             *ort.AdvancedSession
}

type visionEmbeddingSession struct {
	// pixelValues is the tensor-backed backing store. Reallocation breaks
	// tensor references created from the previous array.
	pixelValues []float32

	pixelValuesTensor *ort.Tensor[float32]
	outputTensor      *ort.Tensor[float32]
	session           *ort.AdvancedSession
}

// NewEmbedder creates an OpenCLIP embedder from local artifact files.
func NewEmbedder(textModelPath string, visionModelPath string, tokenizerPath string, preprocessorConfigPath string, opts ...Option) (*Embedder, error) {
	if err := validateFilePath("text model", textModelPath); err != nil {
		return nil, err
	}
	if err := validateFilePath("vision model", visionModelPath); err != nil {
		return nil, err
	}
	if err := validateFilePath("tokenizer", tokenizerPath); err != nil {
		return nil, err
	}
	if err := validateFilePath("preprocessor config", preprocessorConfigPath); err != nil {
		return nil, err
	}

	cfg := defaultConfig()
	for _, opt := range opts {
		if err := opt(&cfg); err != nil {
			return nil, err
		}
	}

	preprocessor, err := loadPreprocessorConfig(preprocessorConfigPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load preprocessor config: %w", err)
	}
	resolvedImageSize := preprocessor.cropSize
	if resolvedImageSize <= 0 {
		resolvedImageSize = preprocessor.resizeShortestEdge
	}
	if resolvedImageSize <= 0 {
		resolvedImageSize = DefaultImageSize
	}
	if cfg.imageSize > 0 {
		resolvedImageSize = cfg.imageSize
		preprocessor.cropSize = cfg.imageSize
		preprocessor.resizeShortestEdge = cfg.imageSize
		preprocessor.doResize = true
		preprocessor.doCenterCrop = true
	}

	tokenizerOpts := []tokenizers.TokenizerOption{
		tokenizers.WithTruncation(
			uintptr(cfg.sequenceLength), // #nosec G115 -- Default and WithSequenceLength guarantee a positive int, which uintptr can represent.
			tokenizers.TruncationDirectionRight,
			tokenizers.TruncationStrategyLongestFirst,
		),
		tokenizers.WithPadding(true, tokenizers.PaddingStrategy{
			Tag:       tokenizers.PaddingStrategyFixed,
			FixedSize: uintptr(cfg.sequenceLength), // #nosec G115 -- Default and WithSequenceLength guarantee a positive int, which uintptr can represent.
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
		textModelPath:         textModelPath,
		visionModelPath:       visionModelPath,
		sequenceLength:        cfg.sequenceLength,
		imageSize:             resolvedImageSize,
		embeddingDimension:    cfg.embeddingDimension,
		l2Normalize:           cfg.l2Normalize,
		tokenizer:             tokenizer,
		preprocessor:          preprocessor,
		textInputNames:        []string{cfg.textInputIDsName, cfg.textAttentionMask},
		textOutputNames:       []string{cfg.textOutputName},
		visionInputNames:      []string{cfg.visionInputName},
		visionOutputNames:     []string{cfg.visionOutputName},
		textSessionsByBatch:   make(map[int]*textEmbeddingSession),
		textSessionLRU:        list.New(),
		textSessionLRUIndex:   make(map[int]*list.Element),
		visionSessionsByBatch: make(map[int]*visionEmbeddingSession),
		visionSessionLRU:      list.New(),
		visionSessionLRUIndex: make(map[int]*list.Element),
		maxCachedBatchCount:   cfg.maxCachedBatchCount,
	}, nil
}

func validateFilePath(label string, path string) error {
	if path == "" {
		return fmt.Errorf("%s path cannot be empty", label)
	}
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("%s path %q is not usable: %w", label, path, err)
	}
	if info.IsDir() {
		return fmt.Errorf("%s path %q is a directory, expected a file", label, path)
	}
	return nil
}

// Close releases ONNX session resources and tokenizer resources.
func (e *Embedder) Close() error {
	if e == nil {
		return nil
	}

	e.runMu.Lock()
	defer e.runMu.Unlock()

	var err error

	for batchSize, session := range e.textSessionsByBatch {
		if destroyErr := session.Destroy(); destroyErr != nil {
			err = errors.Join(err, fmt.Errorf("failed to destroy text batch-%d resources: %w", batchSize, destroyErr))
		}
	}
	for batchSize, session := range e.visionSessionsByBatch {
		if destroyErr := session.Destroy(); destroyErr != nil {
			err = errors.Join(err, fmt.Errorf("failed to destroy vision batch-%d resources: %w", batchSize, destroyErr))
		}
	}

	e.textSessionsByBatch = nil
	e.textSessionLRU = nil
	e.textSessionLRUIndex = nil
	e.visionSessionsByBatch = nil
	e.visionSessionLRU = nil
	e.visionSessionLRUIndex = nil

	if e.tokenizer != nil {
		if closeErr := e.tokenizer.Close(); closeErr != nil {
			err = errors.Join(err, closeErr)
		}
		e.tokenizer = nil
	}

	return err
}

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

	session, err := e.textSessionForBatchLocked(len(texts))
	if err != nil {
		return nil, err
	}

	if err := e.tokenizeTextsInto(texts, session.inputIDs, session.attentionMask); err != nil {
		return nil, err
	}

	if err := session.session.Run(); err != nil {
		return nil, fmt.Errorf("text embedding inference failed: %w", err)
	}

	return postProcessDenseEmbeddings(
		session.outputTensor.GetData(),
		len(texts),
		e.embeddingDimension,
		e.l2Normalize,
		"text output",
	)
}

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

// EmbedImages embeds input images with the CLIP vision encoder.
func (e *Embedder) EmbedImages(images []image.Image) (_ [][]float32, err error) {
	if e == nil {
		return nil, fmt.Errorf("embedder is nil")
	}
	if len(images) == 0 {
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

	session, err := e.visionSessionForBatchLocked(len(images))
	if err != nil {
		return nil, err
	}
	if err := e.preprocessImagesInto(images, session.pixelValues); err != nil {
		return nil, err
	}

	if err := session.session.Run(); err != nil {
		return nil, fmt.Errorf("vision embedding inference failed: %w", err)
	}

	return postProcessDenseEmbeddings(
		session.outputTensor.GetData(),
		len(images),
		e.embeddingDimension,
		e.l2Normalize,
		"vision output",
	)
}

// EmbedImage embeds a single image with the CLIP vision encoder.
func (e *Embedder) EmbedImage(img image.Image) ([]float32, error) {
	embeddings, err := e.EmbedImages([]image.Image{img})
	if err != nil {
		return nil, err
	}
	if len(embeddings) != 1 {
		return nil, fmt.Errorf("unexpected embedding row count: got %d, want 1", len(embeddings))
	}
	return embeddings[0], nil
}

func (e *Embedder) textSessionForBatchLocked(batchSize int) (_ *textEmbeddingSession, err error) {
	if batchSize <= 0 {
		return nil, fmt.Errorf("batch size must be > 0, got %d", batchSize)
	}

	if session, ok := e.textSessionsByBatch[batchSize]; ok {
		e.touchTextBatchSizeLocked(batchSize)
		return session, nil
	}
	if e.maxCachedBatchCount > 0 && len(e.textSessionsByBatch) >= e.maxCachedBatchCount {
		if err := e.evictLeastRecentlyUsedTextSessionLocked(); err != nil {
			return nil, err
		}
	}

	session, err := newTextEmbeddingSession(
		e.textModelPath,
		e.textInputNames,
		e.textOutputNames,
		e.sequenceLength,
		batchSize,
		e.embeddingDimension,
	)
	if err != nil {
		return nil, err
	}
	e.textSessionsByBatch[batchSize] = session
	e.touchTextBatchSizeLocked(batchSize)
	return session, nil
}

func (e *Embedder) touchTextBatchSizeLocked(batchSize int) {
	if existing := e.textSessionLRUIndex[batchSize]; existing != nil {
		e.textSessionLRU.MoveToBack(existing)
		return
	}
	e.textSessionLRUIndex[batchSize] = e.textSessionLRU.PushBack(batchSize)
}

func (e *Embedder) evictLeastRecentlyUsedTextSessionLocked() error {
	if e.textSessionLRU == nil {
		return nil
	}
	oldest := e.textSessionLRU.Front()
	if oldest == nil {
		return nil
	}
	batchSize, ok := oldest.Value.(int)
	if !ok {
		return fmt.Errorf("invalid text cache bookkeeping value: %T", oldest.Value)
	}
	session := e.textSessionsByBatch[batchSize]
	delete(e.textSessionsByBatch, batchSize)
	delete(e.textSessionLRUIndex, batchSize)
	e.textSessionLRU.Remove(oldest)
	if session == nil {
		return nil
	}
	if err := session.Destroy(); err != nil {
		return fmt.Errorf("failed to evict text batch-%d resources: %w", batchSize, err)
	}
	return nil
}

func (e *Embedder) visionSessionForBatchLocked(batchSize int) (_ *visionEmbeddingSession, err error) {
	if batchSize <= 0 {
		return nil, fmt.Errorf("batch size must be > 0, got %d", batchSize)
	}

	if session, ok := e.visionSessionsByBatch[batchSize]; ok {
		e.touchVisionBatchSizeLocked(batchSize)
		return session, nil
	}
	if e.maxCachedBatchCount > 0 && len(e.visionSessionsByBatch) >= e.maxCachedBatchCount {
		if err := e.evictLeastRecentlyUsedVisionSessionLocked(); err != nil {
			return nil, err
		}
	}

	session, err := newVisionEmbeddingSession(
		e.visionModelPath,
		e.visionInputNames,
		e.visionOutputNames,
		e.imageSize,
		batchSize,
		e.embeddingDimension,
	)
	if err != nil {
		return nil, err
	}
	e.visionSessionsByBatch[batchSize] = session
	e.touchVisionBatchSizeLocked(batchSize)
	return session, nil
}

func (e *Embedder) touchVisionBatchSizeLocked(batchSize int) {
	if existing := e.visionSessionLRUIndex[batchSize]; existing != nil {
		e.visionSessionLRU.MoveToBack(existing)
		return
	}
	e.visionSessionLRUIndex[batchSize] = e.visionSessionLRU.PushBack(batchSize)
}

func (e *Embedder) evictLeastRecentlyUsedVisionSessionLocked() error {
	if e.visionSessionLRU == nil {
		return nil
	}
	oldest := e.visionSessionLRU.Front()
	if oldest == nil {
		return nil
	}
	batchSize, ok := oldest.Value.(int)
	if !ok {
		return fmt.Errorf("invalid vision cache bookkeeping value: %T", oldest.Value)
	}
	session := e.visionSessionsByBatch[batchSize]
	delete(e.visionSessionsByBatch, batchSize)
	delete(e.visionSessionLRUIndex, batchSize)
	e.visionSessionLRU.Remove(oldest)
	if session == nil {
		return nil
	}
	if err := session.Destroy(); err != nil {
		return fmt.Errorf("failed to evict vision batch-%d resources: %w", batchSize, err)
	}
	return nil
}

func newTextEmbeddingSession(modelPath string, inputNames []string, outputNames []string, sequenceLength int, batchSize int, embeddingDimension int64) (_ *textEmbeddingSession, err error) {
	totalTokens := batchSize * sequenceLength
	inputIDs := make([]int64, totalTokens)
	attentionMask := make([]int64, totalTokens)

	shape := ort.Shape{int64(batchSize), int64(sequenceLength)}
	inputIDsTensor, err := ort.NewTensor[int64](shape, inputIDs)
	if err != nil {
		return nil, fmt.Errorf("failed to create text input_ids tensor: %w", err)
	}
	attentionMaskTensor, err := ort.NewTensor[int64](shape, attentionMask)
	if err != nil {
		cleanupErr := ortutil.DestroyAll(inputIDsTensor)
		if cleanupErr != nil {
			return nil, errors.Join(fmt.Errorf("failed to create text attention_mask tensor: %w", err), fmt.Errorf("failed to clean up text session tensors: %w", cleanupErr))
		}
		return nil, fmt.Errorf("failed to create text attention_mask tensor: %w", err)
	}

	outputShape := ort.Shape{int64(batchSize), embeddingDimension}
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		cleanupErr := ortutil.DestroyAll(attentionMaskTensor, inputIDsTensor)
		if cleanupErr != nil {
			return nil, errors.Join(fmt.Errorf("failed to create text output tensor: %w", err), fmt.Errorf("failed to clean up text session tensors: %w", cleanupErr))
		}
		return nil, fmt.Errorf("failed to create text output tensor: %w", err)
	}

	session, err := ort.NewAdvancedSession(
		modelPath,
		inputNames,
		outputNames,
		[]ort.Value{inputIDsTensor, attentionMaskTensor},
		[]ort.Value{outputTensor},
		nil,
	)
	if err != nil {
		cleanupErr := ortutil.DestroyAll(outputTensor, attentionMaskTensor, inputIDsTensor)
		if cleanupErr != nil {
			return nil, errors.Join(fmt.Errorf("failed to create text embedding session: %w", err), fmt.Errorf("failed to clean up text session tensors: %w", cleanupErr))
		}
		return nil, fmt.Errorf("failed to create text embedding session: %w", err)
	}

	return &textEmbeddingSession{
		inputIDs:            inputIDs,
		attentionMask:       attentionMask,
		inputIDsTensor:      inputIDsTensor,
		attentionMaskTensor: attentionMaskTensor,
		outputTensor:        outputTensor,
		session:             session,
	}, nil
}

func newVisionEmbeddingSession(modelPath string, inputNames []string, outputNames []string, imageSize int, batchSize int, embeddingDimension int64) (_ *visionEmbeddingSession, err error) {
	pixelsPerImage := 3 * imageSize * imageSize
	pixelValues := make([]float32, batchSize*pixelsPerImage)

	inputShape := ort.Shape{int64(batchSize), 3, int64(imageSize), int64(imageSize)}
	pixelValuesTensor, err := ort.NewTensor[float32](inputShape, pixelValues)
	if err != nil {
		return nil, fmt.Errorf("failed to create vision pixel_values tensor: %w", err)
	}

	outputShape := ort.Shape{int64(batchSize), embeddingDimension}
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		cleanupErr := ortutil.DestroyAll(pixelValuesTensor)
		if cleanupErr != nil {
			return nil, errors.Join(fmt.Errorf("failed to create vision output tensor: %w", err), fmt.Errorf("failed to clean up vision session tensors: %w", cleanupErr))
		}
		return nil, fmt.Errorf("failed to create vision output tensor: %w", err)
	}

	session, err := ort.NewAdvancedSession(
		modelPath,
		inputNames,
		outputNames,
		[]ort.Value{pixelValuesTensor},
		[]ort.Value{outputTensor},
		nil,
	)
	if err != nil {
		cleanupErr := ortutil.DestroyAll(outputTensor, pixelValuesTensor)
		if cleanupErr != nil {
			return nil, errors.Join(fmt.Errorf("failed to create vision embedding session: %w", err), fmt.Errorf("failed to clean up vision session tensors: %w", cleanupErr))
		}
		return nil, fmt.Errorf("failed to create vision embedding session: %w", err)
	}

	return &visionEmbeddingSession{
		pixelValues:       pixelValues,
		pixelValuesTensor: pixelValuesTensor,
		outputTensor:      outputTensor,
		session:           session,
	}, nil
}

func (s *textEmbeddingSession) Destroy() error {
	if s == nil {
		return nil
	}

	err := ortutil.DestroyAll(
		s.session,
		s.outputTensor,
		s.attentionMaskTensor,
		s.inputIDsTensor,
	)

	s.inputIDs = nil
	s.attentionMask = nil
	s.session = nil
	s.outputTensor = nil
	s.attentionMaskTensor = nil
	s.inputIDsTensor = nil
	return err
}

func (s *visionEmbeddingSession) Destroy() error {
	if s == nil {
		return nil
	}

	err := ortutil.DestroyAll(
		s.session,
		s.outputTensor,
		s.pixelValuesTensor,
	)

	s.pixelValues = nil
	s.session = nil
	s.outputTensor = nil
	s.pixelValuesTensor = nil
	return err
}

func (e *Embedder) tokenizeTextsInto(texts []string, inputIDs []int64, attentionMask []int64) error {
	sequenceLength := e.sequenceLength
	batchSize := len(texts)
	totalTokens := batchSize * sequenceLength

	if len(inputIDs) != totalTokens || len(attentionMask) != totalTokens {
		return fmt.Errorf(
			"text token buffer length mismatch: got input_ids=%d attention_mask=%d, want %d",
			len(inputIDs),
			len(attentionMask),
			totalTokens,
		)
	}

	clear(inputIDs)
	clear(attentionMask)

	for i, text := range texts {
		encoding, err := e.tokenizer.Encode(
			text,
			tokenizers.WithAddSpecialTokens(),
			tokenizers.WithReturnAttentionMask(),
		)
		if err != nil {
			return fmt.Errorf("failed to tokenize text %d: %w", i, err)
		}
		if encoding == nil {
			return fmt.Errorf("failed to tokenize text %d: empty tokenizer result", i)
		}

		rowStart := i * sequenceLength
		rowEnd := rowStart + sequenceLength
		fillUint32AsInt64(inputIDs[rowStart:rowEnd], encoding.IDs)

		if len(encoding.AttentionMask) > 0 {
			fillUint32AsInt64(attentionMask[rowStart:rowEnd], encoding.AttentionMask)
		} else {
			deriveAttentionMask(attentionMask[rowStart:rowEnd], inputIDs[rowStart:rowEnd])
		}
	}

	return nil
}

func (e *Embedder) preprocessImagesInto(images []image.Image, pixelValues []float32) error {
	imageSize := e.imageSize
	if imageSize <= 0 {
		return fmt.Errorf("image size must be > 0, got %d", imageSize)
	}

	pixelsPerImage := 3 * imageSize * imageSize
	if len(pixelValues) != len(images)*pixelsPerImage {
		return fmt.Errorf("pixel_values buffer length mismatch: got %d, want %d", len(pixelValues), len(images)*pixelsPerImage)
	}

	clear(pixelValues)
	for i, img := range images {
		if img == nil {
			return fmt.Errorf("image %d is nil", i)
		}

		processed := img
		if e.preprocessor.doResize {
			processed = resizeKeepingAspect(processed, e.preprocessor.resizeShortestEdge)
		}
		if e.preprocessor.doCenterCrop {
			processed = centerCropImage(processed, e.preprocessor.cropSize, e.preprocessor.cropSize)
		}
		if processed.Bounds().Dx() != imageSize || processed.Bounds().Dy() != imageSize {
			processed = resizeImage(processed, imageSize, imageSize)
		}

		imageOffset := i * pixelsPerImage
		plane := imageSize * imageSize
		for y := 0; y < imageSize; y++ {
			for x := 0; x < imageSize; x++ {
				r, g, b := getRGB(processed.At(processed.Bounds().Min.X+x, processed.Bounds().Min.Y+y), e.preprocessor.doConvertRGB)
				if e.preprocessor.doRescale {
					r *= e.preprocessor.rescaleFactor
					g *= e.preprocessor.rescaleFactor
					b *= e.preprocessor.rescaleFactor
				}
				if e.preprocessor.doNormalize {
					r = (r - e.preprocessor.imageMean[0]) / e.preprocessor.imageStd[0]
					g = (g - e.preprocessor.imageMean[1]) / e.preprocessor.imageStd[1]
					b = (b - e.preprocessor.imageMean[2]) / e.preprocessor.imageStd[2]
				}

				pixelIndex := y*imageSize + x
				pixelValues[imageOffset+pixelIndex] = r
				pixelValues[imageOffset+plane+pixelIndex] = g
				pixelValues[imageOffset+(2*plane)+pixelIndex] = b
			}
		}
	}

	return nil
}

func getRGB(c color.Color, convertRGB bool) (float32, float32, float32) {
	if !convertRGB {
		gray := color.GrayModel.Convert(c).(color.Gray)
		v := float32(gray.Y)
		return v, v, v
	}
	rgba := color.NRGBAModel.Convert(c).(color.NRGBA)
	return float32(rgba.R), float32(rgba.G), float32(rgba.B)
}

func resizeKeepingAspect(src image.Image, shortestEdge int) image.Image {
	if shortestEdge <= 0 {
		return src
	}
	bounds := src.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()
	if width <= 0 || height <= 0 {
		return src
	}
	if min(width, height) == shortestEdge {
		return src
	}

	var newWidth, newHeight int
	if width < height {
		newWidth = shortestEdge
		newHeight = int(math.Round(float64(height) * float64(shortestEdge) / float64(width)))
	} else {
		newHeight = shortestEdge
		newWidth = int(math.Round(float64(width) * float64(shortestEdge) / float64(height)))
	}
	if newWidth <= 0 {
		newWidth = 1
	}
	if newHeight <= 0 {
		newHeight = 1
	}
	return resizeImage(src, newWidth, newHeight)
}

func centerCropImage(src image.Image, targetWidth int, targetHeight int) image.Image {
	if targetWidth <= 0 || targetHeight <= 0 {
		return src
	}
	srcBounds := src.Bounds()
	srcWidth := srcBounds.Dx()
	srcHeight := srcBounds.Dy()
	if srcWidth <= 0 || srcHeight <= 0 {
		return src
	}

	cropWidth := min(targetWidth, srcWidth)
	cropHeight := min(targetHeight, srcHeight)
	srcStartX := srcBounds.Min.X + (srcWidth-cropWidth)/2
	srcStartY := srcBounds.Min.Y + (srcHeight-cropHeight)/2

	dst := image.NewNRGBA(image.Rect(0, 0, targetWidth, targetHeight))
	dstOffsetX := (targetWidth - cropWidth) / 2
	dstOffsetY := (targetHeight - cropHeight) / 2

	for y := 0; y < cropHeight; y++ {
		for x := 0; x < cropWidth; x++ {
			dst.SetNRGBA(dstOffsetX+x, dstOffsetY+y, color.NRGBAModel.Convert(src.At(srcStartX+x, srcStartY+y)).(color.NRGBA))
		}
	}
	return dst
}

func resizeImage(src image.Image, targetWidth int, targetHeight int) *image.NRGBA {
	if targetWidth <= 0 || targetHeight <= 0 {
		return image.NewNRGBA(image.Rect(0, 0, 1, 1))
	}
	srcBounds := src.Bounds()
	srcWidth := srcBounds.Dx()
	srcHeight := srcBounds.Dy()
	if srcWidth <= 0 || srcHeight <= 0 {
		return image.NewNRGBA(image.Rect(0, 0, targetWidth, targetHeight))
	}

	dst := image.NewNRGBA(image.Rect(0, 0, targetWidth, targetHeight))
	for y := 0; y < targetHeight; y++ {
		srcY := int((float64(y)+0.5)*float64(srcHeight)/float64(targetHeight) - 0.5)
		if srcY < 0 {
			srcY = 0
		}
		if srcY >= srcHeight {
			srcY = srcHeight - 1
		}
		for x := 0; x < targetWidth; x++ {
			srcX := int((float64(x)+0.5)*float64(srcWidth)/float64(targetWidth) - 0.5)
			if srcX < 0 {
				srcX = 0
			}
			if srcX >= srcWidth {
				srcX = srcWidth - 1
			}
			dst.SetNRGBA(x, y, color.NRGBAModel.Convert(src.At(srcBounds.Min.X+srcX, srcBounds.Min.Y+srcY)).(color.NRGBA))
		}
	}
	return dst
}

func postProcessDenseEmbeddings(output []float32, batchSize int, embeddingDimension int64, l2Normalize bool, label string) ([][]float32, error) {
	if batchSize <= 0 {
		return nil, fmt.Errorf("batch size must be > 0, got %d", batchSize)
	}
	if embeddingDimension <= 0 {
		return nil, fmt.Errorf("embedding dimension must be > 0, got %d", embeddingDimension)
	}

	embeddingWidth := int(embeddingDimension)
	expectedLen := batchSize * embeddingWidth
	if len(output) != expectedLen {
		return nil, fmt.Errorf("%s length mismatch: got %d, want %d", label, len(output), expectedLen)
	}

	embeddings := make([][]float32, batchSize)
	for i := 0; i < batchSize; i++ {
		rowStart := i * embeddingWidth
		rowEnd := rowStart + embeddingWidth
		row := make([]float32, embeddingWidth)
		copy(row, output[rowStart:rowEnd])
		if l2Normalize {
			l2NormalizeInPlace(row)
		}
		embeddings[i] = row
	}
	return embeddings, nil
}

func l2NormalizeInPlace(values []float32) {
	var normSquared float64
	for i := range values {
		normSquared += float64(values[i]) * float64(values[i])
	}
	norm := float32(math.Sqrt(normSquared))
	if norm <= l2NormEpsilon {
		return
	}
	for i := range values {
		values[i] /= norm
	}
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

type preprocessorConfigFile struct {
	DoResize     *bool           `json:"do_resize"`
	Size         json.RawMessage `json:"size"`
	DoCenterCrop *bool           `json:"do_center_crop"`
	CropSize     json.RawMessage `json:"crop_size"`
	DoNormalize  *bool           `json:"do_normalize"`
	ImageMean    []float64       `json:"image_mean"`
	ImageStd     []float64       `json:"image_std"`
	DoRescale    *bool           `json:"do_rescale"`
	RescaleFact  *float64        `json:"rescale_factor"`
	DoConvertRGB *bool           `json:"do_convert_rgb"`
}

func loadPreprocessorConfig(path string) (clipPreprocessorConfig, error) {
	// #nosec G304 -- path is validated by NewEmbedder before this call.
	bytes, err := os.ReadFile(path)
	if err != nil {
		return clipPreprocessorConfig{}, err
	}

	parsed := defaultClipPreprocessorConfig()
	var raw preprocessorConfigFile
	if err := json.Unmarshal(bytes, &raw); err != nil {
		return clipPreprocessorConfig{}, fmt.Errorf("failed to parse JSON: %w", err)
	}

	if raw.DoResize != nil {
		parsed.doResize = *raw.DoResize
	}
	if raw.DoCenterCrop != nil {
		parsed.doCenterCrop = *raw.DoCenterCrop
	}
	if raw.DoNormalize != nil {
		parsed.doNormalize = *raw.DoNormalize
	}
	if raw.DoRescale != nil {
		parsed.doRescale = *raw.DoRescale
	}
	if raw.DoConvertRGB != nil {
		parsed.doConvertRGB = *raw.DoConvertRGB
	}
	if raw.RescaleFact != nil {
		if *raw.RescaleFact <= 0 {
			return clipPreprocessorConfig{}, fmt.Errorf("rescale_factor must be > 0, got %f", *raw.RescaleFact)
		}
		parsed.rescaleFactor = float32(*raw.RescaleFact)
		parsed.doRescale = true
	}

	size, err := parseSizeField(raw.Size)
	if err != nil {
		return clipPreprocessorConfig{}, fmt.Errorf("invalid size field: %w", err)
	}
	if size > 0 {
		parsed.resizeShortestEdge = size
	}
	cropSize, err := parseSizeField(raw.CropSize)
	if err != nil {
		return clipPreprocessorConfig{}, fmt.Errorf("invalid crop_size field: %w", err)
	}
	if cropSize > 0 {
		parsed.cropSize = cropSize
	}

	if len(raw.ImageMean) > 0 {
		if len(raw.ImageMean) != 3 {
			return clipPreprocessorConfig{}, fmt.Errorf("image_mean must have exactly 3 entries, got %d", len(raw.ImageMean))
		}
		for i := range raw.ImageMean {
			parsed.imageMean[i] = float32(raw.ImageMean[i])
		}
	}
	if len(raw.ImageStd) > 0 {
		if len(raw.ImageStd) != 3 {
			return clipPreprocessorConfig{}, fmt.Errorf("image_std must have exactly 3 entries, got %d", len(raw.ImageStd))
		}
		for i := range raw.ImageStd {
			if raw.ImageStd[i] == 0 {
				return clipPreprocessorConfig{}, fmt.Errorf("image_std[%d] must be non-zero", i)
			}
			parsed.imageStd[i] = float32(raw.ImageStd[i])
		}
	}

	return parsed, nil
}

func parseSizeField(raw json.RawMessage) (int, error) {
	if len(raw) == 0 {
		return 0, nil
	}

	var asInt int
	if err := json.Unmarshal(raw, &asInt); err == nil {
		if asInt <= 0 {
			return 0, fmt.Errorf("size value must be > 0, got %d", asInt)
		}
		return asInt, nil
	}

	var asMap map[string]int
	if err := json.Unmarshal(raw, &asMap); err != nil {
		return 0, fmt.Errorf("size must be an integer or object, got %s", string(raw))
	}

	if value, ok := asMap["shortest_edge"]; ok {
		if value <= 0 {
			return 0, fmt.Errorf("shortest_edge must be > 0, got %d", value)
		}
		return value, nil
	}

	height, hasHeight := asMap["height"]
	width, hasWidth := asMap["width"]
	if hasHeight && hasWidth {
		if height <= 0 || width <= 0 {
			return 0, fmt.Errorf("height and width must be > 0, got height=%d width=%d", height, width)
		}
		if height != width {
			return 0, fmt.Errorf("non-square crop size is unsupported: height=%d width=%d", height, width)
		}
		return height, nil
	}

	return 0, fmt.Errorf("size object must include shortest_edge or height/width")
}

// CosineSimilarity returns cosine similarity between two vectors.
func CosineSimilarity(a []float32, b []float32) (float32, error) {
	if len(a) == 0 || len(b) == 0 {
		return 0, fmt.Errorf("vectors must be non-empty")
	}
	if len(a) != len(b) {
		return 0, fmt.Errorf("vector dimension mismatch: got %d and %d", len(a), len(b))
	}

	var dot float64
	var normA float64
	var normB float64
	for i := range a {
		av := float64(a[i])
		bv := float64(b[i])
		dot += av * bv
		normA += av * av
		normB += bv * bv
	}
	if normA == 0 || normB == 0 {
		return 0, nil
	}
	return float32(dot / (math.Sqrt(normA) * math.Sqrt(normB))), nil
}

// CLIPSimilarityLogits computes image-text similarity logits using cosine similarity and logit scale.
func CLIPSimilarityLogits(imageEmbeddings [][]float32, textEmbeddings [][]float32, logitScale float32) ([][]float32, error) {
	if len(imageEmbeddings) == 0 || len(textEmbeddings) == 0 {
		return [][]float32{}, nil
	}
	if !isFiniteFloat32(logitScale) {
		return nil, fmt.Errorf("logit scale must be finite, got %f", logitScale)
	}

	imageDim, err := validateEmbeddingRows(imageEmbeddings, "image embeddings")
	if err != nil {
		return nil, err
	}
	textDim, err := validateEmbeddingRows(textEmbeddings, "text embeddings")
	if err != nil {
		return nil, err
	}
	if imageDim != textDim {
		return nil, fmt.Errorf("embedding dimension mismatch: image=%d text=%d", imageDim, textDim)
	}

	logits := make([][]float32, len(imageEmbeddings))
	for i := range imageEmbeddings {
		row := make([]float32, len(textEmbeddings))
		for j := range textEmbeddings {
			cosine, cosErr := CosineSimilarity(imageEmbeddings[i], textEmbeddings[j])
			if cosErr != nil {
				return nil, fmt.Errorf("failed to compute similarity for image row %d and text row %d: %w", i, j, cosErr)
			}
			row[j] = cosine * logitScale
		}
		logits[i] = row
	}
	return logits, nil
}

func validateEmbeddingRows(rows [][]float32, label string) (int, error) {
	if len(rows) == 0 {
		return 0, fmt.Errorf("%s cannot be empty", label)
	}
	expectedDim := len(rows[0])
	if expectedDim == 0 {
		return 0, fmt.Errorf("%s row 0 is empty", label)
	}
	for i := range rows {
		if len(rows[i]) != expectedDim {
			return 0, fmt.Errorf("%s row %d has dimension %d, want %d", label, i, len(rows[i]), expectedDim)
		}
	}
	return expectedDim, nil
}

func isFiniteFloat32(v float32) bool {
	return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0)
}
