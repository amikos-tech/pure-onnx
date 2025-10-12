package ort

// OrtApiBase represents the base API structure
type OrtApiBase struct {
	GetApi           uintptr
	GetVersionString uintptr
}

// OrtApi represents the ONNX Runtime C API function pointers
type OrtApi struct {
	CreateStatus    uintptr
	GetErrorCode    uintptr
	GetErrorMessage uintptr

	CreateEnv                 uintptr
	CreateEnvWithCustomLogger uintptr
	EnableTelemetryEvents     uintptr
	DisableTelemetryEvents    uintptr

	CreateSession          uintptr
	CreateSessionFromArray uintptr
	Run                    uintptr

	CreateSessionOptions             uintptr
	SetOptimizedModelFilePath        uintptr
	CloneSessionOptions              uintptr
	SetSessionExecutionMode          uintptr
	EnableProfiling                  uintptr
	DisableProfiling                 uintptr
	EnableMemPattern                 uintptr
	DisableMemPattern                uintptr
	EnableCpuMemArena                uintptr
	DisableCpuMemArena               uintptr
	SetSessionLogId                  uintptr
	SetSessionLogVerbosityLevel      uintptr
	SetSessionLogSeverityLevel       uintptr
	SetSessionGraphOptimizationLevel uintptr
	SetIntraOpNumThreads             uintptr
	SetInterOpNumThreads             uintptr

	CreateCustomOpDomain     uintptr
	CustomOpDomain_Add       uintptr
	AddCustomOpDomain        uintptr
	RegisterCustomOpsLibrary uintptr

	SessionGetInputCount                     uintptr
	SessionGetOutputCount                    uintptr
	SessionGetOverridableInitializerCount    uintptr
	SessionGetInputTypeInfo                  uintptr
	SessionGetOutputTypeInfo                 uintptr
	SessionGetOverridableInitializerTypeInfo uintptr
	SessionGetInputName                      uintptr
	SessionGetOutputName                     uintptr
	SessionGetOverridableInitializerName     uintptr

	CreateRunOptions                  uintptr
	RunOptionsSetRunLogVerbosityLevel uintptr
	RunOptionsSetRunLogSeverityLevel  uintptr
	RunOptionsSetRunTag               uintptr
	RunOptionsGetRunLogVerbosityLevel uintptr
	RunOptionsGetRunLogSeverityLevel  uintptr
	RunOptionsGetRunTag               uintptr
	RunOptionsSetTerminate            uintptr
	RunOptionsUnsetTerminate          uintptr

	CreateTensorAsOrtValue         uintptr
	CreateTensorWithDataAsOrtValue uintptr
	IsTensor                       uintptr
	GetTensorMutableData           uintptr

	FillStringTensor          uintptr
	GetStringTensorDataLength uintptr
	GetStringTensorContent    uintptr

	CastTypeInfoToTensorInfo     uintptr
	GetOnnxTypeFromTypeInfo      uintptr
	CreateTensorTypeAndShapeInfo uintptr
	SetTensorElementType         uintptr

	SetDimensions              uintptr
	GetTensorElementType       uintptr
	GetDimensionsCount         uintptr
	GetDimensions              uintptr
	GetSymbolicDimensions      uintptr
	GetTensorShapeElementCount uintptr
	GetTensorTypeAndShape      uintptr
	GetTypeInfo                uintptr
	GetValueType               uintptr
	CreateMemoryInfo           uintptr
	CreateCpuMemoryInfo        uintptr
	CompareMemoryInfo          uintptr
	MemoryInfoGetName          uintptr
	MemoryInfoGetId            uintptr
	MemoryInfoGetMemType       uintptr
	MemoryInfoGetType          uintptr

	ReleaseEnv                    uintptr
	ReleaseStatus                 uintptr
	ReleaseMemoryInfo             uintptr
	ReleaseSession                uintptr
	ReleaseValue                  uintptr
	ReleaseRunOptions             uintptr
	ReleaseTypeInfo               uintptr
	ReleaseTensorTypeAndShapeInfo uintptr
	ReleaseSessionOptions         uintptr
	ReleaseCustomOpDomain         uintptr

	// Additional function pointers would be added here as needed
	// See internal/c_api/ort_apis.h for full list
}

// Status represents an ONNX Runtime status
type Status uintptr

// Environment represents an ONNX Runtime environment
type Environment uintptr

// Session represents an ONNX Runtime session
type Session uintptr

// Value represents an ONNX Runtime value (tensor, sequence, map, etc.)
type Value interface {
	// Destroy releases the underlying resources
	Destroy() error
	// Type returns the type of the value
	Type() ValueType
}

// ValueType represents the type of an ONNX Runtime value
type ValueType int

const (
	ValueTypeUnknown ValueType = iota
	ValueTypeTensor
	ValueTypeSequence
	ValueTypeMap
	ValueTypeOpaque
	ValueTypeOptional
)

// Shape represents the shape of a tensor
type Shape []int64

// NewShape creates a new shape from dimensions
func NewShape(dims ...int64) Shape {
	return Shape(dims)
}

// SessionOptions represents options for creating a session
type SessionOptions struct {
	// TODO: Add session options fields as per issue #4
}

// MemoryInfo represents memory allocation information
type MemoryInfo struct {
	// TODO: Add memory info fields as per issue #5
}