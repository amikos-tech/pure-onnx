package customdiagnosticsink

import (
	"context"
	"sync/atomic"
)

// Level is the severity of a non-returnable library diagnostic.
type Level uint8

const (
	LevelInfo Level = iota
	LevelWarn
)

// Field is one structured key-value pair attached to a diagnostic.
type Field struct {
	Key   string
	Value any
}

// DiagnosticSink receives non-returnable library diagnostics.
//
// Implementations must be safe for concurrent use.
type DiagnosticSink interface {
	Log(context.Context, Level, string, ...Field)
}

type discardSink struct{}

func (discardSink) Log(context.Context, Level, string, ...Field) {}

type sinkState struct {
	sink DiagnosticSink
}

var configuredSink = newSinkStore()

func newSinkStore() *atomic.Pointer[sinkState] {
	store := &atomic.Pointer[sinkState]{}
	store.Store(&sinkState{sink: discardSink{}})
	return store
}

// SetDiagnosticSink configures the process-wide diagnostic sink. Passing nil
// restores silent behavior.
func SetDiagnosticSink(sink DiagnosticSink) {
	if sink == nil {
		sink = discardSink{}
	}
	configuredSink.Store(&sinkState{sink: sink})
}

// emitDiagnostic is deliberately private: callers cannot use the package as a
// general logging facade, and returned failures have no automatic route here.
func emitDiagnostic(ctx context.Context, level Level, message string, fields ...Field) {
	if ctx == nil {
		ctx = context.Background()
	}
	configuredSink.Load().sink.Log(ctx, level, message, fields...)
}
