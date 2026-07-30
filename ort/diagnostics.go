package ort

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"sync"
	"sync/atomic"
)

type diagnosticState struct {
	logger *slog.Logger
}

var diagnostics = newDiagnosticStore()
var emergencyDiagnosticMu sync.Mutex

func newDiagnosticStore() *atomic.Pointer[diagnosticState] {
	store := &atomic.Pointer[diagnosticState]{}
	store.Store(&diagnosticState{logger: slog.New(newDefaultDiagnosticHandler())})
	return store
}

func newDefaultDiagnosticHandler() slog.Handler {
	return slog.NewTextHandler(
		os.Stderr,
		&slog.HandlerOptions{Level: slog.LevelWarn},
	)
}

// SetDiagnosticHandler installs the process-wide diagnostic handler.
// The handler is trusted synchronous consumer code and must be safe for
// concurrent use. Its panics propagate to the caller except at the
// best-effort finalizer boundary. Handlers may call read-only runtime queries,
// including IsInitialized and GetVersionString. They must not call lifecycle
// mutation or bootstrap APIs because bootstrap diagnostics may be emitted while
// an interprocess cache lock is held. Passing nil restores the default text
// handler, which writes warning-level diagnostics to the current os.Stderr.
func SetDiagnosticHandler(handler slog.Handler) {
	if handler == nil {
		handler = newDefaultDiagnosticHandler()
	}
	diagnostics.Store(&diagnosticState{logger: slog.New(handler)})
}

func emitDiagnostic(
	ctx context.Context,
	level slog.Level,
	message string,
	attrs ...slog.Attr,
) {
	if ctx == nil {
		ctx = context.Background()
	}
	diagnostics.Load().logger.LogAttrs(ctx, level, message, attrs...)
}

func emitEmergencyDiagnostic(message, resource string, cleanupFailure, handlerPanic any) {
	defer func() {
		_ = recover()
	}()

	emergencyDiagnosticMu.Lock()
	defer emergencyDiagnosticMu.Unlock()

	_, _ = fmt.Fprintf(
		os.Stderr,
		"onnx-purego emergency diagnostic: %s resource=%q cleanup_failure=%q handler_panic=%q\n",
		message,
		resource,
		cleanupFailure,
		handlerPanic,
	)
}

func emitFinalizerDiagnostic(resource string, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			emitEmergencyDiagnostic(
				"diagnostic handler panicked during finalizer cleanup",
				resource,
				err,
				recovered,
			)
		}
	}()

	emitDiagnostic(
		context.Background(),
		slog.LevelWarn,
		"finalizer cleanup failed",
		slog.String("resource", resource),
		slog.Any("error", err),
	)
}
