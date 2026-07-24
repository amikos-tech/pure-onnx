package ort

import (
	"context"
	"log/slog"
	"sync/atomic"
)

type diagnosticState struct {
	logger *slog.Logger
}

var diagnostics = newDiagnosticStore()

func newDiagnosticStore() *atomic.Pointer[diagnosticState] {
	store := &atomic.Pointer[diagnosticState]{}
	store.Store(&diagnosticState{logger: slog.New(slog.DiscardHandler)})
	return store
}

// SetDiagnosticHandler installs the process-wide diagnostic handler.
// The handler is trusted synchronous consumer code and must be safe for
// concurrent use. Its panics propagate to the caller except at the
// best-effort finalizer boundary. Handlers may call read-only runtime queries,
// including IsInitialized and GetVersionString. They must not call lifecycle
// mutation or bootstrap APIs because bootstrap diagnostics may be emitted while
// an interprocess cache lock is held. Passing nil restores silent behavior.
func SetDiagnosticHandler(handler slog.Handler) {
	if handler == nil {
		handler = slog.DiscardHandler
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

func emitFinalizerDiagnostic(resource string, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			_ = recovered
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
