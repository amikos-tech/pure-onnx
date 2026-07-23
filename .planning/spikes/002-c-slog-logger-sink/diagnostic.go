package slogloggersink

import (
	"context"
	"log/slog"
	"sync/atomic"
)

type loggerState struct {
	logger *slog.Logger
}

var configuredLogger = newLoggerStore()

func newLoggerStore() *atomic.Pointer[loggerState] {
	store := &atomic.Pointer[loggerState]{}
	store.Store(&loggerState{logger: slog.New(slog.DiscardHandler)})
	return store
}

// SetDiagnosticLogger configures the process-wide diagnostic logger. Passing
// nil restores silent behavior.
func SetDiagnosticLogger(logger *slog.Logger) {
	if logger == nil {
		logger = slog.New(slog.DiscardHandler)
	}
	configuredLogger.Store(&loggerState{logger: logger})
}

// emitDiagnostic is deliberately private: callers cannot use the package as a
// general logging facade, and returned failures have no automatic route here.
func emitDiagnostic(ctx context.Context, level slog.Level, message string, attrs ...slog.Attr) {
	if ctx == nil {
		ctx = context.Background()
	}
	configuredLogger.Load().logger.LogAttrs(ctx, level, message, attrs...)
}
