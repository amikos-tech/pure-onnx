package sloghandlersink

import (
	"context"
	"log/slog"
	"sync/atomic"
)

type handlerState struct {
	logger *slog.Logger
}

var configuredHandler = newHandlerStore()

func newHandlerStore() *atomic.Pointer[handlerState] {
	store := &atomic.Pointer[handlerState]{}
	store.Store(&handlerState{logger: slog.New(slog.DiscardHandler)})
	return store
}

// SetDiagnosticHandler configures the process-wide diagnostic handler.
// Passing nil restores silent behavior.
func SetDiagnosticHandler(handler slog.Handler) {
	if handler == nil {
		handler = slog.DiscardHandler
	}
	configuredHandler.Store(&handlerState{logger: slog.New(handler)})
}

// emitDiagnostic is deliberately private: callers cannot use the package as a
// general logging facade, and returned failures have no automatic route here.
func emitDiagnostic(ctx context.Context, level slog.Level, message string, attrs ...slog.Attr) {
	if ctx == nil {
		ctx = context.Background()
	}
	configuredHandler.Load().logger.LogAttrs(ctx, level, message, attrs...)
}
