package ort

import "log"

func logFinalizerWarning(format string, args ...any) {
	// Finalizers may run late during process teardown; guard logging to avoid
	// crashing on best-effort diagnostics.
	defer func() {
		_ = recover()
	}()
	log.Printf(format, args...)
}
