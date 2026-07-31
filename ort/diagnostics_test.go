package ort

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

func TestDiagnostic(t *testing.T) {
	t.Run("default warning reaches stderr while info remains quiet", func(t *testing.T) {
		output := captureDiagnosticStderr(t, func() {
			SetDiagnosticHandler(nil)
			emitDiagnostic(context.Background(), slog.LevelInfo, "quiet lock wait")
			emitDiagnostic(context.Background(), slog.LevelWarn, "visible runtime warning")
		})

		if strings.Contains(output, "quiet lock wait") {
			t.Fatalf("default stderr output contains info diagnostic: %q", output)
		}
		if !strings.Contains(output, "visible runtime warning") {
			t.Fatalf("default stderr output = %q, want warning", output)
		}
	})

	t.Run("nil reset restores warning stderr default", func(t *testing.T) {
		handler := &diagnosticCountingHandler{}
		SetDiagnosticHandler(handler)
		t.Cleanup(func() { SetDiagnosticHandler(nil) })

		emitDiagnostic(context.Background(), slog.LevelInfo, "observable")
		if got := handler.count.Load(); got != 1 {
			t.Fatalf("configured handler count: got %d, want 1", got)
		}

		output := captureDiagnosticStderr(t, func() {
			SetDiagnosticHandler(nil)
			emitDiagnostic(context.Background(), slog.LevelWarn, "reset warning")
		})
		if got := handler.count.Load(); got != 1 {
			t.Fatalf("nil-reset handler count: got %d, want 1", got)
		}
		if !strings.Contains(output, "reset warning") {
			t.Fatalf("nil-reset stderr output = %q, want warning", output)
		}
	})

	t.Run("standard JSON handler receives level message and attrs", func(t *testing.T) {
		var output bytes.Buffer
		SetDiagnosticHandler(slog.NewJSONHandler(&output, nil))
		t.Cleanup(func() { SetDiagnosticHandler(nil) })

		//nolint:staticcheck // A nil context intentionally verifies emitDiagnostic's fallback.
		emitDiagnostic(
			nil,
			slog.LevelWarn,
			"runtime fallback selected",
			slog.String("resource", "runtime"),
			slog.Int("attempt", 2),
		)

		record := decodeDiagnosticRecord(t, &output)
		if got := record["msg"]; got != "runtime fallback selected" {
			t.Fatalf("message: got %v, want runtime fallback selected", got)
		}
		if got := record["level"]; got != "WARN" {
			t.Fatalf("level: got %v, want WARN", got)
		}
		if got := record["resource"]; got != "runtime" {
			t.Fatalf("resource: got %v, want runtime", got)
		}
		if got := record["attempt"]; got != float64(2) {
			t.Fatalf("attempt: got %v, want 2", got)
		}
	})

	t.Run("consumer logger attributes survive handler extraction", func(t *testing.T) {
		var output bytes.Buffer
		consumerLogger := slog.New(slog.NewJSONHandler(&output, nil)).
			With(slog.String("component", "inference"))
		SetDiagnosticHandler(consumerLogger.Handler())
		t.Cleanup(func() { SetDiagnosticHandler(nil) })

		emitDiagnostic(
			context.Background(),
			slog.LevelInfo,
			"waiting for bootstrap lock",
			slog.Int("seconds", 2),
		)

		record := decodeDiagnosticRecord(t, &output)
		if got := record["component"]; got != "inference" {
			t.Fatalf("component: got %v, want inference", got)
		}
		if got := record["seconds"]; got != float64(2) {
			t.Fatalf("seconds: got %v, want 2", got)
		}
	})

	t.Run("concurrent configuration and emission delivers every event once", func(t *testing.T) {
		const (
			writers       = 32
			events        = 200
			reconfigurers = 8
		)

		first := &diagnosticCountingHandler{}
		second := &diagnosticCountingHandler{}
		SetDiagnosticHandler(first)
		t.Cleanup(func() { SetDiagnosticHandler(nil) })

		start := make(chan struct{})
		var wg sync.WaitGroup
		for writer := range writers {
			wg.Add(1)
			go func() {
				defer wg.Done()
				<-start
				for sequence := range events {
					emitDiagnostic(
						context.Background(),
						slog.LevelInfo,
						"bootstrap lock wait",
						slog.Int("writer", writer),
						slog.Int("sequence", sequence),
					)
				}
			}()
		}
		for reconfigurer := range reconfigurers {
			wg.Add(1)
			go func() {
				defer wg.Done()
				<-start
				for change := range events {
					if (reconfigurer+change)%2 == 0 {
						SetDiagnosticHandler(first)
						continue
					}
					SetDiagnosticHandler(second)
				}
			}()
		}

		close(start)
		wg.Wait()

		got := first.count.Load() + second.count.Load()
		want := int64(writers * events)
		if got != want {
			t.Fatalf("captured events: got %d, want %d", got, want)
		}
	})

	t.Run("finalizer diagnostic uses structured warning", func(t *testing.T) {
		var output bytes.Buffer
		SetDiagnosticHandler(slog.NewJSONHandler(&output, nil))
		t.Cleanup(func() { SetDiagnosticHandler(nil) })

		emitFinalizerDiagnostic("tensor", errors.New("release failed"))

		record := decodeDiagnosticRecord(t, &output)
		if got := record["level"]; got != "WARN" {
			t.Fatalf("level: got %v, want WARN", got)
		}
		if got := record["resource"]; got != "tensor" {
			t.Fatalf("resource: got %v, want tensor", got)
		}
		if got := record["error"]; got != "release failed" {
			t.Fatalf("error: got %v, want release failed", got)
		}
	})

	t.Run("finalizer handler panic falls back to stderr", func(t *testing.T) {
		output := captureDiagnosticStderr(t, func() {
			SetDiagnosticHandler(diagnosticPanicHandler{value: "handler panic"})
			emitFinalizerDiagnostic("session", errors.New("release failed"))
		})

		for _, want := range []string{"session", "release failed", "handler panic"} {
			if !strings.Contains(output, want) {
				t.Fatalf("emergency stderr output = %q, want %q", output, want)
			}
		}
	})

	t.Run("general diagnostic propagates handler panic", func(t *testing.T) {
		const panicValue = "handler panic"
		SetDiagnosticHandler(diagnosticPanicHandler{value: panicValue})
		t.Cleanup(func() { SetDiagnosticHandler(nil) })

		var recovered any
		func() {
			defer func() {
				recovered = recover()
			}()
			emitDiagnostic(context.Background(), slog.LevelInfo, "consumer callback")
		}()

		if recovered != panicValue {
			t.Fatalf("recovered panic: got %v, want %q", recovered, panicValue)
		}
	})

	t.Run("returned error emits no diagnostic", func(t *testing.T) {
		handler := &diagnosticCountingHandler{}
		SetDiagnosticHandler(handler)
		t.Cleanup(func() { SetDiagnosticHandler(nil) })

		if err := diagnosticReturnedError(); err == nil {
			t.Fatal("diagnosticReturnedError returned nil")
		}
		if got := handler.count.Load(); got != 0 {
			t.Fatalf("returned error produced %d diagnostic records, want 0", got)
		}
	})
}

func captureDiagnosticStderr(t *testing.T, operation func()) string {
	t.Helper()

	reader, writer, err := os.Pipe()
	if err != nil {
		t.Fatalf("create stderr pipe: %v", err)
	}
	original := os.Stderr
	os.Stderr = writer

	defer func() {
		os.Stderr = original
		SetDiagnosticHandler(nil)
		_ = writer.Close()
		_ = reader.Close()
	}()

	operation()
	if err := writer.Close(); err != nil {
		t.Fatalf("close stderr writer: %v", err)
	}
	output, err := io.ReadAll(reader)
	if err != nil {
		t.Fatalf("read stderr: %v", err)
	}
	return string(output)
}

func decodeDiagnosticRecord(t *testing.T, output *bytes.Buffer) map[string]any {
	t.Helper()

	var record map[string]any
	if err := json.Unmarshal(output.Bytes(), &record); err != nil {
		t.Fatalf("decode diagnostic record: %v", err)
	}
	return record
}

type diagnosticCountingHandler struct {
	count atomic.Int64
}

func (*diagnosticCountingHandler) Enabled(context.Context, slog.Level) bool {
	return true
}

func (h *diagnosticCountingHandler) Handle(context.Context, slog.Record) error {
	h.count.Add(1)
	return nil
}

func (h *diagnosticCountingHandler) WithAttrs([]slog.Attr) slog.Handler {
	return h
}

func (h *diagnosticCountingHandler) WithGroup(string) slog.Handler {
	return h
}

type diagnosticPanicHandler struct {
	value any
}

func (diagnosticPanicHandler) Enabled(context.Context, slog.Level) bool {
	return true
}

func (h diagnosticPanicHandler) Handle(context.Context, slog.Record) error {
	panic(h.value)
}

func (h diagnosticPanicHandler) WithAttrs([]slog.Attr) slog.Handler {
	return h
}

func (h diagnosticPanicHandler) WithGroup(string) slog.Handler {
	return h
}

func diagnosticReturnedError() error {
	return errors.New("returned to caller")
}
