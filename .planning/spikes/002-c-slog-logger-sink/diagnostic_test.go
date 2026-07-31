package slogloggersink

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"sync"
	"sync/atomic"
	"testing"
)

func TestNilLoggerRestoresSilentDefault(t *testing.T) {
	SetDiagnosticLogger(nil)
	emitDiagnostic(context.Background(), slog.LevelWarn, "not observable", slog.Int("attempt", 1))
}

func TestStructuredDiagnosticIsOptInAndReturnedErrorIsNotLogged(t *testing.T) {
	var output bytes.Buffer
	logger := slog.New(slog.NewJSONHandler(&output, nil))
	SetDiagnosticLogger(logger)
	t.Cleanup(func() { SetDiagnosticLogger(nil) })

	if err := operationThatReturnsError(); err == nil {
		t.Fatal("operationThatReturnsError returned nil")
	}
	if output.Len() != 0 {
		t.Fatalf("returned error produced diagnostic output: %q", output.String())
	}

	emitDiagnostic(
		context.Background(),
		slog.LevelWarn,
		"finalizer cleanup failed",
		slog.String("resource", "tensor"),
		slog.Any("error", errors.New("release failed")),
	)

	var record map[string]any
	if err := json.Unmarshal(output.Bytes(), &record); err != nil {
		t.Fatalf("decode slog output: %v", err)
	}
	if record["msg"] != "finalizer cleanup failed" || record["resource"] != "tensor" {
		t.Fatalf("unexpected slog record: %#v", record)
	}
}

func TestConsumerLoggerAttributesArePreserved(t *testing.T) {
	var output bytes.Buffer
	logger := slog.New(slog.NewJSONHandler(&output, nil)).With("component", "inference")
	SetDiagnosticLogger(logger)
	t.Cleanup(func() { SetDiagnosticLogger(nil) })

	emitDiagnostic(context.Background(), slog.LevelInfo, "waiting for bootstrap lock", slog.Int("seconds", 2))

	var record map[string]any
	if err := json.Unmarshal(output.Bytes(), &record); err != nil {
		t.Fatalf("decode slog output: %v", err)
	}
	if record["component"] != "inference" || record["seconds"] != float64(2) {
		t.Fatalf("configured attributes were not preserved: %#v", record)
	}
}

func TestConcurrentConfigureAndEmit(t *testing.T) {
	const (
		writers = 32
		events  = 200
	)

	firstHandler := &countingHandler{}
	secondHandler := &countingHandler{}
	first := slog.New(firstHandler)
	second := slog.New(secondHandler)
	SetDiagnosticLogger(first)
	t.Cleanup(func() { SetDiagnosticLogger(nil) })

	var wg sync.WaitGroup
	for writer := range writers {
		wg.Add(1)
		go func() {
			defer wg.Done()
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
	for change := range events {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if change%2 == 0 {
				SetDiagnosticLogger(first)
				return
			}
			SetDiagnosticLogger(second)
		}()
	}
	wg.Wait()

	got := firstHandler.count.Load() + secondHandler.count.Load()
	want := int64(writers * events)
	if got != want {
		t.Fatalf("captured events: got %d, want %d", got, want)
	}
}

func BenchmarkDiscardedDiagnostic(b *testing.B) {
	SetDiagnosticLogger(nil)
	b.ReportAllocs()
	for b.Loop() {
		emitDiagnostic(
			context.Background(),
			slog.LevelWarn,
			"cleanup failed",
			slog.String("resource", "tensor"),
			slog.Int("id", 42),
		)
	}
}

func operationThatReturnsError() error {
	return errors.New("returned to caller")
}

type countingHandler struct {
	count atomic.Int64
}

func (*countingHandler) Enabled(context.Context, slog.Level) bool {
	return true
}

func (h *countingHandler) Handle(context.Context, slog.Record) error {
	h.count.Add(1)
	return nil
}

func (h *countingHandler) WithAttrs([]slog.Attr) slog.Handler {
	return h
}

func (h *countingHandler) WithGroup(string) slog.Handler {
	return h
}
