package customdiagnosticsink

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"sync"
	"testing"
)

type event struct {
	level   Level
	message string
	fields  []Field
}

type captureSink struct {
	mu     sync.Mutex
	events []event
}

func (s *captureSink) Log(_ context.Context, level Level, message string, fields ...Field) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.events = append(s.events, event{
		level:   level,
		message: message,
		fields:  append([]Field(nil), fields...),
	})
}

func (s *captureSink) snapshot() []event {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]event(nil), s.events...)
}

func TestNilSinkRestoresSilentDefault(t *testing.T) {
	SetDiagnosticSink(nil)
	emitDiagnostic(context.Background(), LevelWarn, "not observable", Field{Key: "attempt", Value: 1})
}

func TestStructuredDiagnosticIsOptInAndReturnedErrorIsNotLogged(t *testing.T) {
	sink := &captureSink{}
	SetDiagnosticSink(sink)
	t.Cleanup(func() { SetDiagnosticSink(nil) })

	if err := operationThatReturnsError(); err == nil {
		t.Fatal("operationThatReturnsError returned nil")
	}
	if got := len(sink.snapshot()); got != 0 {
		t.Fatalf("returned error produced %d diagnostic events, want 0", got)
	}

	emitDiagnostic(
		context.Background(),
		LevelWarn,
		"finalizer cleanup failed",
		Field{Key: "resource", Value: "tensor"},
		Field{Key: "error", Value: errors.New("release failed")},
	)

	events := sink.snapshot()
	if len(events) != 1 {
		t.Fatalf("diagnostic count: got %d, want 1", len(events))
	}
	if events[0].level != LevelWarn || events[0].message != "finalizer cleanup failed" {
		t.Fatalf("unexpected event: %#v", events[0])
	}
	if len(events[0].fields) != 2 || events[0].fields[0].Key != "resource" {
		t.Fatalf("structured fields were not preserved: %#v", events[0].fields)
	}
}

func TestConcurrentConfigureAndEmit(t *testing.T) {
	const (
		writers = 32
		events  = 200
	)

	first := &captureSink{}
	second := &captureSink{}
	SetDiagnosticSink(first)
	t.Cleanup(func() { SetDiagnosticSink(nil) })

	var wg sync.WaitGroup
	for writer := range writers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for sequence := range events {
				emitDiagnostic(
					context.Background(),
					LevelInfo,
					"bootstrap lock wait",
					Field{Key: "writer", Value: writer},
					Field{Key: "sequence", Value: sequence},
				)
			}
		}()
	}
	for change := range events {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if change%2 == 0 {
				SetDiagnosticSink(first)
				return
			}
			SetDiagnosticSink(second)
		}()
	}
	wg.Wait()

	got := len(first.snapshot()) + len(second.snapshot())
	want := writers * events
	if got != want {
		t.Fatalf("captured events: got %d, want %d", got, want)
	}
}

func TestSlogAdapter(t *testing.T) {
	var output bytes.Buffer
	adapter := slogSink{logger: slog.New(slog.NewJSONHandler(&output, nil))}
	SetDiagnosticSink(adapter)
	t.Cleanup(func() { SetDiagnosticSink(nil) })

	emitDiagnostic(context.Background(), LevelWarn, "runtime is old", Field{Key: "version", Value: "1.20.0"})

	var record map[string]any
	if err := json.Unmarshal(output.Bytes(), &record); err != nil {
		t.Fatalf("decode slog output: %v", err)
	}
	if record["msg"] != "runtime is old" || record["version"] != "1.20.0" {
		t.Fatalf("unexpected slog record: %#v", record)
	}
}

func BenchmarkDiscardedDiagnostic(b *testing.B) {
	SetDiagnosticSink(nil)
	b.ReportAllocs()
	for b.Loop() {
		emitDiagnostic(
			context.Background(),
			LevelWarn,
			"cleanup failed",
			Field{Key: "resource", Value: "tensor"},
			Field{Key: "id", Value: 42},
		)
	}
}

func operationThatReturnsError() error {
	return errors.New("returned to caller")
}

// slogSink is consumer-side adapter code. A Zap adapter has the same one-method
// shape but maps Field values to zap.Field instead.
type slogSink struct {
	logger *slog.Logger
}

func (s slogSink) Log(ctx context.Context, level Level, message string, fields ...Field) {
	attrs := make([]slog.Attr, len(fields))
	for i, field := range fields {
		attrs[i] = slog.Any(field.Key, field.Value)
	}

	slogLevel := slog.LevelInfo
	if level == LevelWarn {
		slogLevel = slog.LevelWarn
	}
	s.logger.LogAttrs(ctx, slogLevel, message, attrs...)
}
