package main

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/amikos-tech/pure-onnx/ort"
)

func TestDiagnosticForUnsupportedPlatform(t *testing.T) {
	err := fmt.Errorf("%w: GOOS=%s GOARCH=%s", ort.ErrUnsupportedPlatform, "plan9", "386")
	got := diagnosticFor(err, "plan9", "386")

	for _, want := range []string{"GOOS=plan9", "GOARCH=386", "ONNXRUNTIME_LIB_PATH"} {
		if !strings.Contains(got, want) {
			t.Fatalf("expected diagnostic to contain %q, got: %q", want, got)
		}
	}
}

func TestDiagnosticForOtherBootstrapFailureUnchanged(t *testing.T) {
	err := errors.New("checksum mismatch")
	got := diagnosticFor(err, "linux", "amd64")

	if strings.Contains(got, "ONNXRUNTIME_LIB_PATH") {
		t.Fatalf("expected no ONNXRUNTIME_LIB_PATH hint for non-platform failure, got: %q", got)
	}
	want := "failed to initialize ONNX Runtime: checksum mismatch"
	if got != want {
		t.Fatalf("expected exact message %q, got: %q", want, got)
	}
}
