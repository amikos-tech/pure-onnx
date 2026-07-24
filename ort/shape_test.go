package ort

import (
	"errors"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

func TestNewShape(t *testing.T) {
	tests := []struct {
		name     string
		dims     []int64
		expected Shape
	}{
		{
			name:     "empty shape",
			dims:     []int64{},
			expected: Shape{},
		},
		{
			name:     "1D shape",
			dims:     []int64{10},
			expected: Shape{10},
		},
		{
			name:     "2D shape",
			dims:     []int64{3, 4},
			expected: Shape{3, 4},
		},
		{
			name:     "3D shape",
			dims:     []int64{2, 3, 4},
			expected: Shape{2, 3, 4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := NewShape(tt.dims...)
			if !reflect.DeepEqual(got, tt.expected) {
				t.Errorf("NewShape() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestStatus_IsOK(t *testing.T) {
	tests := []struct {
		name   string
		status Status
		want   bool
	}{
		{
			name:   "status is OK when handle is 0",
			status: Status(0),
			want:   true,
		},
		{
			name:   "status is not OK when handle is non-zero",
			status: Status(1),
			want:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.status.IsOK(); got != tt.want {
				t.Errorf("Status.IsOK() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestPublicHandleTypesRemainUintptrConvertible(t *testing.T) {
	const handle = uintptr(42)

	if uintptr(Status(handle)) != handle {
		t.Fatal("Status handle conversion changed")
	}
	if uintptr(Environment(handle)) != handle {
		t.Fatal("Environment handle conversion changed")
	}
	if uintptr(Session(handle)) != handle {
		t.Fatal("Session handle conversion changed")
	}
}

func TestStatusNativeErrorAccessors(t *testing.T) {
	resetEnvironmentState()
	t.Cleanup(resetEnvironmentState)

	firstMessage, firstMessagePtr := GoToCstring("invalid graph")
	secondMessage, secondMessagePtr := GoToCstring("runtime failure")
	mu.Lock()
	getErrorCodeFunc = func(status uintptr) ErrorCode {
		switch status {
		case 11:
			return ErrorCodeInvalidGraph
		case 12:
			return ErrorCodeRuntimeException
		default:
			return ErrorCodeFail
		}
	}
	getErrorMessageFunc = func(status uintptr) uintptr {
		switch status {
		case 11:
			return firstMessagePtr
		case 12:
			return secondMessagePtr
		default:
			return 0
		}
	}
	mu.Unlock()

	tests := []struct {
		status      Status
		wantCode    ErrorCode
		wantMessage string
	}{
		{status: Status(0), wantCode: ErrorCodeOK, wantMessage: ""},
		{status: Status(11), wantCode: ErrorCodeInvalidGraph, wantMessage: "invalid graph"},
		{status: Status(12), wantCode: ErrorCodeRuntimeException, wantMessage: "runtime failure"},
	}
	for _, test := range tests {
		if got := test.status.GetErrorCode(); got != test.wantCode {
			t.Errorf("Status(%d).GetErrorCode() = %v, want %v", test.status, got, test.wantCode)
		}
		if got := test.status.GetErrorMessage(); got != test.wantMessage {
			t.Errorf("Status(%d).GetErrorMessage() = %q, want %q", test.status, got, test.wantMessage)
		}
	}

	runtime.KeepAlive(firstMessage)
	runtime.KeepAlive(secondMessage)
}

func TestParseShape(t *testing.T) {
	tests := []struct {
		name         string
		raw          string
		want         Shape
		wantErr      string
		wantNumError bool
	}{
		{
			name: "standard",
			raw:  "1,384",
			want: Shape{1, 384},
		},
		{
			name: "trim spaces",
			raw:  " 2, 3 ,4 ",
			want: Shape{2, 3, 4},
		},
		{
			name: "single dimension",
			raw:  "512",
			want: Shape{512},
		},
		{
			name:    "empty input",
			raw:     "",
			wantErr: "shape string must not be empty",
		},
		{
			name:    "empty dimension",
			raw:     "1,,3",
			wantErr: "empty dimension",
		},
		{
			name:    "negative dimension",
			raw:     "1,-1,3",
			wantErr: "negative dimension",
		},
		{
			name:         "invalid integer",
			raw:          "1,a,3",
			wantErr:      "failed to parse dimension",
			wantNumError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseShape(tt.raw)
			if tt.wantErr != "" {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil", tt.wantErr)
				}
				if !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("expected error containing %q, got %q", tt.wantErr, err.Error())
				}
				if !errors.Is(err, ErrInvalidArgument) {
					t.Fatalf("ParseShape(%q) error = %v, want ErrInvalidArgument", tt.raw, err)
				}
				var numErr *strconv.NumError
				if got := errors.As(err, &numErr); got != tt.wantNumError {
					t.Fatalf("errors.As(%v, *strconv.NumError) = %t, want %t", err, got, tt.wantNumError)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("unexpected shape: got %v, want %v", got, tt.want)
			}
		})
	}
}

func TestShapeElementCountExported(t *testing.T) {
	maxInt := int64(int(^uint(0) >> 1))
	tests := []struct {
		name      string
		shape     Shape
		wantCount int
		wantErr   string
	}{
		{
			name:      "scalar",
			shape:     Shape{},
			wantCount: 1,
		},
		{
			name:      "standard",
			shape:     Shape{2, 3, 4},
			wantCount: 24,
		},
		{
			name:      "zero dimension",
			shape:     Shape{5, 0, 7},
			wantCount: 0,
		},
		{
			name:    "negative dimension",
			shape:   Shape{2, -1},
			wantErr: "must be >= 0",
		},
		{
			name:    "product overflow",
			shape:   Shape{maxInt, 2},
			wantErr: "exceeds maximum supported element count",
		},
	}
	if strconv.IntSize < 64 {
		tests = append(tests, struct {
			name      string
			shape     Shape
			wantCount int
			wantErr   string
		}{
			name:    "single dimension too large",
			shape:   Shape{maxInt + 1},
			wantErr: "too large",
		})
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ShapeElementCount(tt.shape)
			if tt.wantErr != "" {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil", tt.wantErr)
				}
				if !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("expected error containing %q, got %q", tt.wantErr, err.Error())
				}
				if !errors.Is(err, ErrInvalidArgument) {
					t.Fatalf("ShapeElementCount(%v) error = %v, want ErrInvalidArgument", tt.shape, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tt.wantCount {
				t.Fatalf("unexpected count: got %d, want %d", got, tt.wantCount)
			}
		})
	}
}
