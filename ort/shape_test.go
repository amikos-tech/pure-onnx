package ort

import (
	"reflect"
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
			status: Status{handle: 0},
			want:   true,
		},
		{
			name:   "status is not OK when handle is non-zero",
			status: Status{handle: 1},
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

func TestStatus_GetErrorCode(t *testing.T) {
	tests := []struct {
		name   string
		status Status
		want   ErrorCode
	}{
		{
			name:   "returns ErrorCodeOK when status is OK",
			status: Status{handle: 0},
			want:   ErrorCodeOK,
		},
		{
			name:   "returns ErrorCodeFail when status is not OK",
			status: Status{handle: 1},
			want:   ErrorCodeFail,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.status.GetErrorCode(); got != tt.want {
				t.Errorf("Status.GetErrorCode() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestParseShape(t *testing.T) {
	tests := []struct {
		name    string
		raw     string
		want    Shape
		wantErr string
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
			name:    "invalid integer",
			raw:     "1,a,3",
			wantErr: "failed to parse dimension",
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
	tests := []struct {
		name      string
		shape     Shape
		wantCount int
		wantErr   string
	}{
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
