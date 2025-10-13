package ort

import (
	"reflect"
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
