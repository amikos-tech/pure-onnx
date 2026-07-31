package ort

import "testing"

type valueKindStub struct {
	kind ValueType
}

func (*valueKindStub) Destroy() error { return nil }
func (v *valueKindStub) Type() ValueType {
	return v.kind
}
func (*valueKindStub) ortValue() {}

func TestValueIsTensorChecksKind(t *testing.T) {
	floatTensor := &Tensor[float32]{}
	intTensor := &Tensor[int64]{}
	var typedNilTensor *Tensor[float32]

	tests := []struct {
		name  string
		value Value
		want  bool
	}{
		{
			name:  "matching tensor element type",
			value: floatTensor,
			want:  true,
		},
		{
			name:  "different tensor element type",
			value: intTensor,
			want:  true,
		},
		{
			name:  "typed nil tensor still reports its kind",
			value: typedNilTensor,
			want:  true,
		},
		{
			name:  "nil interface",
			value: nil,
			want:  false,
		},
		{
			name:  "non-tensor kind",
			value: &valueKindStub{kind: ValueTypeSequence},
			want:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsTensor(tt.value); got != tt.want {
				t.Fatalf("IsTensor() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestValueAsTensorRequiresExactNonNilType(t *testing.T) {
	floatTensor := &Tensor[float32]{}
	intTensor := &Tensor[int64]{}
	var typedNilTensor *Tensor[float32]

	tests := []struct {
		name  string
		value Value
		want  *Tensor[float32]
		ok    bool
	}{
		{
			name:  "exact type",
			value: floatTensor,
			want:  floatTensor,
			ok:    true,
		},
		{
			name:  "different element type",
			value: intTensor,
			want:  nil,
			ok:    false,
		},
		{
			name:  "nil interface",
			value: nil,
			want:  nil,
			ok:    false,
		},
		{
			name:  "typed nil tensor",
			value: typedNilTensor,
			want:  nil,
			ok:    false,
		},
		{
			name:  "non-tensor value",
			value: &valueKindStub{kind: ValueTypeSequence},
			want:  nil,
			ok:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := AsTensor[float32](tt.value)
			if ok != tt.ok {
				t.Fatalf("AsTensor() ok = %v, want %v", ok, tt.ok)
			}
			if got != tt.want {
				t.Fatalf("AsTensor() = %p, want %p", got, tt.want)
			}
		})
	}
}

func TestValueAsTensorDoesNotAllocateOrCopy(t *testing.T) {
	tensor := &Tensor[float32]{data: []float32{1, 2, 3}}
	var got *Tensor[float32]
	var ok bool

	allocs := testing.AllocsPerRun(1000, func() {
		got, ok = AsTensor[float32](tensor)
	})

	if allocs != 0 {
		t.Fatalf("AsTensor() allocated %v times per call, want 0", allocs)
	}
	if !ok || got != tensor {
		t.Fatalf("AsTensor() = (%p, %v), want original tensor %p and true", got, ok, tensor)
	}
	if &got.data[0] != &tensor.data[0] {
		t.Fatal("AsTensor() copied tensor data")
	}
}

func TestValueSupportsHeterogeneousTensorTypes(t *testing.T) {
	floatTensor := &Tensor[float32]{}
	intTensor := &Tensor[int64]{}

	values := []Value{floatTensor, intTensor}
	if len(values) != 2 {
		t.Fatalf("len(values) = %d, want 2", len(values))
	}
}
