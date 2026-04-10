package tensor

import (
	"slices"
	"testing"
)

func TestNewDense_Strides(t *testing.T) {
	d := NewDense[float32]([]int{2, 3, 4}, nil)

	expected := []int{12, 4, 1}

	if !slices.Equal(d.strides, expected) {
		t.Fatalf("expected %v but got %v", expected, d.strides)
	}
}

func TestDense_IsContiguous(t *testing.T) {
	d := NewDense[float32]([]int{2, 3}, nil)

	if !d.IsContiguous() {
		t.Fatalf("expected contiguous true")
	}

	permuted := d.Permute([]int{1, 0})

	if permuted.IsContiguous() {
		t.Fatalf("expected contiguous false")
	}
}

func TestDense_IsContiguousStrict(t *testing.T) {
	d := NewDense[int]([]int{1, 3}, nil)

	v := Dense[int]{
		base:    d.base,
		offset:  0,
		shape:   []int{1, 3},
		strides: []int{999, 1},
	}

	if v.IsContiguous() {
		t.Fatalf("expected contiguous false")
	}
}

func TestDense_Contiguous(t *testing.T) {
	d := NewDense[float32]([]int{2, 3}, nil)

	permuted := d.Permute([]int{1, 0})

	contiguous := permuted.Contiguous()

	if !contiguous.IsContiguous() {
		t.Fatalf("expected contiguous true")
	}
}

func TestDense_Permute(t *testing.T) {
	d := NewDense[float32]([]int{2, 3}, nil)

	for i := range d.Size() {
		d.base[i] = float32(i)
	}

	permuted := d.Permute([]int{1, 0})

	expected := []float32{
		0, 3,
		1, 4,
		2, 5,
	}

	buffer := make([]int, permuted.Rank())

	linear := 0

	for _, v := range permuted.All(buffer) {
		a := v
		b := expected[linear]

		if a != b {
			t.Fatalf("expected %.2f but got %.2f at linear %d", b, a, linear)
		}

		linear++
	}
}

func TestDense_All(t *testing.T) {
	d := NewDense[float32]([]int{2, 3}, nil)

	for i := range d.Size() {
		d.base[i] = float32(i)
	}

	d = d.Permute([]int{1, 0})

	out := []float32{
		0, 3,
		1, 4,
		2, 5,
	}

	buffer := make([]int, d.Rank())

	step := 0

	for idxs := range d.All(buffer) {
		a := d.At(idxs)
		b := out[step]

		if a != b {
			t.Fatalf("expected %.2f but got %.2f", a, b)
		}

		step++
	}

	if step != d.Size() {
		t.Fatalf("premature termination")
	}
}
