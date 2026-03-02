package matrix

import (
	"slices"
	"testing"
)

func TestMatrix_Rows(t *testing.T) {
	m := New[float32](2, 3)

	idxsBuffer := make([]int, 2)

	linear := float32(0)

	for idxs := range m.All(idxsBuffer) {
		m.Set(idxs, linear)

		linear++
	}

	expectd := [][]float32{
		{
			0, 1, 2,
		},
		{
			3, 4, 5,
		},
	}

	rowBuffer := make([]float32, 3)

	for i, row := range m.Rows(rowBuffer) {
		a := row
		b := expectd[i]

		if !slices.Equal(a, b) {
			t.Fatalf("expected %v but got %v", b, a)
		}
	}
}
