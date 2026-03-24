package split

import (
	"slices"
	"testing"

	"github.com/jonasknobloch/mbpe"
)

var s = "The quick brown fox jumps over the lazy dog's back."

func TestFSA_FindAll(t *testing.T) {
	a, b := mbpe.NewFSA(), NewFSA()

	x, y := a.FindAll(s), b.FindAll(s)

	if !slices.Equal(x, y) {
		t.Errorf("expected %v\nbut got %v\n", x, y)
	}
}

func BenchmarkFSA_FindAll(b *testing.B) {
	f := NewFSA()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = f.FindAll(s)
	}
}

func BenchmarkFSA_FindAllReference(b *testing.B) {
	f := mbpe.NewFSA()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = f.FindAll(s)
	}
}
