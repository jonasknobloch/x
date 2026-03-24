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

func FuzzFSA_FindAll(f *testing.F) {
	fsa := NewFSA()

	ref := mbpe.NewFSA()

	f.Add("foo")
	f.Add(" ")
	f.Add("\n")
	f.Add("   bar")

	f.Fuzz(func(t *testing.T, s string) {
		out := fsa.FindAll(s)

		expected := ref.FindAll(s)

		if len(out) != len(expected) {
			t.Fatalf("expected %s but got %s", expected, out)
		}

		for i, m := range out {
			if m != expected[i] {
				t.Errorf("expected [%s] but got [%s]", expected[i], m)
			}
		}
	})
}
