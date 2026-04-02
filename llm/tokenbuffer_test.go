package llm

import (
	"fmt"
	"slices"
	"testing"
)

type byteTokenizer struct{}

func (bt byteTokenizer) Tokenize(text string) []int {
	r := make([]int, len(text))

	for i := 0; i < len(text); i++ {
		r[i] = int(text[i])
	}

	return r
}

func TestTokenBuffer_Push(t *testing.T) {
	type gold struct {
		window   int
		stride   int
		text     string
		expected [][]int64
	}

	tests := []gold{
		{
			window: 5, stride: 5,
			text:     "abcdefghij",
			expected: [][]int64{{97, 98, 99, 100, 101}, {102, 103, 104, 105, 106}},
		},
		{
			window: 4, stride: 2,
			text:     "abcdef",
			expected: [][]int64{{97, 98, 99, 100}, {99, 100, 101, 102}},
		},
		{
			window: 10, stride: 5,
			text:     "abcde",
			expected: nil,
		},
	}

	for _, tt := range tests {
		t.Run(
			fmt.Sprintf("window%d_stride%d", tt.window, tt.stride),

			func(t *testing.T) {
				tb := NewTokenBuffer(byteTokenizer{}, tt.window, tt.stride)

				tb.SetIncludeTail(false)

				var got [][]int64

				for w := range tb.Push(0, tt.text) {
					got = append(got, w)
				}

				if len(got) != len(tt.expected) {
					t.Fatalf("expected %d windows but got %d", len(tt.expected), len(got))
				}

				for i := range got {
					if !slices.Equal(got[i], tt.expected[i]) {
						t.Errorf("window %d: expected %v but got %v", i, tt.expected[i], got[i])
					}
				}
			},
		)
	}
}

func TestTokenBuffer_PushAccumulates(t *testing.T) {
	tb := NewTokenBuffer(byteTokenizer{}, 4, 4)

	var got [][]int64

	for w := range tb.Push(0, "ab") {
		got = append(got, w)
	}

	if len(got) != 0 {
		t.Fatalf("expected 0 windows but got %d", len(got))
	}

	for w := range tb.Push(0, "cd") {
		got = append(got, w)
	}

	expected := [][]int64{{97, 98, 99, 100}}

	if len(got) != len(expected) {
		t.Fatalf("expected %d windows but got %d", len(expected), len(got))
	}

	for i := range got {
		if !slices.Equal(got[i], expected[i]) {
			t.Errorf("window %d: expected %v but got %v", i, expected[i], got[i])
		}
	}
}

func TestTokenBuffer_Tail(t *testing.T) {
	tb := NewTokenBuffer(byteTokenizer{}, 5, 5)

	tb.SetIncludeTail(true)

	var got [][]int64

	for w := range tb.Push(0, "abcdef") {
		got = append(got, w)
	}

	if tail := tb.Tail(); tail != nil {
		got = append(got, tail)
	}

	expected := [][]int64{{97, 98, 99, 100, 101}, {102}}

	if len(got) != len(expected) {
		t.Fatalf("expected %d windows but got %d", len(expected), len(got))
	}

	for i := range got {
		if !slices.Equal(got[i], expected[i]) {
			t.Errorf("window %d: expected %v but got %v", i, expected[i], got[i])
		}
	}
}

func TestTokenBuffer_DocumentBoundaryYieldsTail(t *testing.T) {
	tb := NewTokenBuffer(byteTokenizer{}, 10, 10)

	tb.SetIncludeTail(true)

	var a [][]int64

	for w := range tb.Push(0, "abcde") {
		a = append(a, w)
	}

	if len(a) != 0 {
		t.Fatalf("expected 0 windows but got %d", len(a))
	}

	var b [][]int64

	for w := range tb.Push(2, "fghij") {
		b = append(b, w)
	}

	if len(b) != 1 {
		t.Fatalf("expected 1 window but got %d", len(b))
	}

	expected := toInt64(byteTokenizer{}.Tokenize("abcde"))

	if !slices.Equal(b[0], expected) {
		t.Errorf("expected %v but got %v", expected, b[0])
	}
}
