package llm

import "iter"

type TokenBuffer struct {
	tokenizer   Tokenizer
	window      int
	stride      int
	buffer      []int64
	document    int
	seen        int
	includeTail bool
}

func NewTokenBuffer(tokenizer Tokenizer, window, stride int) *TokenBuffer {
	if stride > window {
		panic("stride exceeds window")
	}

	return &TokenBuffer{
		tokenizer:   tokenizer,
		window:      window,
		stride:      stride,
		buffer:      make([]int64, 0, 2*window),
		document:    -1,
		seen:        0,
		includeTail: true,
	}
}

func (tb *TokenBuffer) IncludeTail() bool {
	return tb.includeTail
}

func (tb *TokenBuffer) SetIncludeTail(includeTail bool) {
	tb.includeTail = includeTail
}

func (tb *TokenBuffer) Seen() int {
	return tb.seen
}

func (tb *TokenBuffer) Push(document int, text string) iter.Seq2[[]int64, int] {
	return func(yield func([]int64, int) bool) {
		if tb.document != -1 && document != tb.document {
			tail, seen := tb.Tail()

			if len(tail) > 0 && !yield(tail, seen) {
				return
			}
		}

		tb.document = document

		if text == "" {
			return
		}

		ids := toInt64(tb.tokenizer.Tokenize(text))

		tb.buffer = append(tb.buffer, ids...)

		for len(tb.buffer) >= tb.window {
			w := make([]int64, tb.window)

			copy(w, tb.buffer[:tb.window])

			if !yield(w, tb.seen) {
				return
			}

			tb.seen += tb.stride

			tb.buffer = append(tb.buffer[:0], tb.buffer[tb.stride:]...)
		}
	}
}

func (tb *TokenBuffer) Tail() ([]int64, int) {
	seen := tb.seen

	tb.document = -1
	tb.seen = 0

	if !tb.includeTail || len(tb.buffer) == 0 {
		tb.buffer = tb.buffer[:0]

		return nil, seen
	}

	w := make([]int64, len(tb.buffer))

	copy(w, tb.buffer)

	tb.buffer = tb.buffer[:0]

	return w, seen
}
