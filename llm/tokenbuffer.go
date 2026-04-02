package llm

import "iter"

type TokenBuffer struct {
	tokenizer   Tokenizer
	window      int
	stride      int
	buffer      []int64
	document    int
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
		includeTail: true,
	}
}

func (tb *TokenBuffer) IncludeTail() bool {
	return tb.includeTail
}

func (tb *TokenBuffer) SetIncludeTail(includeTail bool) {
	tb.includeTail = includeTail
}

func (tb *TokenBuffer) Push(document int, text string) iter.Seq[[]int64] {
	return func(yield func([]int64) bool) {
		if tb.document != -1 && document != tb.document {
			if tb.includeTail && len(tb.buffer) > 0 {
				if !yield(tb.buffer) {
					return
				}
			}

			tb.buffer = nil
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

			if !yield(w) {
				return
			}

			tb.buffer = append(tb.buffer[:0], tb.buffer[tb.stride:]...)
		}
	}
}

func (tb *TokenBuffer) Tail() []int64 {
	if !tb.includeTail || len(tb.buffer) == 0 {
		return nil
	}

	w := make([]int64, len(tb.buffer))

	copy(w, tb.buffer)

	tb.buffer = nil
	tb.document = -1

	return w
}
