package llm

import "iter"

type TokenBufferConfig struct {
	Window int
	Stride int
}

type TokenBuffer struct {
	tokenizer   Tokenizer
	window      int
	stride      int
	buffer      []int64
	document    int
	position    int
	includeTail bool
}

func NewTokenBuffer(tokenizer Tokenizer, cfg TokenBufferConfig) *TokenBuffer {
	if cfg.Stride > cfg.Window {
		panic("stride exceeds window")
	}

	return &TokenBuffer{
		tokenizer:   tokenizer,
		window:      cfg.Window,
		stride:      cfg.Stride,
		buffer:      make([]int64, 0, 2*cfg.Window),
		document:    -1,
		position:    0,
		includeTail: true,
	}
}

func (tb *TokenBuffer) IncludeTail() bool {
	return tb.includeTail
}

func (tb *TokenBuffer) SetIncludeTail(includeTail bool) {
	tb.includeTail = includeTail
}

func (tb *TokenBuffer) Position() int {
	return tb.position
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

			if !yield(w, min(tb.Position(), tb.window-tb.stride)) {
				return
			}

			tb.position += tb.stride

			tb.buffer = append(tb.buffer[:0], tb.buffer[tb.stride:]...)
		}
	}
}

func (tb *TokenBuffer) Tail() ([]int64, int) {
	seen := min(tb.Position(), min(len(tb.buffer), tb.window-tb.stride))

	tb.document = -1
	tb.position = 0

	if !tb.includeTail || len(tb.buffer) == 0 {
		tb.buffer = tb.buffer[:0]

		return nil, seen
	}

	w := make([]int64, len(tb.buffer))

	copy(w, tb.buffer)

	tb.buffer = tb.buffer[:0]

	return w, seen
}
