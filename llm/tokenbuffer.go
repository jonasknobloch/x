package llm

import "iter"

type TokenBuffer struct {
	tokenizer   Tokenizer
	window      int
	stride      int
	buffer      []int64
	document    int
	position    int
	includeTail bool
	config      TokenBufferConfig
}

type TokenBufferConfig struct {
	Window     int
	Stride     int
	PadLeft    bool
	PadRight   bool
	PadTokenID int64
}

type TokenWindow struct {
	Document     int
	Tokens       []int64
	Seen         int
	PaddingLeft  int
	PaddingRight int
}

func NewTokenBuffer(tokenizer Tokenizer, cfg TokenBufferConfig) *TokenBuffer {
	if cfg.Stride > cfg.Window {
		panic("stride exceeds window")
	}

	if cfg.PadLeft && cfg.PadRight {
		panic("either pad left or right")
	}

	return &TokenBuffer{
		tokenizer:   tokenizer,
		window:      cfg.Window,
		stride:      cfg.Stride,
		buffer:      make([]int64, 0, 2*cfg.Window),
		document:    -1,
		position:    0,
		includeTail: true,
		config:      cfg,
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

func (tb *TokenBuffer) Push(document int, text string) iter.Seq[TokenWindow] {
	return func(yield func(window TokenWindow) bool) {
		if tb.document != -1 && document != tb.document {
			w, ok := tb.Tail()

			if ok && !yield(w) {
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
			w := TokenWindow{
				Document: tb.document,
				Tokens:   make([]int64, tb.window),
				Seen:     min(tb.Position(), tb.window-tb.stride),
			}

			copy(w.Tokens, tb.buffer[:tb.window])

			if !yield(w) {
				return
			}

			tb.position += tb.stride

			tb.buffer = append(tb.buffer[:0], tb.buffer[tb.stride:]...)
		}
	}
}

func (tb *TokenBuffer) Tail() (TokenWindow, bool) {
	seen := min(tb.Position(), min(len(tb.buffer), tb.window-tb.stride))

	w := TokenWindow{
		Document: tb.document,
		Seen:     seen,
	}

	tb.document = -1
	tb.position = 0

	if !tb.includeTail || len(tb.buffer) == 0 {
		tb.buffer = tb.buffer[:0]

		return w, false
	}

	w.Tokens = make([]int64, len(tb.buffer))

	copy(w.Tokens, tb.buffer)

	if tb.config.PadLeft || tb.config.PadRight {
		padding := make([]int64, tb.window-len(w.Tokens))

		for i := range padding {
			padding[i] = tb.config.PadTokenID
		}

		if tb.config.PadLeft {
			w.Tokens = append(padding, w.Tokens...)
			w.PaddingLeft = len(padding)
		} else {
			w.Tokens = append(w.Tokens, padding...)
			w.PaddingRight = len(padding)
		}
	}

	tb.buffer = tb.buffer[:0]

	return w, true
}

func (tb *TokenBuffer) Stream(docs iter.Seq2[int, string]) iter.Seq[TokenWindow] {
	return func(yield func(window TokenWindow) bool) {
		for n, d := range docs {
			for w := range tb.Push(n, d) {
				if !yield(w) {
					return
				}
			}

			if w, ok := tb.Tail(); ok && !yield(w) {
				return
			}
		}
	}
}
