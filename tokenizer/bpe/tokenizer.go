package bpe

import "github.com/jonasknobloch/mbpe"

type Tokenizer struct {
	mbpe *mbpe.Tokenizer
}

func NewTokenizer(mbpe *mbpe.Tokenizer) *Tokenizer {
	return &Tokenizer{
		mbpe: mbpe,
	}
}

func (t *Tokenizer) Encode(s string) []int {
	return t.mbpe.Tokenize(s)
}

func (t *Tokenizer) Decode(ids []int) string {
	d := t.mbpe.Decoder()

	m, ok := t.mbpe.Model().(*mbpe.MBPE)

	if !ok {
		panic("unsupported model")
	}

	return d.Decode(m.ToString(ids))
}

func (t *Tokenizer) Tokenize(s string) (ids []int) {
	defer func() {
		if r := recover(); r != nil {
			// defer UnknownRunes(t, s)

			ids = []int{}
		}
	}()

	return t.Encode(s)
}
