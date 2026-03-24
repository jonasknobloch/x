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
	panic("unimplemented") // TODO implement
}

func (t *Tokenizer) Tokenize(s string) []int {
	return t.Encode(s)
}
