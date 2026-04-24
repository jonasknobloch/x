package bpe

import "github.com/jonasknobloch/mbpe"

func NewTokenizerFromFiles(vocab, merges string, cfg Config) (*Tokenizer, error) {
	model := mbpe.NewMBPE()

	if err := model.Load(vocab, merges); err != nil {
		return nil, err
	}

	tokenizer := mbpe.NewTokenizer(model)

	pre := NewByteLevel(false)

	tokenizer.SetPreTokenizer(pre)
	tokenizer.SetDecoder(pre)

	return NewTokenizer(tokenizer, cfg), nil
}

func Vocab(t *Tokenizer) []string {
	m, ok := t.mbpe.Model().(*mbpe.MBPE)

	if !ok {
		panic("unimplemented")
	}

	return m.Vocab()
}

func Merges(t *Tokenizer) [][2]string {
	m, ok := t.mbpe.Model().(*mbpe.MBPE)

	if !ok {
		panic("unimplemented")
	}

	return m.Merges()
}
