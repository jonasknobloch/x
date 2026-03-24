package bpe

import "github.com/jonasknobloch/mbpe"

func NewTokenizerFromFiles(vocab, merges string) (*Tokenizer, error) {
	model := mbpe.NewMBPE()

	if err := model.Load(vocab, merges); err != nil {
		return nil, err
	}

	tokenizer := mbpe.NewTokenizer(model)

	pre := mbpe.NewByteLevel(false)

	tokenizer.SetPreTokenizer(pre)
	tokenizer.SetDecoder(pre)

	return NewTokenizer(tokenizer), nil
}
