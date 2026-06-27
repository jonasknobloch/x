package main

import (
	"log"

	"go.jknobloc.com/x/dataset"
	"go.jknobloc.com/x/gpt2"
	"go.jknobloc.com/x/shelf"
	"go.jknobloc.com/x/tokenizer/bpe"
)

func main() {
	if err := gpt2.InitializeEnvironment(); err != nil {
		log.Fatal(err)
	}

	perplexity()

	// pad right: 11.930400581007826 / 22544 tokens
	// unpadded: 11.246722646921588 / 19451 tokens

	if err := gpt2.DestroyEnvironment(); err != nil {
		log.Fatal(err)
	}
}

func data() dataset.Reader {
	var miniPile *dataset.ParquetReader

	if r, err := dataset.NewParquetReader(shelf.Abs("data/minipile/validation")); err != nil {
		log.Fatal(err)
	} else {
		miniPile = r
	}

	return dataset.NewClampedReader(miniPile, 10)
}

func model() *gpt2.Model {
	opts := gpt2.Options{
		WithCache:    false,
		WithLogits:   false,
		WithLogProbs: true,
	}

	m := gpt2.NewModel(shelf.Abs("models/gpt2/model_eval.onnx"), gpt2.ConfigDefault(), opts)

	if err := m.Init(); err != nil {
		log.Fatal(err)
	}

	return m
}

func tokenizer() *bpe.Tokenizer {
	var tok *bpe.Tokenizer

	v := shelf.Abs("models/gpt2/vocab.json")
	m := shelf.Abs("models/gpt2/merges.txt")

	if t, err := bpe.NewTokenizerFromFiles(v, m, bpe.DefaultConfig()); err != nil {
		log.Fatal(err)
	} else {
		tok = t
	}

	return tok
}
