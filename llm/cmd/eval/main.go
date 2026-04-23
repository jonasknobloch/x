package main

import (
	"log"

	"go.jknobloc.com/x/dataset"
	"go.jknobloc.com/x/gpt2"
	"go.jknobloc.com/x/tokenizer/bpe"
)

func main() {
	if err := gpt2.InitializeEnvironment(); err != nil {
		log.Fatal(err)
	}

	perplexity()
	// logprobs()

	if err := gpt2.DestroyEnvironment(); err != nil {
		log.Fatal(err)
	}
}

func data() *dataset.ParquetReader {
	var miniPile *dataset.ParquetReader

	if r, err := dataset.NewParquetReader("dataset/cmd/dataset/tmp/minipile/validation"); err != nil {
		log.Fatal(err)
	} else {
		miniPile = r
	}

	return miniPile
}

func model() *gpt2.Model {
	opts := gpt2.Options{
		WithCache:    false,
		WithLogits:   false,
		WithLogProbs: true,
	}

	m := gpt2.NewModel("gpt2/models/base/model_eval.onnx", gpt2.DefaultConfig(), opts)

	if err := m.Init(); err != nil {
		log.Fatal(err)
	}

	return m
}

func tokenizer() *bpe.Tokenizer {
	var tok *bpe.Tokenizer

	if t, err := bpe.NewTokenizerFromFiles("gpt2/models/base/vocab.json", "gpt2/models/base/merges.txt"); err != nil {
		log.Fatal(err)
	} else {
		tok = t
	}

	return tok
}
