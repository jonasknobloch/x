package main

import (
	"log"

	"github.com/jonasknobloch/mbpe"

	"go.jknobloc.com/x/dataset"
	"go.jknobloc.com/x/gpt2"
)

func main() {
	perplexity()
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
	m := gpt2.NewModel("gpt2/models/base/model.onnx", "0")

	if err := m.Init(); err != nil {
		log.Fatal(err)
	}

	return m
}

func tokenizer() *mbpe.Tokenizer {
	m := mbpe.NewMBPE()

	if err := m.Load("gpt2/models/base/vocab.json", "gpt2/models/base/merges.txt"); err != nil {
		log.Fatal(err)
	}

	t := mbpe.NewTokenizer(m)

	byteLevel := mbpe.NewByteLevel(false)

	t.SetPreTokenizer(byteLevel)
	t.SetDecoder(byteLevel)

	return t
}
