package main

import (
	"log"
	"sync/atomic"
	"time"

	"go.jknobloc.com/x/dataset"
	"go.jknobloc.com/x/llm"
	"go.jknobloc.com/x/profile"
	"go.jknobloc.com/x/shelf"
	"go.jknobloc.com/x/tokenizer/bpe"
	"go.jknobloc.com/x/tui"
)

func main() {
	stop := profile.CPU()

	reader := data()

	t := tokenizer()

	pb := tui.NewProgressBar("Tokenize", 20, 1000, time.Now())

	var processed atomic.Int64

	pb.Start(1*time.Second, func() int {
		return int(processed.Load())
	})

	defer pb.Close()

	for n, d := range reader.Texts() {
		if n >= 1000 {
			break
		}

		tokens := t.Tokenize(d)

		_ = tokens

		processed.Add(1)
	}

	profile.Mem()

	stop()
}

func data() *dataset.ParquetReader {
	var simple *dataset.ParquetReader

	if r, err := dataset.NewParquetReader(shelf.Abs("data/wikipedia/20231101/simple/train")); err != nil {
		log.Fatal(err)
	} else {
		simple = r
	}

	return simple
}

func tokenizer() llm.Tokenizer {
	var tok *bpe.Tokenizer

	v := shelf.Abs("models/gpt2/vocab.json")
	m := shelf.Abs("models/gpt2/merges.txt")

	if t, err := bpe.NewTokenizerFromFiles(v, m, bpe.DefaultConfig()); err != nil {
		log.Fatal(err)
	} else {
		tok = t
	}

	_ = bpe.ByteCoverage(tok)

	return tok
}
