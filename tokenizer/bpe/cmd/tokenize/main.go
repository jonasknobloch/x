package main

import (
	"log"
	"sync/atomic"
	"time"

	"go.jknobloc.com/x/dataset"
	"go.jknobloc.com/x/llm"
	"go.jknobloc.com/x/tokenizer/bpe"
	"go.jknobloc.com/x/tui"
)

func main() {
	reader := data()

	t := tokenizer()

	pb := tui.NewProgressBar("Tokenize", 20, 1000, time.Now())

	var processed atomic.Int64

	pb.Start(1*time.Second, func() int {
		return int(processed.Load())
	})

	defer pb.Close()

	n := 0

	for d := range reader.Texts("text") {
		if n >= 1000 {
			break
		}

		tokens := t.Tokenize(d)

		_ = tokens

		processed.Add(1)

		n++
	}
}

func data() *dataset.ParquetReader {
	var simple *dataset.ParquetReader

	if r, err := dataset.NewParquetReader("dataset/cmd/dataset/tmp/wikipedia/simple/train"); err != nil {
		log.Fatal(err)
	} else {
		simple = r
	}

	return simple
}

func tokenizer() llm.Tokenizer {
	var tok *bpe.Tokenizer

	if t, err := bpe.NewTokenizerFromFiles("gpt2/models/base/vocab.json", "gpt2/models/base/merges.txt"); err != nil {
		log.Fatal(err)
	} else {
		tok = t
	}

	return tok
}
