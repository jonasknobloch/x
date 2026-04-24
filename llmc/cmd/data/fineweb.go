package main

import (
	"log"

	"go.jknobloc.com/x/dataset"
	"go.jknobloc.com/x/llm"
	"go.jknobloc.com/x/llmc"
	"go.jknobloc.com/x/shelf"
	"go.jknobloc.com/x/tokenizer/bpe"
)

// classic + 10B: fineweb10B; sample-10BT
// classic + 100B: fineweb100B; sample-100BT
// edu + 10B: edu_fineweb10B; sample-10BT
// edu + 100B: edu_fineweb100B; sample-100BT

func fineWeb() {
	var reader dataset.Reader

	if r, err := dataset.NewParquetReader(shelf.Abs("data/fineweb-edu/sample-100BT/train")); err != nil {
		log.Fatal(err)
	} else {
		reader = r
	}

	var tokenizer llm.Tokenizer

	v := shelf.Abs("models/gpt2/vocab.json")
	m := shelf.Abs("models/gpt2/merges.txt")

	if t, err := bpe.NewTokenizerFromFiles(v, m, bpe.DefaultConfig()); err != nil {
		log.Fatal(err)
	} else {
		tokenizer = t
	}

	docs := llmc.TokenizeAll(reader, tokenizer, 50256)

	if err := llmc.WriteShards(shelf.Abs("llmc/edu_fineweb100B"), "edu_fineweb", 100_000_000, docs); err != nil {
		log.Fatal(err)
	}

	if err := reader.Err(); err != nil {
		log.Fatal(err)
	}
}
