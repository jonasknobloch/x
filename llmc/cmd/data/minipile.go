package main

import (
	"log"

	"go.jknobloc.com/x/dataset"
	"go.jknobloc.com/x/llm"
	"go.jknobloc.com/x/llmc"
	"go.jknobloc.com/x/shelf"
	"go.jknobloc.com/x/tokenizer/bpe"
)

func miniPile() {
	prepareSplit := func(datasetPath, shardName string) {
		var reader dataset.Reader

		if r, err := dataset.NewParquetReader(datasetPath); err != nil {
			log.Fatal(err)
		} else {
			reader = r
		}

		var tokenizer llm.Tokenizer

		v := shelf.Abs("tokenizers/tokenizer_gpt2_50256_m000_minipile/vocab.json")
		m := shelf.Abs("tokenizers/tokenizer_gpt2_50256_m000_minipile/merges.txt")

		if t, err := bpe.NewTokenizerFromFiles(v, m, bpe.DefaultConfig()); err != nil {
			log.Fatal(err)
		} else {
			tokenizer = t
		}

		docs := llmc.TokenizeAll(reader, tokenizer, 50256)

		if _, err := llmc.WriteShards(shelf.Abs("llmc/minipile/m000"), "", 100_000_000, docs); err != nil {
			log.Fatal(err)
		}

		if err := reader.Err(); err != nil {
			log.Fatal(err)
		}
	}

	prepareSplit(shelf.Abs("data/minipile/train"), "minipile_train")
	prepareSplit(shelf.Abs("data/minipile/validation"), "minipile_val")
}
