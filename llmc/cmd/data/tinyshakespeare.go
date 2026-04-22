package main

import (
	"fmt"
	"log"

	"go.jknobloc.com/x/dataset"
	"go.jknobloc.com/x/llmc"
	"go.jknobloc.com/x/tokenizer/bpe"
)

func tinyShakespeare() {
	var reader *dataset.FileReader

	if r, err := dataset.NewFileReader("artifacts/data/raw/tinyshakespeare/input.txt", "*.txt"); err != nil {
		log.Fatal(err)
	} else {
		reader = r
	}

	reader.SetDelimiters("\n\n")

	var tokenizer *bpe.Tokenizer

	if t, err := bpe.NewTokenizerFromFiles("gpt2/models/base/vocab.json", "gpt2/models/base/merges.txt"); err != nil {
		log.Fatal(err)
	} else {
		tokenizer = t
	}

	result := make([]int64, 0)

	for _, doc := range reader.Texts() {
		ids := tokenizer.Tokenize(doc)

		toks := make([]int64, len(ids)+1)

		toks[0] = 50256 // end of text

		for i, id := range ids {
			toks[i+1] = int64(id)
		}

		result = append(result, toks...)
	}

	val := llmc.DataFile[int64]{
		Model:  llmc.GPT2,
		Tokens: result[:32768],
	}

	train := llmc.DataFile[int64]{
		Model:  llmc.GPT2,
		Tokens: result[32768:],
	}

	if _, err := llmc.Serialize(&val, "artifacts/data/llmc/tinyshakespeare/tiny_shakespeare_val.bin"); err != nil {
		log.Fatal(err)
	}

	if _, err := llmc.Serialize(&train, "artifacts/data/llmc/tinyshakespeare/tiny_shakespeare_train.bin"); err != nil {
		log.Fatal(err)
	}

	fmt.Println(val)
}
