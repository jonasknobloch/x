package main

import (
	"fmt"
	"log"

	"golang.org/x/exp/constraints"

	"go.jknobloc.com/x/llmc"
	"go.jknobloc.com/x/tokenizer/bpe"
)

func main() {
	var a llmc.DataFile[uint16]
	var b llmc.DataFile[uint16]

	_ = must(llmc.Deserialize("artifacts/data/llmc/edu_fineweb100B/edu_fineweb_val_000000.bin", &a))
	_ = must(llmc.Deserialize("artifacts/test/llmc/edu_fineweb100B/edu_fineweb_val_000000.bin", &b))

	t := must(bpe.NewTokenizerFromFiles("gpt2/models/base/vocab.json", "gpt2/models/base/merges.txt"))

	s := decode(&a, 1024, t)
	k := decode(&b, 1024, t)

	fmt.Println(s)

	fmt.Println()

	fmt.Println(k)
}

func must[T any](v T, err error) T {
	if err != nil {
		log.Fatal(err)
	}

	return v
}

func decode[T constraints.Integer](src *llmc.DataFile[T], n int, t *bpe.Tokenizer) string {
	ids := make([]int, n)

	if len(src.Tokens) < n {
		panic("not enough tokens")
	}

	for i := range n {
		ids[i] = int(src.Tokens[i])
	}

	return t.Decode(ids)
}
