package main

import (
	"fmt"
	"log"

	"golang.org/x/exp/constraints"

	"go.jknobloc.com/x/llmc"
	"go.jknobloc.com/x/shelf"
	"go.jknobloc.com/x/tokenizer/bpe"
)

func main() {
	var a llmc.DataFile[uint16]
	var b llmc.DataFile[uint16]

	_ = must(llmc.Deserialize(shelf.Abs("llmc/edu_fineweb100B/edu_fineweb_val_000000.bin"), &a))
	_ = must(llmc.Deserialize(shelf.Abs("test/llmc/edu_fineweb100B/edu_fineweb_val_000000.bin"), &b))

	v := shelf.Abs("models/gpt2/vocab.json")
	m := shelf.Abs("models/gpt2/merges.txt")

	t := must(bpe.NewTokenizerFromFiles(v, m, bpe.DefaultConfig()))

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
