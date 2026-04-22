package main

import (
	"fmt"
	"log"

	"go.jknobloc.com/x/llmc"
	"go.jknobloc.com/x/tokenizer/bpe"
)

func main() {
	var data llmc.DataFile[uint16]
	var gold llmc.DataFile[uint16]

	_ = must(llmc.Deserialize("artifacts/data/llmc/tinyshakespeare/tiny_shakespeare_train.bin", &data))
	_ = must(llmc.Deserialize("artifacts/test/llmc/tinyshakespeare/tiny_shakespeare_train.bin", &gold))

	t := must(bpe.NewTokenizerFromFiles("gpt2/models/base/vocab.json", "gpt2/models/base/merges.txt"))

	itoa := bpe.Itoa(t)

	fmt.Println(len(data.Tokens))
	fmt.Println(len(gold.Tokens))

	if len(data.Tokens) != len(gold.Tokens) {
		log.Fatal("length mismatch")
	}

	for i, a := range data.Tokens {
		b := gold.Tokens[i]

		if a != b {
			fmt.Printf("token mismatch: %d %d at index %d\n", a, b, i)

			fmt.Println("data", itoa[int64(a)])
			fmt.Println("gold", itoa[int64(b)])

			break
		}
	}
}

func must[T any](v T, err error) T {
	if err != nil {
		log.Fatal(err)
	}

	return v
}
