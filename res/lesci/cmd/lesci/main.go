package main

import (
	"fmt"
	"log"

	"github.com/jonasknobloch/mbpe"
	"github.com/jonasknobloch/x/dataset"
	"github.com/jonasknobloch/x/gpt2"
)

func main() {
	// lesci.Run()

	// foo()
}

func data() *dataset.Reader {
	var miniPile *dataset.Reader

	// TODO "root" is a weird arg name

	if r, err := dataset.NewReader("dataset/cmd/dataset/tmp/minipile/train"); err != nil {
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
		panic("") // TODO
	}

	t := mbpe.NewTokenizer(m)

	byteLevel := mbpe.NewByteLevel(false)

	t.SetPreTokenizer(byteLevel)
	t.SetDecoder(byteLevel)

	return t
}

func foo() {
	tok := tokenizer()
	causal := model()
	pile := data()

	i := 0

	for s := range pile.Texts("text") {
		if i > 0 {
			break
		} else {
			i++
		}

		in := tok.Tokenize(s)

		fmt.Print(in)

		_ = causal
	}
}
