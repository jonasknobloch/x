package main

import (
	"fmt"
	"log"
	"math"

	"github.com/jonasknobloch/mbpe"
	"github.com/jonasknobloch/x/dataset"
	"github.com/jonasknobloch/x/gpt2"
	"github.com/jonasknobloch/x/llm"
)

func main() {
	d := data()
	m := model()
	t := tokenizer()

	e := llm.NewEvaluator()

	e.SetTokenizer(t)
	e.AddModel(m)

	e.Execute = llm.NegLogLikelihood

	if err := e.Run(d, 1024, 512); err != nil {
		log.Fatal(err)
	}

	vals, nums := e.Results()

	total := float64(0)
	n := 0

	for i, v := range vals {
		total += v
		n += nums[i]
	}

	avg := total / float64(n)
	ppl := math.Exp(avg)

	fmt.Println(ppl)
}

func data() *dataset.Reader {
	var miniPile *dataset.Reader

	if r, err := dataset.NewReader("dataset/cmd/dataset/tmp/minipile/validation"); err != nil {
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
