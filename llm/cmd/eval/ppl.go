package main

import (
	"fmt"
	"log"
	"math"
	"strings"

	"go.jknobloc.com/x/dataset"
	"go.jknobloc.com/x/llm"
)

type pplResult struct {
	v float64
	n int
}

func perplexity() {
	d := data()

	m := model()
	t := tokenizer()

	e := llm.NewEvaluator(m, t, func(job llm.Job, logProbs []float32, tokens []int) pplResult {
		total := float64(0)

		n := 0

		for _, p := range logProbs {
			total -= float64(p)

			n++
		}

		return pplResult{
			v: total,
			n: n,
		}
	})

	total := float64(0)
	n := 0

	if err := e.RunAndCollect("Perplexity", d, 1024, 512, func(r pplResult) error {
		total += r.v
		n += r.n

		return nil
	}); err != nil {
		log.Fatal(err)
	}

	avg := total / float64(n)
	ppl := math.Exp(avg)

	fmt.Println(ppl)
}

func joined() dataset.Reader {
	miniPile := data()

	docs := make([]string, 0)

	for _, d := range miniPile.Texts() {
		docs = append(docs, d)
	}

	j := dataset.NewStringReader(strings.Join(docs, "\n\n"))

	return j
}
