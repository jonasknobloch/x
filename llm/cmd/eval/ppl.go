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

	e := llm.NewEvaluator(m, t, func(job llm.Job, logits [][]float32, tokens []int) pplResult {
		p, n := llm.NegLogLikelihood(logits, tokens)

		return pplResult{
			v: p,
			n: n,
		}
	})

	total := float64(0)
	n := 0

	if err := e.RunAndCollect(d, 1024, 512, func(r pplResult) error {
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

	for d := range miniPile.Texts("text") {
		docs = append(docs, d)
	}

	j := dataset.NewStringReader(strings.Join(docs, "\n\n"))

	return j
}
