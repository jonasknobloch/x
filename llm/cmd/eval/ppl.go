package main

import (
	"fmt"
	"log"
	"math"
	"strings"
	"sync"

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

	e := llm.NewEvaluator[pplResult]()

	e.SetTokenizer(t)
	e.AddModel(m)

	e.SetCallback(func(job llm.Job, logits [][]float32, tokens []int) pplResult {
		p, n := llm.NegLogLikelihood(logits, tokens)

		return pplResult{
			v: p,
			n: n,
		}
	})

	results := e.Results()

	var wg sync.WaitGroup

	total := float64(0)
	n := 0

	wg.Add(1)

	go func() {
		defer wg.Done()

		for r := range results {
			total += r.v
			n += r.n
		}
	}()

	if err := e.Run(d, 1024, 512); err != nil {
		log.Fatal(err)
	}

	wg.Wait()

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
