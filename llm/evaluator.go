package llm

import (
	"iter"
	"sync"
)

type Evaluator struct {
	mutex     sync.RWMutex
	models    []Causal
	tokenizer Tokenizer
	results   []result
	jobs      int
	Execute   func(logits [][]float32, tokens []int) float64
}

func NewEvaluator() *Evaluator {
	return &Evaluator{
		models:  make([]Causal, 0),
		results: make([]result, 0),
	}
}

func (e *Evaluator) AddModel(model Causal) {
	e.models = append(e.models, model)
}

func (e *Evaluator) SetTokenizer(tokenizer Tokenizer) {
	e.tokenizer = tokenizer
}

func (e *Evaluator) Results() ([]float64, []int) {
	v := make([]float64, len(e.results))
	n := make([]int, len(e.results))

	for i, r := range e.results {
		v[i] = r.value
		n[i] = r.n
	}

	return v, n
}

type RRow struct {
	Doc    int
	Token  int
	Logit  float32
	Offset int
}

func (e *Evaluator) Foo() iter.Seq[RRow] {
	return func(yield func(row RRow) bool) {
		for _, r := range e.results {
			for j := range r.tokens {
				row := RRow{
					Doc:    r.doc,
					Token:  r.tokens[j],
					Logit:  r.logits[j],
					Offset: j,
				}

				if !yield(row) {
					return
				}
			}
		}
	}
}
