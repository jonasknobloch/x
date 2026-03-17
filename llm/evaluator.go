package llm

import (
	"sync/atomic"
)

type Evaluator[R any] struct {
	models    []Causal
	tokenizer Tokenizer

	batchSize  int
	numWorkers int

	scheduled atomic.Int64
	completed atomic.Int64

	jobs    chan batch
	results chan R

	callback func(job Job, logits [][]float32, tokens []int) R
}

func NewEvaluator[R any]() *Evaluator[R] {
	return &Evaluator[R]{
		models:     make([]Causal, 0),
		batchSize:  1, // TODO arg
		numWorkers: 4, // TODO arg
		jobs:       make(chan batch, 1024),
		results:    make(chan R, 1024),
	}
}

func (e *Evaluator[R]) AddModel(model Causal) {
	e.models = append(e.models, model)
}

func (e *Evaluator[R]) SetTokenizer(tokenizer Tokenizer) {
	e.tokenizer = tokenizer
}

func (e *Evaluator[R]) SetCallback(callback func(job Job, logits [][]float32, tokens []int) R) {
	e.callback = callback
}

func (e *Evaluator[R]) Results() chan R {
	return e.results
}
