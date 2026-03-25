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

func NewEvaluator[R any](model Causal, tokenizer Tokenizer, callback func(job Job, logits [][]float32, tokens []int) R) *Evaluator[R] {
	return &Evaluator[R]{
		models:     []Causal{model}, // TODO multiple devices
		tokenizer:  tokenizer,
		batchSize:  1, // TODO arg
		numWorkers: 4, // TODO arg
		jobs:       make(chan batch, 1024),
		results:    make(chan R, 1024),
		callback:   callback,
	}
}

func (e *Evaluator[R]) Results() chan R {
	return e.results
}
