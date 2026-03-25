package llm

import (
	"errors"
	"sync"
	"sync/atomic"

	"go.jknobloc.com/x/dataset"
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

func (e *Evaluator[R]) RunAndCollect(title string, data dataset.Reader, window, stride int, callback func(R) error) error {
	var wg sync.WaitGroup

	var collectErr error

	wg.Add(1)

	go func() {
		defer wg.Done()

		for r := range e.results {
			if err := callback(r); err != nil && collectErr == nil {
				collectErr = err
			}
		}
	}()

	runErr := e.Run(title, data, window, stride)

	wg.Wait()

	return errors.Join(runErr, collectErr)
}
