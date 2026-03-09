package llm

import "sync"

type Evaluator struct {
	mutex     sync.RWMutex
	models    []Causal
	tokenizer Tokenizer
	results   []result
	jobs      int
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
