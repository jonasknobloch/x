package bpc

import (
	"fmt"
	"llm"
	"log"
)

func Run(model llm.Causal, tokenizer llm.Tokenizer) {
	e := llm.NewEvaluator()

	e.AddModel(model)
	e.SetTokenizer(tokenizer)

	ppl, err := e.Perplexity("data/shakespeare.txt")

	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("\nPerplexity: %.2f\n", ppl)
}
