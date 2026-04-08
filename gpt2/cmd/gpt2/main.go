package main

import (
	"fmt"
	"log"
	"math"
	"slices"

	"go.jknobloc.com/x/gpt2"
)

func main() {
	if err := gpt2.InitializeEnvironment(); err != nil {
		log.Fatal(err)
	}

	prompt := []int64{464, 2068, 7586}

	generate(prompt) // [-13.483142 -11.277906]
	score(prompt)    // [-13.48314 -11.277912]

	_ = prompt

	if err := gpt2.DestroyEnvironment(); err != nil {
		log.Fatal(err)
	}
}

func generate(prompt []int64) {
	m := gpt2.NewModel("gpt2/models/mbpe_conv/gpt2_8192_m000_babylm_v2/model_cache.onnx", "0", gpt2.DefaultConfig().WithVocabSize(8193), true, true, false)

	if err := m.Init(); err != nil {
		log.Fatal(err)
	}

	logits := make([][]float32, 0)

	if out, err := m.Generate(prompt, 0, &logits); err != nil {
		log.Fatal(err)
	} else {
		fmt.Printf("\n%v\n", out)
	}

	fmt.Println(selectLogProbs(logits[:len(logits)-1], prompt[1:]))

	m.Destroy()
}

func score(prompt []int64) {
	m := gpt2.NewModel("gpt2/models/mbpe_conv/gpt2_8192_m000_babylm_v2/model_eval.onnx", "0", gpt2.DefaultConfig().WithVocabSize(8193), false, false, true)

	if err := m.Init(); err != nil {
		log.Fatal(err)
	}

	logProbs := make([]float32, 0, 2)

	if err := m.Score(prompt, 1, &logProbs); err != nil {
		log.Fatal(err)
	}

	fmt.Println(logProbs)

	m.Destroy()
}

func selectLogProbs(logits [][]float32, tokens []int64) []float32 {
	if len(logits) != len(tokens) {
		panic("length mismatch")
	}

	r := make([]float32, len(tokens))

	for i, token := range tokens {
		logprobs := logSoftmax(logits[i])

		r[i] = logprobs[token]
	}

	return r
}

func logSoftmax(logits []float32) []float32 {
	m := slices.Max(logits)

	s := float32(0.0)
	r := make([]float32, len(logits))

	for i, v := range logits {
		e := float32(math.Exp(float64(v - m)))

		r[i] = v
		s += e
	}

	lse := float32(math.Log(float64(s))) + m

	for i := range r {
		r[i] -= lse
	}

	return r
}
