package llm

import (
	"math"
	"slices"
)

func NegLogLikelihood(logits [][]float32, targets []int) (float64, int) {
	if len(logits) != len(targets) {
		panic("mismatched input lengths")
	}

	total := float64(0)

	for i, target := range targets {
		maxLogit := float64(slices.Max(logits[i]))

		sumExp := float64(0)

		for _, v := range logits[i] {
			sumExp += math.Exp(float64(v) - maxLogit)
		}

		logSumExp := maxLogit + math.Log(sumExp)

		targetLogit := float64(logits[i][target])

		logProb := targetLogit - logSumExp

		total -= logProb
	}

	return total, len(targets)
}
