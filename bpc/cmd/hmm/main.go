package main

import (
	"fmt"

	"github.com/jonasknobloch/2021ss-smt/pkg/hmm"
)

func main() {
	Q := []string{"q1", "q2"}
	V := []string{"A", "B"}

	h := hmm.NewHMM(Q, V, "#")

	gamma := 0.5
	alpha := 0.5
	beta := 0.5

	h.PTransition["#"]["q1"] = gamma
	h.PTransition["#"]["q2"] = 1 - gamma

	h.PTransition["q1"]["q1"] = alpha
	h.PTransition["q1"]["q2"] = 1 - alpha
	h.PTransition["q2"]["q1"] = beta
	h.PTransition["q2"]["q2"] = 1 - beta

	h.PEmission["q1"]["A"] = 1
	h.PEmission["q2"]["B"] = 1

	if err := h.Validate(); err != nil {
		fmt.Println(err)

		return
	}

	_, t := h.Forward([]string{"A", "B"})

	fmt.Println(t)

	last := t[len(t)]

	p := 0.0

	for q := range last {
		p += last[q]
	}

	fmt.Println(p)
}
