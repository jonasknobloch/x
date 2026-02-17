package lesci

import (
	"fmt"

	"github.com/jonasknobloch/mbpe"
)

func foo() {
	m := mbpe.NewMBPE()

	if err := m.Load("../gpt2/models/base/vocab.json", "../gpt2/models/base/merges.txt"); err != nil {
		// TODO
	}

	// t := mbpe.NewTokenizer(m)

	vocab := m.Vocab()
	merges := m.Merges()

	// in our case merges -> id is injective
	// lesci used ids so we do the same

	vocabIdx := toIdx(m, vocab)
	mergesIdx, mergedIdx := toIdx2(m, merges)

	a := filterA(vocabIdx)
	b := filterB(a, mergesIdx, mergedIdx)

	fmt.Println(len(vocab), len(a), len(b))

	// TODO refactor to filter merges
	// vocab is not really required

	// [][3]int
	// or [][2]int []int
	// or some kind of struct
}

func toIdx(m *mbpe.MBPE, vocab []string) []int {
	r := make([]int, len(vocab))

	for i, v := range vocab {
		idx := m.Tokenize(v)

		if len(idx) != 1 {
			panic("") // TODO
		}

		r[i] = i
	}

	return r
}

func toIdx2(m *mbpe.MBPE, merges [][2]string) ([][2]int, []int) {
	r := make([][2]int, len(merges))
	v := make([]int, len(merges))

	for i, merge := range merges {
		a := m.Tokenize(merge[0])
		b := m.Tokenize(merge[1])

		c := m.Tokenize(merge[0] + merge[1])

		if len(a) != 1 || len(b) != 1 || len(c) != 1 {
			panic("") // TODO
		}

		r[i], v[i] = [2]int{a[0], b[0]}, c[0]
	}

	return r, v
}

func filterA(vocab []int) []int {
	cutoff := 32000
	window := 5000

	r := make([]int, 0)

	for _, idx := range vocab {
		if (idx >= cutoff-window) && (idx < cutoff+window) {
			r = append(r, idx)
		}
	}

	return r
}

func filterB(vocab []int, merges [][2]int, merged []int) [][2]int {
	intermediate := make(map[int]struct{})

	// TODO check with filter A if merge is relevant for window

	// drop all merges a + b -> c where c is a or b elsewhere in window
}
