package lesci

import (
	"fmt"

	"github.com/jonasknobloch/mbpe"
	"github.com/jonasknobloch/x/tensor"
)

func Run() {
	m := mbpe.NewMBPE()

	if err := m.Load("../../gpt2/models/base/vocab.json", "../../gpt2/models/base/merges.txt"); err != nil {
		panic("") // TODO
	}

	merges := m.Merges()

	// gpt2 merge rank -> token id is injective
	// lesci used token ids so we do the same

	rules := Rules(m, merges)

	clamped := Window(rules, 32000, 5000)
	filtered := Filter(rules, clamped)
	oov := OutOfVocab(rules, filtered, 32000)

	elements := func(mask []bool) int {
		n := 0

		for _, v := range mask {
			if !v {
				continue
			}

			n++
		}

		return n
	}

	fmt.Println("clamped:", elements(clamped))
	fmt.Println("filtered:", elements(filtered))
	fmt.Println("oov:", elements(oov))

	return
}

func Rules(m *mbpe.MBPE, merges [][2]string) tensor.Dense[int64] {
	out := tensor.NewDense[int64]([]int{50000, 3})

	for i, merge := range merges {
		a := m.Tokenize(merge[0])
		b := m.Tokenize(merge[1])

		c := m.Tokenize(merge[0] + merge[1])

		if len(a) != 1 || len(b) != 1 || len(c) != 1 {
			panic("") // TODO
		}

		out.Set([]int{i, 0}, int64(a[0]))
		out.Set([]int{i, 1}, int64(b[0]))
		out.Set([]int{i, 2}, int64(c[0]))
	}

	return out
}

// func Mask(src tensor.Dense[int64]) tensor.Dense[int64] {
// 	mask := src.Contiguous()
//
// 	buffer := make([]int, src.Rank())
//
// 	for idxs := range src.All(buffer) {
// 		mask.Set(idxs, 1)
// 	}
//
// 	return mask
// }

func Window(rules tensor.Dense[int64], cutoff, window int64) []bool {
	shape := rules.Shape()

	if len(shape) != 2 {
		// TODO
	}

	rows := shape[0]
	cols := shape[1]

	if cols != 3 {
		// TODO
	}

	mask := make([]bool, rows)

	for i := range rows {
		token := rules.At([]int{i, 2})

		if (token >= cutoff-window) && (int64(i) < cutoff+window) {
			mask[i] = true
		}
	}

	return mask
}

func Filter(rules tensor.Dense[int64], mask []bool) []bool {
	shape := rules.Shape()

	if len(shape) != 2 {
		// TODO
	}

	rows := shape[0]
	cols := shape[1]

	if len(mask) != rows {
		panic("") // TODO
	}

	if cols != 3 {
		panic("") // TODO
	}

	// drop all rules a + b -> c where c is a or b elsewhere in window

	intermediate := make(map[int64]struct{})

	for i, m := range mask {
		if !m {
			continue
		}

		var a, b int64

		if row, ok := rules.Select(0, i).Contiguous().Data(); !ok {
			panic("") // TODO
		} else {
			a, b = row[0], row[1]
		}

		if _, ok := intermediate[a]; !ok {
			intermediate[a] = struct{}{}
		}

		if _, ok := intermediate[b]; !ok {
			intermediate[b] = struct{}{}
		}
	}

	filtered := make([]bool, rows)

	for i, m := range mask {
		if !m {
			continue
		}

		c := rules.At([]int{i, 2})

		if _, ok := intermediate[c]; !ok {
			continue
		}

		filtered[i] = true
	}

	return filtered
}

func OutOfVocab(rules tensor.Dense[int64], mask []bool, cutoff int64) []bool {
	shape := rules.Shape()

	if len(shape) != 2 {
		// TODO
	}

	rows := shape[0]
	cols := shape[1]

	if len(mask) != rows {
		panic("") // TODO
	}

	if cols != 3 {
		panic("") // TODO
	}

	oov := make([]bool, rows)

	for i, m := range mask {
		if !m {
			continue
		}

		c := rules.At([]int{i, 2})

		if c >= cutoff {
			continue
		}

		oov[i] = true
	}

	return oov
}
