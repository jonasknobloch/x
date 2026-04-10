package lesci

import (
	"fmt"

	"go.jknobloc.com/x/tensor"
)

// ExtractData
//
// https://github.com/pietrolesci/tokenisation-bias/blob/376abc0ed6924986cbaf696ea10fdda71e550e45/notebooks/01_extract_data.ipynb
func ExtractData(rules tensor.Dense[int64], valid []bool, cutoff, window int64) []bool {
	shape := rules.Shape()

	if len(shape) != 2 || shape[0] != len(valid) || shape[1] != 3 {
		panic("shape mismatch")
	}

	// clamped := Window(rules, valid, cutoff, window)

	clamped := valid // collect everything for now

	filtered := Filter(rules, clamped, cutoff)
	oov := OutOfVocab(rules, filtered, cutoff)

	num := func(mask []bool) int {
		n := 0

		for _, v := range mask {
			if !v {
				continue
			}

			n++
		}

		return n
	}

	fmt.Println("clamped:", num(clamped))
	fmt.Println("filtered:", num(filtered))
	fmt.Println("oov:", num(oov))

	return oov
}

// Window
//
// # Filter merges based on the window size and vocab size
// merges_df = (
//
//	merges_df.filter((pl.col("tok") < vocab_size + window_size) & (pl.col("tok") >= vocab_size - window_size))
//	.sort("tok")
//	.drop("count")
//
// )
func Window(rules tensor.Dense[int64], mask []bool, cutoff, window int64) []bool {
	shape := rules.Shape()

	if len(shape) != 2 {
		// TODO
	}

	rows := shape[0]
	cols := shape[1]

	if cols != 3 {
		// TODO
	}

	windowed := make([]bool, rows)

	for i, m := range mask {
		if !m {
			continue
		}

		token := rules.At([]int{i, 2})

		if (token >= cutoff-window) && (token < cutoff+window) {
			windowed[i] = true
		}
	}

	return windowed
}

// Filter
//
// # Find tokens (in-vocab) that got merged into others, either as first or second part of the token
// to_drop = pl.concat(
//
//	[
//	    merges_df.filter(pl.col("tok") < vocab_size).join(
//	        merges_df.select(["tok", col]), left_on="tok", right_on=col, how="inner", suffix="_new"
//	    )
//	    for col in ["tok_a", "tok_b"]
//	]
//
// ).select(["tok", "tok_new"])
// print(f"{len(to_drop)} tokens dropped because are part of other tokens in the window (window size: {window_size} * 2)")
// merges_df = merges_df.filter(pl.col("tok").is_in(to_drop["tok"].implode()).not_())
func Filter(rules tensor.Dense[int64], mask []bool, cutoff int64) []bool {
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

		if _, ok := intermediate[c]; ok {
			if c < cutoff {
				continue // only drop in-vocab tokens
			}
		}

		filtered[i] = true
	}

	return filtered
}

// OutOfVocab
//
// # We only need this to get the tokens composing the OOV tokens
// merges_df = merges_df.filter(pl.col("tok") >= vocab_size)  # notice the '='
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

		if c < cutoff {
			continue
		}

		oov[i] = true
	}

	return oov
}
