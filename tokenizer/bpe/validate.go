package bpe

import (
	"fmt"
	"slices"

	"github.com/jonasknobloch/mbpe"
)

func InitialAlphabet() []rune {
	alphabet := make([]rune, 256)

	bc := mbpe.BytesChar

	for i := 0; i < 256; i++ {
		b := uint8(i)

		runes := []rune(bc[b])

		if len(runes) != 1 {
			panic("unexpected replacement")
		}

		alphabet[i] = runes[0]
	}

	slices.Sort(alphabet)

	return alphabet
}

func UnknownRunes(t *Tokenizer, s string) []rune {
	atoi := make(map[string]int)

	vocab := t.mbpe.Model().(*mbpe.MBPE).Vocab()

	for i, token := range vocab {
		if _, ok := atoi[token]; ok {
			continue
		}

		atoi[token] = i
	}

	unknown := make(map[rune]struct{})

	chunks := t.mbpe.PreTokenizer().PreTokenize(s)

	for _, chunk := range chunks {
		for _, r := range chunk {
			i, ok := atoi[string(r)]

			if !ok {
				fmt.Printf("%d %s %v not in vocabulary\n", i, string(r), []byte(string(r)))

				if _, ok := unknown[r]; ok {
					continue
				}

				unknown[r] = struct{}{}
			}
		}
	}

	result := make([]rune, 0, len(unknown))

	for r := range unknown {
		result = append(result, r)
	}

	slices.Sort(result)

	return result
}

func ByteCoverage(t *Tokenizer) bool {
	atoi := make(map[string]int)

	vocab := t.mbpe.Model().(*mbpe.MBPE).Vocab()

	for i, token := range vocab {
		if _, ok := atoi[token]; ok {
			continue
		}

		atoi[token] = i
	}

	bc := mbpe.BytesChar

	covered := true

	for i := 0; i < 256; i++ {
		c, ok := bc[byte(i)]

		if !ok {
			panic("not in replacement table")
		}

		if _, ok := atoi[c]; !ok {
			fmt.Printf("%d %s %v not in vocabulary\n", i, c, []byte(c))

			covered = false
		}
	}

	return covered
}

func ReachableMerges(t *Tokenizer, merges [][2]string) []bool {
	atoi := make(map[string]int)

	vocab := Vocab(t)

	for i, token := range vocab {
		if _, ok := atoi[token]; ok {
			continue
		}

		atoi[token] = i
	}

	reachable := make(map[string]struct{})

	for _, a := range InitialAlphabet() {
		if _, ok := atoi[string(a)]; ok {
			reachable[string(a)] = struct{}{}
		}
	}

	mask := make([]bool, len(merges))

	for i, merge := range merges {
		_, a := reachable[merge[0]]
		_, b := reachable[merge[1]]

		if !a || !b {
			continue
		}

		reachable[merge[0]+merge[1]] = struct{}{}

		mask[i] = true
	}

	return mask
}
