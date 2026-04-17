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
	atoi := Atoi(t)

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
	atoi := Atoi(t)

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

func reachableTokens(t *Tokenizer) map[string]struct{} {
	atoi := Atoi(t)

	reachable := make(map[string]struct{})

	for _, a := range InitialAlphabet() {
		if _, ok := atoi[string(a)]; ok {
			reachable[string(a)] = struct{}{}
		}
	}

	for _, merge := range Merges(t) {
		if _, ok := reachable[merge[0]]; !ok {
			continue
		}

		if _, ok := reachable[merge[1]]; !ok {
			continue
		}

		reachable[merge[0]+merge[1]] = struct{}{}
	}

	return reachable
}

func ReachableTokens(t *Tokenizer, vocab []string) []bool {
	reachable := reachableTokens(t)

	mask := make([]bool, len(vocab))

	for i, token := range vocab {
		if _, ok := reachable[token]; !ok {
			continue
		}

		mask[i] = true
	}

	return mask
}

func ReachableMerges(t *Tokenizer, merges [][2]string) []bool {
	reachable := reachableTokens(t)

	mask := make([]bool, len(merges))

	for i, merge := range merges {
		if _, ok := reachable[merge[0]+merge[1]]; !ok {
			continue
		}

		mask[i] = true
	}

	return mask
}

func Atoi(t *Tokenizer) map[string]int64 {
	atoi := make(map[string]int64)

	for i, token := range Vocab(t) {
		if _, ok := atoi[token]; ok {
			panic("duplicate token")
		}

		atoi[token] = int64(i)
	}

	return atoi
}

func Itoa(t *Tokenizer) map[int64]string {
	itoa := make(map[int64]string)

	for i, token := range Vocab(t) {
		itoa[int64(i)] = token
	}

	return itoa
}
