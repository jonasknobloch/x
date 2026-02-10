package bpc

import (
	"fmt"
	"llm"
	"log"
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/jonasknobloch/mbpe"
)

var M *mbpe.MBPE
var B *mbpe.ByteLevel
var T *mbpe.Tokenizer

var Vocab []string

func initVocab() {
	v := M.Vocab()
	r := make([]string, len(v))

	for i, t := range v {
		r[i] = B.Decode([]string{t})
	}

	Vocab = r
}

func encode(s string) []int64 {
	// TODO implement

	return []int64{}
}

func decode(ids []int64) string {
	// TODO implement

	return ""
}

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

func extractCoverProbBytesFoo(raw string) ([]string, []float64, [][]int64, []int) {
	if len(raw) == 0 {
		panic("") // TODO
	}

	condition, query := whitespaceSplit(raw)
	// diverged := []int{} // TODO implement: encode via tokenizer (make sure to apply byte replacements)
	// query := []byte{}   // TODO convert to bytes: again byte replacements ??

	for i := range len(query) {
		left := query[:i]
		right := Supertokens(query[i:], Vocab)

		if len(right) == 0 {
			continue
		}

		foo := condition + left

		if !utf8.ValidString(foo) {
			continue
		}

		fooIDs := encode(foo)

		proposals := make([][]int64, len(right))

		for j, r := range right {
			proposals[j] = append(fooIDs, r)
		}

		valid := make([][]int64, 0)

		for j, ok := range checkCoverEncodings(proposals) {
			if ok {
				valid = append(valid, proposals[j])
			}
		}

		// TODO get likelihoods per encoding

		// TODO assert no duplicate encodings
	}

	return nil, nil, nil, nil
}

func coverTokenLikelihoodsFoo(left, diverged []int64) (float64, []bool) {
	if !slices.Equal(left[:len(diverged)], diverged) {
		// TODO implement
	}

	return 0, nil
}

func runTRFoo() {

}

func logProbsNextToken() {
	// Uses transformer API to compute the logprobs of all tokens for one
	// step. Takes inputs either as string or token ids.

}

func checkCoverEncodings(proposals [][]int64) []bool {
	m := make([]bool, len(proposals))

	for i, p := range proposals {
		gold := encode(decode(p))

		// TODO handle BOS token

		m[i] = slices.Equal(p, gold)
	}

	return m
}

// Supertokens returns a list of token IDs whose string representation starts with prefix.
func Supertokens(prefix string, vocab []string) []int64 {
	ids := make([]int64, 0)

	for i, t := range vocab {
		if t == "<|endoftext|>" {
			continue
		}

		if len(t) == 6 && strings.HasPrefix(t, "<0x") && strings.HasSuffix(t, ">") {
			panic("unimplemented")
		}

		if strings.HasPrefix(t, prefix) {
			ids = append(ids, int64(i))
		}
	}

	return ids
}

func atoi(a string) int64 {
	// TODO implement

	return 0
}

func itoa(i int64) string {
	// TODO implement

	return ""
}

func whitespaceSplit(raw string) (string, string) {
	runes := []rune(raw)

	b := -1

	for i, r := range runes {
		if i == 0 {
			continue
		}

		if r == ' ' && runes[i-1] != ' ' {
			b = i
		}
	}

	if b < 0 {
		panic("") // TODO
	}

	return string(runes[:b]), string(runes[:b])
}
