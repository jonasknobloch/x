package bpc

import (
	"fmt"
	"gpt2"
	"llm"
	"log"
	"math"
	mbpe "mbpe-dyn"
	"os"
)

var model *gpt2.Model
var tokenizer *mbpe.Tokenizer

var vocab []string

func Run() {
	fmt.Println(compute("foo"))

	if err := model.Destroy(); err != nil {
		log.Fatal(err)
	}
}

func init() {
	initModel()
	initTokenizer()
}

func initModel() {
	m := gpt2.NewModel("../gpt2/models/base/model.onnx", os.Getenv("BPC_CUDA_DEVICE_ID"))

	if err := m.Init(); err != nil {
		log.Fatal(err)
	}

	model = m
}

func initTokenizer() {
	m := mbpe.NewMBPE()

	if err := m.Load("../gpt2/models/base/vocab.json", "../gpt2/models/base/merges.txt"); err != nil {
		panic(err)
	}

	vocab = m.Vocab()

	t := mbpe.NewTokenizer(m)

	byteLevel := mbpe.NewByteLevel(false)

	t.SetPreTokenizer(byteLevel)
	t.SetDecoder(byteLevel)

	tokenizer = t
}

func compute(s string) float32 {
	n := len(s)

	var out float32

	for i := n - 1; i >= 0; i-- {
		b := make([]int, 0)

		for t := range vocab {
			pre := x(s, i+1, n)

			for _, v := range prefix(decode([]int{t})) {
				if v == pre {
					b = append(b, t)
				}
			}
		}

		previous := encode(x(s, 1, i))

		var val float32

		for _, t := range b {
			val += p(previous, []int{t})
		}

		out += p([]int{}, previous) * val
	}

	return out
}

func p(foo, bar []int) float32 {

	tokens := append(foo, bar...)

	if len(tokens) < 2 {
		return 1
	}

	l := make([][]float32, 0, len(tokens))

	if _, err := model.Generate(llm.ToInt64(tokens), 0, &l); err != nil {
		panic(err)
	}

	nll := llm.NegLogLikelihood(l[len(bar):len(tokens)-1], tokens[len(bar)+1:])

	return float32(math.Exp(nll))
}

func x(s string, i, j int) string {
	if j == 0 {
		return ""
	}

	if i < 1 || j < 1 || i > j || j > len(s) {
		panic("invalid indices")
	}

	return s[i-1 : j] // TODO capacity
}

func prefix(s string) []string {
	r := make([]string, len(s))

	for i := range len(s) {
		r[i] = x(s, 1, i+1)
	}

	return r
}

func encode(s string) []int {
	return tokenizer.Tokenize(s)
}

func decode(tokens []int) string {
	s := make([]string, len(tokens))

	for i, t := range tokens {
		s[i] = vocab[t]
	}

	d := tokenizer.Decoder()

	return d.Decode(s)
}

// func continuations(prefix string) [][]int {
// 	r := make([][]int, 0)
//
// 	for _, s := range vocab {
// 		if strings.HasPrefix(s, prefix) {
// 			r = append(r, encode(s))
// 		}
// 	}
//
// 	return r
// }

// func query(context []int, next []int) float64 {
// 	return 0
// }
