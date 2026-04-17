package main

import (
	"fmt"
	"log"
	"path"

	"go.jknobloc.com/x/research/sander"
	"go.jknobloc.com/x/tokenizer/bpe"
)

func main() {
	vocab := []string{"50256"}
	alpha := []string{"m000", "m010", "m020", "m030", "m040", "m050", "m060", "m070", "m080", "m090", "m100"}

	for _, v := range vocab {
		for _, a := range alpha {
			src := fmt.Sprintf("gpt2/models/mbpe/gpt2_%s_%s_babylm_v2", v, a)
			dst := fmt.Sprintf("out/sander/mbpe/gpt2_%s_%s_babylm_v2", v, a)

			if err := run(src, dst); err != nil {
				log.Fatal(err)
			}
		}
	}

}

func run(src, dst string) error {
	t := must(bpe.NewTokenizerFromFiles(path.Join(src, "vocab.json"), path.Join(src, "merges.txt")))
	e := must(sander.NewExperiment(dst, path.Join(src, "model.onnx"), sander.UnusedTokensMBPE(t)))

	return e.Run()
}

func must[T any](v T, err error) T {
	if err != nil {
		log.Fatal(err)
	}

	return v
}
