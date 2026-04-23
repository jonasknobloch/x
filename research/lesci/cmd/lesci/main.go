package main

import (
	"fmt"
	"log"
	"path"

	"go.jknobloc.com/x/dataset"
	"go.jknobloc.com/x/gpt2"
	"go.jknobloc.com/x/research/lesci"
	"go.jknobloc.com/x/tokenizer/bpe"
)

func main() {
	if err := gpt2.InitializeEnvironment(); err != nil {
		log.Fatal(err)
	}

	sizes := []int{8192, 16384, 32768, 50256, 100512}

	for i := range len(sizes) - 1 {
		e, m := setup(sizes[i], sizes[len(sizes)-1])

		if err := m.Init(); err != nil {
			log.Fatal(err)
		}

		if err := e.Run(); err != nil {
			log.Fatal(err)
		}

		if err := m.Destroy(); err != nil {
			log.Fatal(err)
		}
	}

	if err := gpt2.DestroyEnvironment(); err != nil {
		log.Fatal(err)
	}
}

func setup(control, treatment int) (*lesci.Experiment, *gpt2.Model) {
	a := fmt.Sprintf("gpt2/models/onnx_eval/gpt2_%d_m000_babylm_v2", control)
	b := fmt.Sprintf("gpt2/models/onnx_eval/gpt2_%d_m000_babylm_v2", 100512)

	m := must(model(path.Join(a, "model_eval.onnx"), control))
	t := must(bpe.NewTokenizerFromFiles(path.Join(a, "vocab.json"), path.Join(a, "merges.txt")))
	c := must(bpe.NewTokenizerFromFiles(path.Join(b, "vocab.json"), path.Join(b, "merges.txt")))
	d := must(dataset.NewFileReader("dataset/cmd/dataset/tmp/babylm/train_100M", "*.train"))

	o := fmt.Sprintf("out/lesci/m000/babylm_%d_%d", control, treatment)

	return must(lesci.NewExperiment(m, t, c, d, o, control, 5000)), m
}

func must[T any](v T, err error) T {
	if err != nil {
		log.Fatal(err)
	}

	return v
}

func model(name string, vocabSize int) (*gpt2.Model, error) {
	cfg := gpt2.DefaultConfig()

	cfg.VocabSize = vocabSize + 1

	m := gpt2.NewModel(name, cfg, gpt2.Options{
		WithCache:    false,
		WithLogits:   false,
		WithLogProbs: true,
	})

	return m, nil
}
