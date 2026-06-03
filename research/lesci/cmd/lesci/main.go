package main

import (
	"flag"
	"log"
	"path"

	"go.jknobloc.com/x/dataset"
	"go.jknobloc.com/x/gpt2"
	"go.jknobloc.com/x/research/lesci"
	"go.jknobloc.com/x/shelf"
	"go.jknobloc.com/x/tokenizer/bpe"
)

func main() {
	obs := flag.String("tok-obs", "", "observed tokenizer")
	ctf := flag.String("tok-ctf", "", "counterfactual tokenizer")

	chkpt := flag.String("c", "", "")

	outPath := flag.String("o", "", "")

	dry := flag.Bool("dry", false, "")

	flag.Parse()

	if flag.NArg() < 1 {
		log.Fatal("usage: lesci [flags] <model>")
	}

	if *obs == "" || *ctf == "" {
		log.Fatal("tokenizers not specified")
	}

	modelPath := flag.Arg(0)

	if err := gpt2.InitializeEnvironment(); err != nil {
		log.Fatal(err)
	}

	m := must(model(path.Join(shelf.Abs(shelf.Item(modelPath)), *chkpt, "model_eval.onnx"), 50256))

	a := shelf.Abs(shelf.Item(*obs))
	b := shelf.Abs(shelf.Item(*ctf))

	cfg := bpe.Config{
		Recover: true,
	}

	t := must(bpe.NewTokenizerFromFiles(path.Join(a, "vocab.json"), path.Join(a, "merges.txt"), cfg))
	c := must(bpe.NewTokenizerFromFiles(path.Join(b, "vocab.json"), path.Join(b, "merges.txt"), cfg))

	d := must(dataset.NewParquetReader(shelf.Abs("data/minipile/test")))

	o := path.Join(shelf.Abs(shelf.Item(*outPath)), path.Base(shelf.Abs(shelf.Item(modelPath))), *chkpt)

	e := must(lesci.NewExperiment(m, t, c, d, o, 50256, 5000))

	if err := m.Init(); err != nil {
		log.Fatal(err)
	}

	if !*dry {
		if err := e.Run(); err != nil {
			log.Fatal(err)
		}
	}

	if err := gpt2.DestroyEnvironment(); err != nil {
		log.Fatal(err)
	}
}

func must[T any](v T, err error) T {
	if err != nil {
		log.Fatal(err)
	}

	return v
}

func model(name string, vocabSize int) (*gpt2.Model, error) {
	cfg := gpt2.ConfigDefault()

	cfg.VocabSize = vocabSize + 1

	m := gpt2.NewModel(name, cfg, gpt2.Options{
		WithCache:    false,
		WithLogits:   false,
		WithLogProbs: true,
	})

	return m, nil
}
