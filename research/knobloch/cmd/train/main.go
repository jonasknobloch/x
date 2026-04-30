package main

import (
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/jonasknobloch/mbpe"

	"go.jknobloc.com/x/dataset"
	"go.jknobloc.com/x/shelf"
	"go.jknobloc.com/x/tokenizer/bpe"
	"go.jknobloc.com/x/tokenizer/bpe/split"
)

func main() {
	// train()
	// serialize()
}

func train() {
	out := shelf.Abs("results/knobloch/minipile")

	if err := os.MkdirAll(out, os.ModePerm); err != nil {
		log.Fatal(err)
	}

	morfessor := func(alpha float64) mbpe.Segmenter {
		m := mbpe.NewMorfessor(alpha)

		if err := m.LoadModel(shelf.Abs("morfessor/semisup_model.proto")); err != nil {
			log.Fatal(err)
		}

		return m
	}

	// mbpe.InvertWeightFunction = true

	m000 := morfessor(0.0)
	m010 := morfessor(0.1)
	m020 := morfessor(0.2)
	m030 := morfessor(0.3)
	m040 := morfessor(0.4)
	m050 := morfessor(0.5)
	m060 := morfessor(0.6)
	m070 := morfessor(0.7)
	m080 := morfessor(0.8)
	m090 := morfessor(0.9)
	m100 := morfessor(1.0)

	newTrainer := func(segmenter mbpe.Segmenter) *mbpe.MBPETrainer {
		alphabet := make(map[string]struct{})

		for _, r := range bpe.InitialAlphabet() {
			alphabet[string(r)] = struct{}{}
		}

		b := mbpe.NewByteLevel(true)

		b.SetMatcher(split.NewFSA())

		return mbpe.NewMBPETrainer(b, segmenter, mbpe.NewMBPE(), 1<<17, alphabet)
	}

	trainers := []struct {
		*mbpe.MBPETrainer
		string
	}{
		{newTrainer(m000), "m000_minipile"},
		{newTrainer(m010), "m010_minipile"},
		{newTrainer(m020), "m020_minipile"},
		{newTrainer(m030), "m030_minipile"},
		{newTrainer(m040), "m040_minipile"},
		{newTrainer(m050), "m050_minipile"},
		{newTrainer(m060), "m060_minipile"},
		{newTrainer(m070), "m070_minipile"},
		{newTrainer(m080), "m080_minipile"},
		{newTrainer(m090), "m090_minipile"},
		{newTrainer(m100), "m100_minipile"},
	}

	for i, t := range trainers {
		dict := filepath.Join(out, "dict.txt")

		if dictErr := t.LoadDict(dict); dictErr != nil {
			var reader dataset.Reader

			if r, err := dataset.NewParquetReader(shelf.Abs("data/minipile/train")); err != nil {
				log.Fatal(err)
			} else {
				reader = r
			}

			if err := t.InitDict(reader); err != nil {
				log.Fatal(err)
			}

			if err := t.Dict().Save(dict); err != nil {
				log.Fatal(err)
			}
		}

		if i > 0 {
			fmt.Println()
		}

		fmt.Printf("%s\n\n", t.string)

		t.Train()

		dir := filepath.Join(out, t.string)

		if err := os.Mkdir(dir, 0755); err != nil && !errors.Is(err, os.ErrExist) {
			log.Fatal(err)
		}

		if err := t.Model().Save(filepath.Join(dir, "vocab.json"), filepath.Join(dir, "merges.txt")); err != nil {
			log.Fatal(err)
		}
	}
}
