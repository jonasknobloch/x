package main

import (
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/jonasknobloch/mbpe"

	"go.jknobloc.com/x/dataset"
	"go.jknobloc.com/x/profile"
	"go.jknobloc.com/x/shelf"
	"go.jknobloc.com/x/tokenizer/bpe"
	"go.jknobloc.com/x/tokenizer/bpe/split"
)

func main() {
	stop := profile.CPU()

	// train()
	// serialize()

	profile.Mem()

	stop()
}

func train() {
	out := shelf.Abs("results/knobloch/minipile")

	if err := os.MkdirAll(out, os.ModePerm); err != nil {
		log.Fatal(err)
	}

	morfessor := func() mbpe.Segmenter {
		m := mbpe.NewMorfessor()

		if err := m.LoadModel(shelf.Abs("morfessor/semisup_model.proto")); err != nil {
			log.Fatal(err)
		}

		return m
	}()

	// mbpe.InvertWeightFunction = true
	// mbpe.UseSimpleClashes = true

	newTrainer := func(segmenter mbpe.Segmenter, alpha float64) *mbpe.MBPETrainer {
		alphabet := make(map[string]struct{})

		for _, r := range bpe.InitialAlphabet() {
			alphabet[string(r)] = struct{}{}
		}

		b := mbpe.NewByteLevel(true)

		b.SetMatcher(split.NewFSA())

		return mbpe.NewMBPETrainer(b, segmenter, alpha, mbpe.NewMBPE(), 1<<17, alphabet)
	}

	configs := []struct {
		float64
		string
	}{
		{0.0, "m000_minipile_v2"},
		{0.1, "m010_minipile_v2"},
		{0.2, "m020_minipile_v2"},
		{0.3, "m030_minipile_v2"},
		{0.4, "m040_minipile_v2"},
		{0.5, "m050_minipile_v2"},
		{0.6, "m060_minipile_v2"},
		{0.7, "m070_minipile_v2"},
		{0.8, "m080_minipile_v2"},
		{0.9, "m090_minipile_v2"},
		{1.0, "m100_minipile_v2"},
	}

	for i, c := range configs {
		t := newTrainer(morfessor, c.float64)

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

		fmt.Printf("%s\n\n", c.string)

		saveSegments := false

		if fileOpen, errOpen := os.Open(filepath.Join(out, "segments.gob")); errOpen != nil {
			saveSegments = true
		} else {
			defer fileOpen.Close()

			if err := t.LoadSegments(fileOpen); err != nil {
				log.Fatal(err)
			}
		}

		t.Train()

		if saveSegments {
			if fileCreate, errCreate := os.Create(filepath.Join(out, "segments.gob")); errCreate != nil {
				log.Fatal(errCreate)
			} else {
				defer fileCreate.Close()

				if err := t.SaveSegments(fileCreate); err != nil {
					log.Fatal(err)
				}
			}
		}

		dir := filepath.Join(out, c.string)

		if err := os.Mkdir(dir, 0755); err != nil && !errors.Is(err, os.ErrExist) {
			log.Fatal(err)
		}

		if err := t.Model().Save(filepath.Join(dir, "vocab.json"), filepath.Join(dir, "merges.txt")); err != nil {
			log.Fatal(err)
		}
	}
}
