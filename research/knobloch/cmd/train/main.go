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
	}

	// mbpe.InvertWeightFunction = true
	// mbpe.UseSimpleClashes = true

	m000 := morfessor()
	m010 := morfessor()
	m020 := morfessor()
	m030 := morfessor()
	m040 := morfessor()
	m050 := morfessor()
	m060 := morfessor()
	m070 := morfessor()
	m080 := morfessor()
	m090 := morfessor()
	m100 := morfessor()

	newTrainer := func(segmenter mbpe.Segmenter, alpha float64) *mbpe.MBPETrainer {
		alphabet := make(map[string]struct{})

		for _, r := range bpe.InitialAlphabet() {
			alphabet[string(r)] = struct{}{}
		}

		b := mbpe.NewByteLevel(true)

		b.SetMatcher(split.NewFSA())

		return mbpe.NewMBPETrainer(b, segmenter, alpha, mbpe.NewMBPE(), 1<<17, alphabet)
	}

	trainers := []struct {
		*mbpe.MBPETrainer
		string
	}{
		{newTrainer(m000, 0.0), "m000_minipile_v2"},
		{newTrainer(m010, 0.1), "m010_minipile_v2"},
		{newTrainer(m020, 0.2), "m020_minipile_v2"},
		{newTrainer(m030, 0.3), "m030_minipile_v2"},
		{newTrainer(m040, 0.4), "m040_minipile_v2"},
		{newTrainer(m050, 0.5), "m050_minipile_v2"},
		{newTrainer(m060, 0.6), "m060_minipile_v2"},
		{newTrainer(m070, 0.7), "m070_minipile_v2"},
		{newTrainer(m080, 0.8), "m080_minipile_v2"},
		{newTrainer(m090, 0.9), "m090_minipile_v2"},
		{newTrainer(m100, 1.0), "m100_minipile_v2"},
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

		if fileOpen, errOpen := os.Open(filepath.Join(out, "segments.gob")); errOpen != nil {
			if fileCreate, errCreate := os.Create(filepath.Join(out, "segments.gob")); errCreate != nil {
				log.Fatal(errCreate)
			} else {
				defer fileCreate.Close()

				if err := t.SaveSegments(fileCreate); err != nil {
					log.Fatal(errOpen)
				}
			}
		} else {
			defer fileOpen.Close()

			if err := t.LoadSegments(fileOpen); err != nil {
				log.Fatal(errOpen)
			}
		}

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
