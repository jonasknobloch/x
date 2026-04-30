package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/jonasknobloch/mbpe"

	"go.jknobloc.com/x/shelf"
)

func serialize() {
	base := shelf.Abs("results/knobloch/minipile")

	var paths []string

	if ps, err := subDirs(base); err != nil {
		log.Fatal(err)
	} else {
		paths = ps
	}

	steps := []int{100512, 50256, 32768, 16384, 8192}

	outRoot := shelf.Abs("tokenizers")

	if err := os.MkdirAll(outRoot, os.ModePerm); err != nil {
		log.Fatal(err)
	}

	for _, step := range steps {
		for _, path := range paths {
			model := mbpe.NewMBPE()

			if err := model.Load(filepath.Join(path, "vocab.json"), filepath.Join(path, "merges.txt")); err != nil {
				log.Fatal(err)
			}

			model.Trim(step)

			dir := fmt.Sprintf("tokenizer_gpt2_%d_%s", step, filepath.Base(path))

			out := filepath.Join(outRoot, dir)

			if err := os.Mkdir(out, os.ModePerm); err != nil {
				log.Fatal(err)
			}

			if err := model.Save(filepath.Join(out, "vocab.json"), filepath.Join(out, "merges.txt")); err != nil {
				log.Fatal(err)
			}

			var config string
			var special string

			if bs, err := os.ReadFile(shelf.Abs("models/gpt2/tokenizer_config.json")); err != nil {
				log.Fatal(bs)
			} else {
				config = string(bs)
			}

			if bs, err := os.ReadFile(shelf.Abs("models/gpt2/special_tokens_map.json")); err != nil {
				log.Fatal(err)
			} else {
				special = string(bs)
			}

			config = strings.Replace(config, "50256", fmt.Sprintf("%d", step), -1)

			if err := os.WriteFile(filepath.Join(out, "tokenizer_config.json"), []byte(config), os.ModePerm); err != nil {
				log.Fatal(err)
			}

			if err := os.WriteFile(filepath.Join(out, "special_tokens_map.json"), []byte(special), os.ModePerm); err != nil {
				log.Fatal(err)
			}
		}
	}
}

func subDirs(base string) ([]string, error) {
	paths := make([]string, 0)

	err := filepath.WalkDir(base, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}

		rel, err := filepath.Rel(base, path)

		if err != nil {
			return err
		}

		depth := strings.Count(rel, string(os.PathSeparator))

		if d.IsDir() && rel != "." && depth == 0 {
			paths = append(paths, path)
		}

		return nil
	})

	return paths, err
}
