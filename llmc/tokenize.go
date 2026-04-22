package llmc

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"go.jknobloc.com/x/dataset"
	"go.jknobloc.com/x/llm"
)

func TokenizeAll(reader dataset.Reader, tok llm.Tokenizer, eot int) <-chan []uint32 {
	texts := func(yield func(string) bool) {
		for _, text := range reader.Texts() {
			if !yield(text) {
				return
			}
		}
	}

	return Pool(texts, max(1, runtime.NumCPU()-1), func(text string) []uint32 {
		ids := tok.Tokenize(text)

		tokens := make([]uint32, 0, len(ids)+1)

		tokens = append(tokens, uint32(eot))

		for _, id := range ids {
			tokens = append(tokens, uint32(id))
		}

		return tokens
	})
}

func WriteShards(name, data string, shardSize int, docs <-chan []uint32) error {
	if err := os.MkdirAll(name, os.ModePerm); err != nil {
		return err
	}

	buffer := make([]uint32, 0, shardSize)

	shardIdx := 0

	flush := func() error {
		split := "train"

		if shardIdx == 0 {
			split = "val"
		}

		path := filepath.Join(name, fmt.Sprintf("%s_%s_%06d.bin", data, split, shardIdx))

		d := DataFile[uint32]{
			Model:  GPT2,
			Tokens: buffer,
		}

		if n, err := Serialize(&d, path); err != nil {
			return err
		} else {
			fmt.Printf("wrote %s (%d tokens)\n", filepath.Base(path), n)
		}

		buffer = buffer[:0]

		shardIdx++

		return nil
	}

	for tokens := range docs {
		for len(tokens) > 0 {
			space := shardSize - len(buffer)

			if space >= len(tokens) {
				buffer = append(buffer, tokens...)

				break
			}

			buffer = append(buffer, tokens[:space]...)

			tokens = tokens[space:] // carry remainder

			if err := flush(); err != nil {
				return err
			}
		}
	}

	if len(buffer) > 0 {
		return flush()
	}

	return nil
}
