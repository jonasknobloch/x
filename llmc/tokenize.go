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

func WriteShards(path, name string, size int, docs <-chan []uint32) (int, error) {
	if err := os.MkdirAll(path, os.ModePerm); err != nil {
		return 0, err
	}

	buffer := make([]uint32, 0, size)

	n := 0

	flush := func() error {
		split := "train"

		if n == 0 {
			split = "val"
		}

		shardName := filepath.Join(path, fmt.Sprintf("%s_%s_%06d.bin", name, split, n))

		d := DataFile[uint32]{
			Model:  GPT2,
			Tokens: buffer,
		}

		if numTokens, err := Serialize(&d, shardName); err != nil {
			return err
		} else {
			fmt.Printf("wrote %s (%d tokens)\n", filepath.Base(shardName), numTokens)
		}

		buffer = buffer[:0]

		n++

		return nil
	}

	for tokens := range docs {
		for len(tokens) > 0 {
			space := size - len(buffer)

			if space >= len(tokens) {
				buffer = append(buffer, tokens...)

				break
			}

			buffer = append(buffer, tokens[:space]...)

			tokens = tokens[space:] // carry remainder

			if err := flush(); err != nil {
				return n, err
			}
		}
	}

	if len(buffer) > 0 {
		if err := flush(); err != nil {
			return n, err
		}
	}

	return n, nil
}
