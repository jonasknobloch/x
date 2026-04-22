package llmc

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"go.jknobloc.com/x/dataset"
	"go.jknobloc.com/x/llm"
)

func TokenizeAll(reader dataset.Reader, tok llm.Tokenizer, eot int) <-chan []uint32 {
	out := make(chan []uint32, 256)

	go func() {
		defer close(out)

		sem := make(chan struct{}, max(1, runtime.NumCPU()-1))

		var wg sync.WaitGroup

		for _, text := range reader.Texts() {
			sem <- struct{}{}

			wg.Add(1)

			go func(t string) {
				defer func() { <-sem; wg.Done() }()

				tokens := tok.Tokenize(t)

				tokensU32 := make([]uint32, len(tokens)+1)

				tokensU32[0] = uint32(eot)

				for i, id := range tokens {
					tokensU32[i+1] = uint32(id)
				}

				out <- tokensU32
			}(text)
		}

		wg.Wait()
	}()

	return out
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
