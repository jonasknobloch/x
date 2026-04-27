package dataset

import (
	"bufio"
	"bytes"
	"fmt"
	"iter"
	"os"
	"path/filepath"
	"slices"
)

type FileReader struct {
	shards     []string
	delimiters []string // TODO single delimiter
	err        error
}

func NewFileReader(name, pattern string) (*FileReader, error) {
	var shards []string

	if matches, err := filepath.Glob(filepath.Join(name, pattern)); err != nil {
		return nil, err
	} else {
		shards = matches
	}

	if len(shards) == 0 {
		return nil, fmt.Errorf("no matching files in %q", name)
	}

	slices.Sort(shards)

	return &FileReader{
		shards:     shards,
		delimiters: []string{"\r\n", "\n"},
	}, nil
}

func (f *FileReader) SetDelimiters(delimiters ...string) *FileReader {
	f.delimiters = delimiters

	return f
}

func (f *FileReader) Num() (int, error) {
	return len(f.shards), nil
}

func (f *FileReader) Err() error {
	return f.err
}

func (f *FileReader) Texts() iter.Seq2[int, string] {
	return func(yield func(int, string) bool) {
		for n, name := range f.shards {
			for text := range f.read(name) {
				if f.err != nil {
					return
				}

				if !yield(n, text) {
					return
				}
			}
		}
	}
}

func (f *FileReader) read(name string) iter.Seq[string] {
	return func(yield func(string) bool) {
		var file *os.File

		if r, err := os.Open(name); err != nil {
			f.err = err

			return
		} else {
			file = r
		}

		defer file.Close()

		delimiters := make([][]byte, len(f.delimiters))

		for i, d := range f.delimiters {
			delimiters[i] = []byte(d)
		}

		slices.SortFunc(delimiters, func(a, b []byte) int {
			return len(b) - len(a)
		})

		scanner := bufio.NewScanner(file)

		buf := make([]byte, 0, 1024*1024)

		scanner.Buffer(buf, 1024*1024)

		scanner.Split(func(data []byte, atEOF bool) (advance int, token []byte, err error) {
			if atEOF && len(data) == 0 {
				return 0, nil, nil
			}

			i, l := -1, -1

			for _, d := range delimiters {
				if j := bytes.Index(data, d); j >= 0 && (i < 0 || j < i) {
					i, l = j, len(d)
				}
			}

			if i >= 0 {
				return i + l, data[:i+l], nil // TODO option to consume delimiter
			}

			if atEOF {
				return len(data), data, nil
			}

			return 0, nil, nil
		})

		for scanner.Scan() {
			if !yield(scanner.Text()) {
				return
			}
		}

		if err := scanner.Err(); err != nil {
			f.err = err
		}
	}
}
