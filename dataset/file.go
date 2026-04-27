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
	shards         []string
	delimiter      string
	stripDelimiter bool
	err            error
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
		shards:    shards,
		delimiter: "\n",
	}, nil
}

func (f *FileReader) SetDelimiter(delimiter string) {
	f.delimiter = delimiter
}

func (f *FileReader) StripDelimiter(stripDelimiter bool) {
	f.stripDelimiter = stripDelimiter
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

		d := []byte(f.delimiter)
		l := len(d)

		scanner := bufio.NewScanner(file)

		buf := make([]byte, 0, 1024*1024)

		scanner.Buffer(buf, 1024*1024)

		scanner.Split(func(data []byte, atEOF bool) (advance int, token []byte, err error) {
			if atEOF && len(data) == 0 {
				return 0, nil, nil
			}

			if i := bytes.Index(data, d); i >= 0 {
				if f.stripDelimiter {
					if f.delimiter == "\n" && i > 0 && data[i-1] == '\r' {
						return i + l, data[:i-1], nil
					}

					return i + l, data[:i], nil
				}

				return i + l, data[:i+l], nil
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
