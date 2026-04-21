package dataset

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"path/filepath"
	"slices"
	"strings"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/apache/arrow-go/v18/parquet/file"
	"github.com/apache/arrow-go/v18/parquet/pqarrow"
)

type ParquetReader struct {
	shards     []string
	textColumn string
	batchSize  int64
	err        error
}

func NewParquetReader(name string) (*ParquetReader, error) {
	shards := make([]string, 0)

	if matches, err := filepath.Glob(filepath.Join(name, "*.parquet")); err != nil {
		return nil, err
	} else {
		// skip AppleDouble metadata files
		for _, m := range matches {
			if !strings.HasPrefix(filepath.Base(m), "._") {
				shards = append(shards, m)
			}
		}
	}

	if len(shards) == 0 {
		return nil, fmt.Errorf("no parquet files in %q", name)
	}

	slices.Sort(shards)

	r := &ParquetReader{
		shards:     shards,
		textColumn: "text",
		batchSize:  1024,
	}

	return r, nil
}

func (p *ParquetReader) Num() (int, error) {
	r := 0

	for _, name := range p.shards {
		n, err := numRows(name)

		if err != nil {
			return 0, err
		}

		r += int(n)
	}

	return r, nil
}

func (p *ParquetReader) Err() error {
	return p.err
}

func (p *ParquetReader) Texts() iter.Seq2[int, string] {
	return func(yield func(int, string) bool) {
		n := 0

		for _, name := range p.shards {
			for text := range p.read(name) {
				if !yield(n, text) {
					return
				}

				n++
			}

			if p.err != nil {
				return
			}
		}
	}
}

func (p *ParquetReader) read(name string) iter.Seq[string] {
	return func(yield func(string) bool) {
		var reader *file.Reader

		if r, err := file.OpenParquetFile(name, false); err != nil {
			p.err = err

			return
		} else {
			reader = r
		}

		defer reader.Close()

		var arrowReader *pqarrow.FileReader

		if r, err := pqarrow.NewFileReader(reader, pqarrow.ArrowReadProperties{BatchSize: p.batchSize}, memory.DefaultAllocator); err != nil {
			p.err = err

			return
		} else {
			arrowReader = r
		}

		var schema *arrow.Schema

		if s, err := arrowReader.Schema(); err != nil {
			p.err = err

			return
		} else {
			schema = s
		}

		idxs := schema.FieldIndices(p.textColumn)

		if len(idxs) == 0 {
			p.err = errors.New("unknown column")

			return
		}

		var recordReader pqarrow.RecordReader

		if r, err := arrowReader.GetRecordReader(context.TODO(), []int{idxs[0]}, nil); err != nil {
			p.err = err

			return
		} else {
			recordReader = r
		}

		defer recordReader.Release()

		for recordReader.Next() {
			record := recordReader.RecordBatch()

			text, ok := record.Column(0).(*array.String)

			if !ok {
				p.err = fmt.Errorf("unexpected column type")

				return
			}

			for i := range int(record.NumRows()) {
				if !text.IsValid(i) {
					continue
				}

				if !yield(text.Value(i)) {
					return
				}
			}
		}

		if err := recordReader.Err(); err != nil {
			p.err = err

			return
		}
	}
}

func numRows(name string) (int64, error) {
	var reader *file.Reader

	if r, err := file.OpenParquetFile(name, false); err != nil {
		return 0, err
	} else {
		reader = r

		defer reader.Close()
	}

	return reader.NumRows(), nil
}
