package dataset

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"path/filepath"
	"slices"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/apache/arrow-go/v18/parquet/file"
	"github.com/apache/arrow-go/v18/parquet/pqarrow"
)

type Reader struct {
	shards    []string
	batchSize int64
	err       error
}

func NewReader(name string) (*Reader, error) {
	var shards []string

	if matches, err := filepath.Glob(filepath.Join(name, "*.parquet")); err != nil {
		return nil, err
	} else {
		shards = matches
	}

	if len(shards) == 0 {
		return nil, fmt.Errorf("no parquet files in %q", name)
	}

	slices.Sort(shards)

	r := &Reader{
		shards:    shards,
		batchSize: 1024,
	}

	return r, nil
}

func (r *Reader) Err() error {
	return r.err
}

func (r *Reader) Texts(column string) iter.Seq[string] {
	return func(yield func(string) bool) {
		for _, name := range r.shards {
			err := read(name, column, r.batchSize, yield)

			if errors.Is(err, stop) {
				return
			}

			if err != nil {
				r.err = err

				return
			}
		}
	}
}

var stop = errors.New("stop")

func read(name, column string, batchSize int64, yield func(string) bool) error {
	var reader *file.Reader

	if r, err := file.OpenParquetFile(name, false); err != nil {
		return err
	} else {
		reader = r

		defer reader.Close()
	}

	var arrowReader *pqarrow.FileReader

	if r, err := pqarrow.NewFileReader(reader, pqarrow.ArrowReadProperties{BatchSize: batchSize}, memory.DefaultAllocator); err != nil {
		return err
	} else {
		arrowReader = r
	}

	var schema *arrow.Schema

	if s, err := arrowReader.Schema(); err != nil {
		return err
	} else {
		schema = s
	}

	idxs := schema.FieldIndices(column)

	if len(idxs) == 0 {
		return errors.New("unknown column")
	}

	var recordReader pqarrow.RecordReader

	if r, err := arrowReader.GetRecordReader(context.TODO(), []int{idxs[0]}, nil); err != nil {
		return err
	} else {
		recordReader = r

		defer recordReader.Release()
	}

	for recordReader.Next() {
		record := recordReader.RecordBatch()

		text, ok := record.Column(0).(*array.String)

		if !ok {
			return fmt.Errorf("unexpected column type")
		}

		for i := range int(record.NumRows()) {
			if !text.IsValid(i) {
				continue
			}

			if !yield(text.Value(i)) {
				return stop
			}
		}
	}

	return nil
}
