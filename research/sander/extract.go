package sander

import (
	"database/sql"
	"database/sql/driver"
	"fmt"

	"go.jknobloc.com/x/onnx"
	"go.jknobloc.com/x/research/lesci"
	"go.jknobloc.com/x/tensor"
)

func (e *Experiment) Extract(db *sql.DB) error {
	var data []float32
	var shape []int

	if d, s, err := onnx.ExtractInitializer(e.model, "transformer.wte.weight"); err != nil {
		return err
	} else {
		data = d
		shape = s
	}

	if shape[1] != e.hiddenDim {
		panic("shape mismatch")
	}

	t := tensor.NewDense[float32](shape, data)

	if _, err := db.Exec(fmt.Sprintf(`
	    DROP TABLE IF EXISTS embeddings;
	    CREATE TABLE embeddings (token_id INTEGER, embedding FLOAT[%d], reference BOOLEAN);
	`, e.hiddenDim)); err != nil {
		return err
	}

	return lesci.AppendRows(db, "embeddings", func(appendFunc lesci.AppendFunc) error {
		for i := range shape[0] {
			embedding, ok := t.Select(0, i).Contiguous().Data()

			if !ok {
				panic("")
			}

			_, reference := e.reference[int64(i)]

			if err := appendFunc([]driver.Value{int32(i), embedding, reference}); err != nil {
				return err
			}
		}
		return nil
	})
}
