package sander

import (
	"context"
	"database/sql"
	"fmt"
)

var (
	sqlSimilarity = `
		CREATE TABLE similarity_buffer AS
		WITH centroid AS (
		    SELECT list(avg_val ORDER BY idx)::FLOAT[768] AS vec
		    FROM (
		        SELECT idx, avg(val) AS avg_val
		        FROM embeddings
		        CROSS JOIN LATERAL UNNEST(embedding::FLOAT[]) WITH ORDINALITY AS t(val, idx)
		        WHERE reference = true
		        GROUP BY idx
		    )
		)
		SELECT e.token_id, array_distance(e.embedding, c.vec) AS distance
		FROM embeddings e, centroid c
`
)

func (e *Experiment) Analyze(db *sql.DB) error {
	ctx := context.Background()

	var tx *sql.Tx

	if t, err := db.BeginTx(ctx, nil); err != nil {
		return err
	} else {
		tx = t
	}

	defer tx.Rollback()

	if _, err := tx.ExecContext(ctx, `DROP TABLE IF EXISTS similarity_buffer`); err != nil {
		return err
	}

	if _, err := tx.ExecContext(ctx, sqlSimilarity); err != nil {
		return fmt.Errorf("similarity: %w", err)
	}

	return tx.Commit()
}
