package lesci

import (
	"context"
	"database/sql"
	"fmt"

	"go.jknobloc.com/x/tokenizer/bpe"
)

var sqlPeek = `
		WITH pairs AS (
			SELECT uid, pos, token, logprob,
			       LEAD(token, 1) OVER w AS next_tok
			FROM context
			WINDOW w AS (PARTITION BY uid ORDER BY pos)
		),
		matched AS (
			SELECT p.*, r.c AS merged_as_first
			FROM pairs p
			JOIN oov_rules r ON p.token = r.a AND p.next_tok = r.b
			WHERE r.c >= ?
		)
		SELECT
			m1.token    AS tok_a,
			m1.logprob  AS logprob_a,
			m1.next_tok AS tok_b,
			m2.logprob  AS logprob_b,
			m1.merged_as_first AS tok_c
		FROM matched m1
		JOIN context m2 ON m1.uid = m2.uid AND m1.pos + 1 = m2.pos
		LIMIT ?`

func (e *Experiment) Peek(db *sql.DB, n int) error {
	vocab := bpe.Vocab(e.counterfactual.(*bpe.Tokenizer))

	var rows *sql.Rows

	if r, err := db.QueryContext(context.Background(), sqlPeek, e.cutoff, n); err != nil {
		return err
	} else {
		rows = r
	}

	defer rows.Close()

	for rows.Next() {
		var idA, idB, idC int
		var lpA, lpB float32

		if err := rows.Scan(&idA, &lpA, &idB, &lpB, &idC); err != nil {
			return err
		}

		tokA := vocab[idA]
		tokB := vocab[idB]
		tokC := vocab[idC]

		fmt.Printf("%-20q + %-20q -> %-20q  (%.4f + %.4f = %.4f)\n", tokA, tokB, tokC, lpA, lpB, lpA+lpB)
	}

	return rows.Err()
}
