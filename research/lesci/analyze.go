package lesci

import (
	"context"
	"database/sql"
	"fmt"
)

var (
	sqlPairs = `
		CREATE TEMP TABLE pairs AS
		SELECT uid, pos, token, logprob,
		       LEAD(token, 1) OVER w AS next_tok,
		       LAG(token, 1)  OVER w AS prev_tok
		FROM context
		WINDOW w AS (PARTITION BY uid ORDER BY pos)`

	sqlMatched = `
		CREATE TEMP TABLE matched AS
		SELECT p.*, r1.c AS merged_as_first, r2.c AS merged_as_second
		FROM pairs p
		LEFT JOIN oov_rules r1 ON p.token = r1.a AND p.next_tok = r1.b
		LEFT JOIN oov_rules r2 ON p.prev_tok = r2.a AND p.token = r2.b`

	sqlSummedOOV = `
		CREATE TEMP TABLE summed_oov AS
		WITH processed AS (
		    SELECT
		        uid,
		        logprob,
		        COALESCE(merged_as_first, merged_as_second, token) AS tok,
		        CASE
		            WHEN merged_as_first  IS NOT NULL THEN pos
		            WHEN merged_as_second IS NOT NULL THEN pos - 1
		            ELSE pos
		        END AS adjusted_pos
		    FROM matched
		)
		SELECT
		    uid,
		    tok,
		    adjusted_pos AS pos,
		    SUM(logprob) AS token_logprob
		FROM processed
		WHERE tok >= ? AND tok < ?
		GROUP BY uid, tok, adjusted_pos`

	sqlResults = `
		CREATE TABLE lesci_results AS
		SELECT tok,
		       QUANTILE_CONT(token_logprob, 0.5)  AS median,
		       QUANTILE_CONT(token_logprob, 0.75) AS q75,
		       QUANTILE_CONT(token_logprob, 0.25) AS q25,
		       QUANTILE_CONT(token_logprob, 0.75)
		         - QUANTILE_CONT(token_logprob, 0.25) AS iqr,
		       AVG(token_logprob)        AS mean,
		       STDDEV_SAMP(token_logprob) AS std,
		       COUNT(*)                  AS num,
		       tok < ?                   AS treat
		FROM summed_oov
		GROUP BY tok`
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

	if _, err := tx.ExecContext(ctx, `DROP TABLE IF EXISTS lesci_results`); err != nil {
		return err
	}

	if _, err := tx.ExecContext(ctx, sqlPairs); err != nil {
		return fmt.Errorf("pairs: %w", err)
	}

	if _, err := tx.ExecContext(ctx, sqlMatched); err != nil {
		return fmt.Errorf("matched: %w", err)
	}

	low := e.cutoff - e.window
	high := e.cutoff + e.window

	if e.window == -1 {
		low = 0
		high = 10000000 // TODO use counterfactual
	}

	if _, err := tx.ExecContext(ctx, sqlSummedOOV, low, high); err != nil {
		return fmt.Errorf("summed_oov: %w", err)
	}

	if _, err := tx.ExecContext(ctx, sqlResults, e.cutoff); err != nil {
		return fmt.Errorf("lesci_results: %w", err)
	}

	return tx.Commit()
}
