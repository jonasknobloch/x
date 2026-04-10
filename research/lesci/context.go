package lesci

import (
	"database/sql"
	"database/sql/driver"
	"fmt"

	"go.jknobloc.com/x/llm"
)

type logProb struct {
	document int
	token    int
	value    float32
	offset   int
}

func (e *Experiment) BuildContext(db *sql.DB) error {
	eval := llm.NewEvaluator(e.model, e.tokenizer, func(job llm.Job, logProbs []float32, tokens []int) []logProb {
		r := make([]logProb, len(tokens))

		for i, token := range tokens {
			r[i] = logProb{
				document: job.Document,
				token:    token,
				value:    logProbs[i],
				offset:   job.Position*512 + job.Seen + i, // TODO refactor
			}
		}

		return r
	}, llm.EvaluatorConfig{
		BatchSize:  32,
		NumWorkers: 64,
	})

	if ok, err := EnsureTable(db, "context", `CREATE TABLE context(uid INTEGER, token INTEGER, logprob FLOAT, pos INTEGER)`); err != nil {
		return err
	} else if !ok {
		fmt.Println("context table not empty")

		return nil
	}

	return AppendRows(db, "context", func(append AppendFunc) error {
		return eval.RunAndCollect("Context", e.data, 1024, 512, func(r []logProb) error {
			for _, l := range r {
				if err := append([]driver.Value{l.document, l.token, l.value, l.offset}); err != nil {
					return err
				}
			}

			return nil
		})
	})
}
