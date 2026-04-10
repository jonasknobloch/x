package lesci

import (
	"context"
	"database/sql"
	"database/sql/driver"
	"fmt"

	"go.jknobloc.com/x/llm"
	"go.jknobloc.com/x/tensor"
	"go.jknobloc.com/x/tokenizer/bpe"
)

func (e *Experiment) ExtractData(db *sql.DB) error {
	if ok, err := EnsureTable(db, "oov_rules", `CREATE TABLE oov_rules (a INTEGER, b INTEGER, c INTEGER)`); err != nil {
		return err
	} else if !ok {
		fmt.Println("oov_rules table not empty")

		if _, err := db.ExecContext(context.Background(), `DELETE FROM oov_rules`); err != nil {
			return err
		}
	}

	merges := bpe.Merges(e.counterfactual.(*bpe.Tokenizer))

	rules, valid := Rules(e.counterfactual, merges)

	mask := ExtractData(rules, valid, int64(e.cutoff), int64(e.window))

	return AppendRows(db, "oov_rules", func(append AppendFunc) error {
		for i, m := range mask {
			if !m {
				continue
			}

			if err := append([]driver.Value{
				rules.At([]int{i, 0}),
				rules.At([]int{i, 1}),
				rules.At([]int{i, 2}),
			}); err != nil {
				return err
			}
		}

		return nil
	})
}

func Rules(tokenizer llm.Tokenizer, merges [][2]string) (tensor.Dense[int64], []bool) {
	valid := bpe.ReachableMerges(tokenizer.(*bpe.Tokenizer), merges)

	rules := tensor.NewDense[int64]([]int{len(merges), 3}, nil)

	for i, merge := range merges {
		if !valid[i] {
			continue
		}

		a := tokenizer.Tokenize(merge[0])
		b := tokenizer.Tokenize(merge[1])
		c := tokenizer.Tokenize(merge[0] + merge[1])

		if len(a) != 1 || len(b) != 1 || len(c) != 1 {
			panic("unexpected token IDs")
		}

		rules.Set([]int{i, 0}, int64(a[0]))
		rules.Set([]int{i, 1}, int64(b[0]))
		rules.Set([]int{i, 2}, int64(c[0]))
	}

	return rules, valid
}
