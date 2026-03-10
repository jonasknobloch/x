package main

import (
	"context"
	"database/sql"
	"log"

	"github.com/jonasknobloch/x/llm"
	"github.com/jonasknobloch/x/tensor"

	_ "github.com/duckdb/duckdb-go/v2"
)

func lesci() {
	d := data()
	m := model()
	t := tokenizer()

	e := llm.NewEvaluator()

	e.SetTokenizer(t)
	e.AddModel(m)

	// l := tensor.NewDense[float32]([]int{1, 50256})

	//e.Execute = foo(&l)
	e.Execute = bar()

	if err := e.RunDocs(d, 1024, 512); err != nil {
		log.Fatal(err)
	}

	// for row := range e.Foo() {
	// 	fmt.Println(row)
	// }

	// fmt.Println(l)

	baz(e)
}

func foo(l *tensor.Dense[float32]) func([][]float32, []int) float64 {
	return func(logits [][]float32, targets []int) float64 {
		for i, token := range targets {
			idxs := []int{0, token}

			l.Set(idxs, l.At(idxs)+logits[i][token])
		}

		return 0
	}
}

func bar() func([][]float32, []int) float64 {
	return func(logits [][]float32, targets []int) float64 {
		return 0
	}
}

var db *sql.DB

func check(args ...any) {
	err := args[len(args)-1]
	if err != nil {
		panic(err)
	}
}

func baz(e *llm.Evaluator) {
	var err error

	db, err = sql.Open("duckdb", "logprobs.db?access_mode=READ_WRITE")

	if err != nil {
		log.Fatal(err)
	}

	check(db.Ping())

	check(db.ExecContext(context.Background(), "CREATE TABLE context(uid INTEGER, token INTEGER, logprob FLOAT, pos INTEGER)"))

	stmt, err := db.PrepareContext(context.Background(), "INSERT INTO context VALUES(?, ?, ?, ?)")

	check(err)

	defer func() { check(stmt.Close()) }()

	for row := range e.Foo() {
		check(stmt.ExecContext(context.Background(), row.Doc, row.Token, row.Logit, row.Offset))
	}
}
