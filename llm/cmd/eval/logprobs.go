package main

import (
	"context"
	"database/sql"
	"log"

	_ "github.com/duckdb/duckdb-go/v2"

	"go.jknobloc.com/x/llm"
)

type logProb struct {
	document int
	token    int
	value    float32
	offset   int
}

type logProbs []logProb

func logprobs() {
	d := data()

	m := model()
	t := tokenizer()

	e := llm.NewEvaluator(m, t, func(j llm.Job, l []float32, tokens []int) logProbs {
		r := make(logProbs, len(tokens))

		for i, token := range tokens {
			r[i] = logProb{
				document: j.Document,
				token:    token,
				value:    l[i],
				offset:   i,
			}
		}

		return r
	}, llm.EvaluatorConfig{
		BatchSize:  1,
		NumWorkers: 4,
	})

	var insertStmt *sql.Stmt

	if stmt, db, err := prepare("logprobs.db?access_mode=READ_WRITE"); err != nil {
		log.Fatal(err)
	} else {
		insertStmt = stmt

		defer db.Close()
		defer stmt.Close()
	}

	if err := e.RunAndCollect("LogProbs", d, 1024, 512, func(r logProbs) error {
		for _, l := range r {
			if err := insert(insertStmt, l); err != nil {
				return err
			}
		}

		return nil
	}); err != nil {
		log.Fatal(err)
	}

	if err := m.Destroy(); err != nil {
		log.Fatal(err)
	}
}

func prepare(name string) (*sql.Stmt, *sql.DB, error) {
	var db *sql.DB

	if database, err := sql.Open("duckdb", name); err != nil {
		return nil, nil, err
	} else {
		db = database
	}

	if err := db.Ping(); err != nil {
		_ = db.Close()

		return nil, nil, err
	}

	queryCreate := "CREATE TABLE context(uid INTEGER, token INTEGER, logProb FLOAT, pos INTEGER)"

	if _, err := db.ExecContext(context.Background(), queryCreate); err != nil {
		_ = db.Close()

		return nil, nil, err
	}

	var stmt *sql.Stmt

	queryInsert := "INSERT INTO context VALUES(?, ?, ?, ?)"

	if statement, err := db.PrepareContext(context.Background(), queryInsert); err != nil {
		_ = db.Close()

		return nil, nil, err
	} else {
		stmt = statement
	}

	return stmt, db, nil
}

func insert(stmt *sql.Stmt, prob logProb) error {
	if _, err := stmt.ExecContext(context.Background(), prob.document, prob.token, prob.value, prob.offset); err != nil {
		return err
	}

	return nil
}
