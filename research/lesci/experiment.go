package lesci

import (
	"database/sql"
	"fmt"
	"log"
	"os"
	"path/filepath"

	_ "github.com/duckdb/duckdb-go/v2"

	"go.jknobloc.com/x/dataset"
	"go.jknobloc.com/x/llm"
)

type Experiment struct {
	model          llm.Causal
	tokenizer      llm.Tokenizer
	counterfactual llm.Tokenizer
	data           dataset.Reader
	name           string
	cutoff         int
	window         int
}

func NewExperiment(model llm.Causal, tokenizer, counterfactual llm.Tokenizer, data dataset.Reader, name string, cutoff, window int) (*Experiment, error) {
	e := &Experiment{
		model:          model,
		tokenizer:      tokenizer,
		counterfactual: counterfactual,
		data:           data,
		name:           name,
		cutoff:         cutoff,
		window:         window,
	}

	return e, nil
}

func (e *Experiment) Run() error {
	if err := os.MkdirAll(e.name, 0775); err != nil {
		log.Fatal(err)
	}

	dsn := filepath.Join(e.name, "lesci.db")

	var db *sql.DB

	if database, err := initDatabase(dsn); err != nil {
		return err
	} else {
		db = database
	}

	defer db.Close()

	db.SetMaxOpenConns(1)

	fmt.Println(e.name)

	if err := e.BuildContext(db); err != nil {
		return err
	}

	if err := e.ExtractData(db); err != nil {
		return err
	}

	if err := e.Analyze(db); err != nil {
		return err
	}

	if err := e.Plot(db); err != nil {
		return err
	}

	return nil
}

func initDatabase(dsn string) (*sql.DB, error) {
	var db *sql.DB

	if database, err := sql.Open("duckdb", dsn); err != nil {
		return nil, err
	} else {
		db = database
	}

	if err := db.Ping(); err != nil {
		_ = db.Close()

		return nil, err
	}

	return db, nil
}
