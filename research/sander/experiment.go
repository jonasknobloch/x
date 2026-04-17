package sander

import (
	"database/sql"
	"fmt"
	"log"
	"os"
	"path/filepath"

	_ "github.com/duckdb/duckdb-go/v2"
)

type Experiment struct {
	name      string
	model     string
	reference map[int64]struct{}
}

func NewExperiment(name, model string, reference map[int64]struct{}) (*Experiment, error) {
	e := &Experiment{
		name:      name,
		model:     model,
		reference: reference,
	}

	return e, nil
}

func (e *Experiment) Run() error {
	if err := os.MkdirAll(e.name, 0775); err != nil {
		log.Fatal(err)
	}

	dsn := filepath.Join(e.name, "wte.db")

	var db *sql.DB

	if database, err := initDatabase(dsn); err != nil {
		return err
	} else {
		db = database
	}

	defer db.Close()

	db.SetMaxOpenConns(1)

	fmt.Println(e.name)

	if _, err := db.Exec(`INSTALL vss; LOAD vss;`); err != nil {
		return err
	}

	if err := e.Extract(db); err != nil {
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
