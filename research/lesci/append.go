package lesci

import (
	"context"
	"database/sql"
	"database/sql/driver"

	"github.com/duckdb/duckdb-go/v2"
)

type AppendFunc func([]driver.Value) error
type AppendRowsFunc func(appendFunc AppendFunc) error

func AppendRows(db *sql.DB, table string, appendRowsFunc AppendRowsFunc) error {
	var conn *sql.Conn

	if c, err := db.Conn(context.Background()); err != nil {
		return err
	} else {
		conn = c
	}

	defer conn.Close()

	return conn.Raw(func(driverConn any) error {
		duckConn := driverConn.(*duckdb.Conn)

		var appender *duckdb.Appender

		if a, err := duckdb.NewAppenderFromConn(duckConn, "", table); err != nil {
			return err
		} else {
			appender = a
		}

		defer appender.Close()

		if err := appendRowsFunc(func(row []driver.Value) error {
			return appender.AppendRow(row...)
		}); err != nil {
			return err
		}

		return appender.Flush()
	})
}
