package lesci

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
)

func EnsureTable(db *sql.DB, table string, query string) (bool, error) {
	ctx := context.Background()

	var exists bool

	queryTableExists := `SELECT COUNT(*) > 0 FROM information_schema.tables WHERE table_name = ?`

	if err := db.QueryRowContext(ctx, queryTableExists, table).Scan(&exists); err != nil {
		return false, err
	}

	if !exists {
		if _, err := db.ExecContext(ctx, query); err != nil {
			return false, err
		}

		return true, nil
	}

	var count int64

	queryNumRows := fmt.Sprintf(`SELECT COUNT(*) FROM "%s"`, table)

	if err := db.QueryRowContext(ctx, queryNumRows).Scan(&count); err != nil {
		return false, errors.New("table not empty")
	}

	return count == 0, nil
}
