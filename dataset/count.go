package dataset

import (
	"bytes"
	"context"
	"io"
	"os"
	"runtime"
	"sync/atomic"

	"golang.org/x/sync/errgroup"
)

func countLinesAll(names []string, delimiter []byte) (int, error) {
	g, ctx := errgroup.WithContext(context.Background())

	g.SetLimit(runtime.NumCPU() - 1)

	var total atomic.Int64

	for _, name := range names {
		g.Go(func() error {
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
			}

			var file *os.File

			if f, err := os.Open(name); err != nil {
				return err
			} else {
				file = f
			}

			if n, err := countLines(file, delimiter); err != nil {
				return err
			} else {
				total.Add(n)
			}

			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return 0, err
	}

	return int(total.Load()), nil
}

func countLines(r io.Reader, delimiter []byte) (int64, error) {
	if len(delimiter) == 0 {
		panic("empty delimiter")
	}

	l := len(delimiter)

	buf := make([]byte, 128*1024)

	var total int64

	carry := 0

	for {
		n, err := r.Read(buf[carry:])

		active := carry + n

		if active >= l {
			total += int64(bytes.Count(buf[:active], delimiter))

			carry = l - 1

			copy(buf[:carry], buf[active-carry:])
		} else {
			carry = active
		}

		if err == io.EOF {
			break
		}

		if err != nil {
			return 0, err
		}
	}

	return total, nil
}
